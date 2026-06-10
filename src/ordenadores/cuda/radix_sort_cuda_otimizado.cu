// Baseado em: Merrill & Grimshaw (2010) "Revisiting Sorting for GPGPU Stream Architectures"
// https://dl.acm.org/doi/10.1145/1854273.1854344
// e NVIDIA CUDA Warp-Level Primitives:
// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// ============================================================
//                  Observações gerais
// ============================================================
/*
    Radix Sort LSD otimizado na GPU via CUDA, base 256 (1 byte por passagem).

    A estrutura de passagens é a mesma do radix_sort_cuda.cu (Contagem →
    PrefixSum → Distribuição). As otimizações estão em dois pontos:

    OTIMIZAÇÃO 1 — ContagemKernelOtim (shared memory atomics):
        Em vez de incrementar diretamente em global memory (que sofre alta
        latência e contenção entre threads do mesmo bloco), cada bloco mantém
        um histograma temporário em shared memory. Shared memory atomics são
        ~30x mais rápidos que global memory atomics para contenção moderada.
        Após contar localmente, o histograma é copiado para global memory.

    OTIMIZAÇÃO 2 — DistribuicaoKernelOtim (warp ballot + prefix inter-warp):
        Em vez de thread 0 distribuir sequencialmente (mantém apenas 1 thread
        ativa por bloco na fase de distribuição), todas as threads calculam
        seu rank usando primitivas de nível de warp:

        - __ballot_sync(mask, pred): retorna um inteiro de 32 bits onde o bit
          i indica se a thread i do warp satisfez o predicado 'pred'. Isso
          revela quais threads do warp têm o mesmo byte que eu.

        - __popc(mask & lane_mask_lt): conta quantas threads com índice de
          lane MENOR que o meu têm o mesmo byte → rank dentro do warp.

        Depois, um prefix sum entre os warps (em shared memory, com apenas
        num_warps elementos) dá o offset do meu warp dentro do bucket.

        Resultado: em vez de 256 iterações com __syncthreads() (Hillis-Steele),
        usamos 256 chamadas a __ballot_sync (nível de warp, sem sync) +
        1 __syncthreads() + soma de no máximo 32 valores em shared memory.

    REFERÊNCIA — técnicas específicas utilizadas:
        - Merrill & Grimshaw (2010), Seção 4: "Warp-level Scan with ballot"
        - NVIDIA blog: "Using CUDA Warp-Level Primitives" (__ballot_sync,
          __popc para rank computation em radix sort)
*/

// ============================================================
//              KERNEL DE CONTAGEM OTIMIZADO (FASE 1)
// ============================================================
/*
 * ContagemKernelOtim: usa shared memory atomics para construir o histograma
 * local do bloco em shared memory, depois copia para contagens[bloco][256].
 *
 * Shared memory: 256 ints (1KB fixo por bloco, independente de blockDim.x).
 */
__global__ void ContagemKernelOtim(int *dados, int *contagens, int num_elementos, int shift)
{
    __shared__ int s_hist[256];

    int id_thread = threadIdx.x;
    int idx       = blockIdx.x * blockDim.x + id_thread;

    // Zera histograma local em shared memory
    for (int i = id_thread; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // Cada thread incrementa em shared memory (muito mais rápido que global)
    if (idx < num_elementos)
        atomicAdd(&s_hist[(dados[idx] >> shift) & 0xFF], 1);
    __syncthreads();

    // Copia histograma local para global memory
    for (int i = id_thread; i < 256; i += blockDim.x)
        contagens[blockIdx.x * 256 + i] = s_hist[i];
}

// ============================================================
//              KERNEL DE PREFIX SUM DE OFFSETS (FASE 2)
// ============================================================
/*
 * PrefixSumKernelOtim: idêntico ao PrefixSumKernel do arquivo simples.
 * Lançamento: <<<1, 256>>> — 1 bloco, 256 threads, uma por bucket.
 */
__global__ void PrefixSumKernelOtim(int *contagens, int *totais, int *offsets, int num_blocos)
{
    int bucket = threadIdx.x;
    if (bucket >= 256) return;

    int acumulador = 0;
    for (int bloco = 0; bloco < num_blocos; bloco++)
    {
        offsets[bloco * 256 + bucket] = acumulador;
        acumulador += contagens[bloco * 256 + bucket];
    }
    totais[bucket] = acumulador;
}

// ============================================================
//              KERNEL DE DISTRIBUIÇÃO OTIMIZADO (FASE 3)
// ============================================================
/*
 * DistribuicaoKernelOtim: cada thread calcula seu rank dentro do bloco
 * usando warp ballot e um prefix sum inter-warp em shared memory.
 *
 * ETAPA 1 — Rank dentro do warp (sem __syncthreads):
 *   Para cada bucket (0..255):
 *     mask = __ballot_sync(FULL, valor_byte == bucket)
 *       → bit i do mask = 1 se thread i do warp tem byte == bucket
 *     warp_rank = __popc(mask & lane_mask_lt)
 *       → quantas threads com lane < meu lane têm o mesmo byte bucket
 *
 * ETAPA 2 — Offset inter-warp (1 __syncthreads):
 *   Lane 0 de cada warp escreve em s_hist[warp][bucket] quantas threads do
 *   seu warp têm byte == bucket (__popc(mask)).
 *   Após __syncthreads, cada thread soma s_hist[0..warp-1][valor_byte]
 *   para saber quantos elementos do bucket chegaram antes do seu warp.
 *
 * Posição final = prefixo[bucket] + offsets[bloco][bucket] + inter_warp + warp_rank
 *
 * Shared memory: (num_warps × 256) ints — para P=1024: 32×256×4 = 32KB.
 */
__global__ void DistribuicaoKernelOtim(int *dados, int *buffer,
                                       int *prefixo, int *offsets,
                                       int num_elementos, int shift)
{
    // s_hist[warp][256]: histograma de cada warp por bucket
    extern __shared__ int s_hist[];

    int id_thread = threadIdx.x;
    int idx       = blockIdx.x * blockDim.x + id_thread;
    int lane      = id_thread & 31;          // posição dentro do warp (0..31)
    int warp      = id_thread >> 5;          // índice do warp dentro do bloco
    int num_warps = blockDim.x >> 5;         // número de warps no bloco

    // lane_mask_lt: máscara com bits 1 para lanes com índice < lane
    unsigned int lane_mask_lt = (1u << lane) - 1;

    int meu_dado   = (idx < num_elementos) ? dados[idx] : 0;
    int valor_byte = (idx < num_elementos) ? ((meu_dado >> shift) & 0xFF) : -1;

    // Zera histograma de warps em shared memory
    for (int i = id_thread; i < num_warps * 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // ── Etapa 1: rank dentro do warp via __ballot_sync ────────────────────
    int warp_rank = 0;

    for (int bucket = 0; bucket < 256; bucket++)
    {
        // Quais threads do warp têm byte == bucket?
        unsigned int mask = __ballot_sync(0xFFFFFFFF, valor_byte == bucket);

        if (valor_byte == bucket)
            warp_rank = __popc(mask & lane_mask_lt);

        // Lane 0 registra a contagem deste warp para o bucket
        if (lane == 0)
            s_hist[warp * 256 + bucket] = __popc(mask);
    }
    __syncthreads();

    // ── Etapa 2: offset inter-warp em shared memory ───────────────────────
    int inter_warp = 0;
    if (idx < num_elementos)
    {
        for (int idx_warp = 0; idx_warp < warp; idx_warp++)
            inter_warp += s_hist[idx_warp * 256 + valor_byte];
    }

    // ── Escrita na posição final ──────────────────────────────────────────
    if (idx < num_elementos)
    {
        int pos = prefixo[valor_byte]
                + offsets[blockIdx.x * 256 + valor_byte]
                + inter_warp
                + warp_rank;
        buffer[pos] = meu_dado;
    }
}

// ============================================================
//       FUNÇÃO DE CONTROLE DO RADIX SORT OTIMIZADO (GPU)
// ============================================================
void RadixSortCudaOtim(int **dados_device, int **buffer_device,
                       int num_elementos, int num_blocos, int threads_por_bloco)
{
    cudaError_t err;

    int *contagens_device = nullptr;
    err = cudaMalloc(&contagens_device, num_blocos * 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar contagens_device: %s\n", cudaGetErrorString(err));
        return;
    }

    int *offsets_device = nullptr;
    err = cudaMalloc(&offsets_device, num_blocos * 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar offsets_device: %s\n", cudaGetErrorString(err));
        cudaFree(contagens_device);
        return;
    }

    int *prefixo_device = nullptr;
    err = cudaMalloc(&prefixo_device, 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar prefixo_device: %s\n", cudaGetErrorString(err));
        cudaFree(contagens_device);
        cudaFree(offsets_device);
        return;
    }

    int *totais_device = nullptr;
    err = cudaMalloc(&totais_device, 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar totais_device: %s\n", cudaGetErrorString(err));
        cudaFree(contagens_device);
        cudaFree(offsets_device);
        cudaFree(prefixo_device);
        return;
    }

    int totais_host[256];
    int prefixo_host[256];

    // Shared memory do DistribuicaoKernelOtim: num_warps × 256 ints
    int num_warps    = threads_por_bloco / 32;
    size_t shmem_sz  = num_warps * 256 * sizeof(int);

    for (int shift = 0; shift < 32; shift += 8)
    {
        // ── Fase 1: Contagem (shared memory atomics) ──────────────────────────
        cudaMemset(contagens_device, 0, num_blocos * 256 * sizeof(int));

        ContagemKernelOtim<<<num_blocos, threads_por_bloco>>>(
            *dados_device, contagens_device, num_elementos, shift);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no ContagemKernelOtim (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        // ── Fase 2: Prefix sum na GPU + prefixo global na CPU (1KB) ──────────
        PrefixSumKernelOtim<<<1, 256>>>(contagens_device, totais_device, offsets_device, num_blocos);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no PrefixSumKernelOtim (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(totais_host, totais_device, 256 * sizeof(int), cudaMemcpyDeviceToHost);

        prefixo_host[0] = 0;
        for (int bucket = 1; bucket < 256; bucket++)
            prefixo_host[bucket] = prefixo_host[bucket - 1] + totais_host[bucket - 1];

        cudaMemcpy(prefixo_device, prefixo_host, 256 * sizeof(int), cudaMemcpyHostToDevice);

        // ── Fase 3: Distribuição (warp ballot + inter-warp prefix) ───────────
        DistribuicaoKernelOtim<<<num_blocos, threads_por_bloco, shmem_sz>>>(
            *dados_device, *buffer_device,
            prefixo_device, offsets_device,
            num_elementos, shift);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no DistribuicaoKernelOtim (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        // ── Swap de ponteiros ────────────────────────────────────────────────
        int *temp      = *dados_device;
        *dados_device  = *buffer_device;
        *buffer_device = temp;
    }

    cudaFree(contagens_device);
    cudaFree(offsets_device);
    cudaFree(prefixo_device);
    cudaFree(totais_device);
}

// ============================================================
//    FUNÇÃO HostParaDeviceRadixOtim — GERENCIA CÓPIAS HOST/DEVICE
// ============================================================
void HostParaDeviceRadixOtim(int *dados_host, int num_elementos, int threads_por_bloco)
{
    cudaError_t err;

    int num_blocos = (num_elementos + threads_por_bloco - 1) / threads_por_bloco;

    int *dados_device  = nullptr;
    int *buffer_device = nullptr;

    err = cudaMalloc(&dados_device, num_elementos * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar dados_device (n=%d): %s\n", num_elementos, cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&buffer_device, num_elementos * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar buffer_device (n=%d): %s\n", num_elementos, cudaGetErrorString(err));
        cudaFree(dados_device);
        return;
    }

    err = cudaMemcpy(dados_device, dados_host, num_elementos * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Host->Device (n=%d): %s\n", num_elementos, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFree(buffer_device);
        return;
    }

    RadixSortCudaOtim(&dados_device, &buffer_device, num_elementos, num_blocos, threads_por_bloco);

    err = cudaMemcpy(dados_host, dados_device, num_elementos * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        fprintf(stderr, "Erro em cudaMemcpy Device->Host (n=%d): %s\n", num_elementos, cudaGetErrorString(err));

    cudaFree(dados_device);
    cudaFree(buffer_device);
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
void ExecRadixCudaOtim(const char **entradas, int num_entradas,
                       int threads_por_bloco, const char *csv_saida)
{
    FILE *csv = AbrirCSV(csv_saida);
    if (!csv) return;

    cudaError_t err;

    for (int i = 0; i < num_entradas; i++)
    {
        FILE *file = fopen(entradas[i], "rb+");
        if (!file) { perror(entradas[i]); continue; }

        fseek(file, 0, SEEK_END);
        long tamanho = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        int *vetor = nullptr;
        err = cudaMallocHost(&vetor, tamanho * sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro ao alocar pinned memory (N=%ld): %s\n", tamanho, cudaGetErrorString(err));
            fclose(file);
            continue;
        }

        if (fread(vetor, sizeof(int), tamanho, file) != (size_t)tamanho)
        {
            perror("Erro ao ler o arquivo");
            fclose(file);
            cudaFreeHost(vetor);
            continue;
        }

        auto inicio = chrono::high_resolution_clock::now();
        HostParaDeviceRadixOtim(vetor, tamanho, threads_por_bloco);
        auto fim = chrono::high_resolution_clock::now();

        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("Radix Sort CUDA Otimizado - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "RadixSort - CUDA Otimizado,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        cudaFreeHost(vetor);
    }

    fclose(csv);
}
