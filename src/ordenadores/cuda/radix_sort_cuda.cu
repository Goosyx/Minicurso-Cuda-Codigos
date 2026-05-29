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
    Este arquivo implementa o algoritmo Radix Sort LSD paralelo na GPU via CUDA.

    DECISÃO DE DESIGN — Base 256 (1 byte por passagem):
        4 passagens fixas para inteiros de 32 bits, comparável entre os 3 modos.

    DECISÃO DE DESIGN — Estrutura de cada passagem:

        Fase 1 — ContagemKernel (GPU):
            Cada thread conta o byte do seu elemento e acumula via atomicAdd
            na linha do seu bloco na matriz contagens[num_blocos][256].

        Fase 2 — Prefix sum e offsets (CPU):
            A CPU transfere a matriz de contagens (D→H), calcula prefixo[256]
            e offsets[num_blocos][256], e os envia de volta (H→D).

        Fase 3 — DistribuicaoKernel (GPU, com shared memory):
            Cada bloco carrega sua fatia em shared memory e faz um prefix sum
            local (Hillis-Steele) para calcular o rank de cada elemento dentro
            do bloco em O(log P) em vez de O(P). A posição final é:
                prefixo[b] + offsets[bloco][b] + rank_local
            Sem loop linear — sem O(N²) implícito.

        Swap de ponteiros após cada passagem (sem cópia de dados).
        Após 4 passagens (par), resultado em dados_device original.

    OTIMIZAÇÃO — Shared memory no DistribuicaoKernel:
        A versão anterior calculava rank_local com um loop de até P iterações
        por thread lendo de global memory → complexidade O(N×P).
        A versão atual usa prefix sum em shared memory → O(N×log P).
        Para P=256: redução de 256 para 8 operações por thread (~32x).

    DECISÃO DE DESIGN — Threads por bloco como parâmetro:
        O número de threads por bloco é configurável via parâmetro em ExecRadixCuda,
        permitindo experimentar sem recompilar. Deve ser potência de 2 (32..1024).

    DECISÃO DE DESIGN — Pinned memory no host:
        cudaMallocHost garante transferências H↔D via DMA direto.
*/

// ============================================================
//              KERNEL DE CONTAGEM (FASE 1)
// ============================================================
/*
 * ContagemKernel: cada thread conta o byte do seu elemento e acumula
 * via atomicAdd na linha do seu bloco na matriz de contagens.
 */
__global__ void ContagemKernel(int *dados, int *contagens, int n, int shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int valor_byte = (dados[idx] >> shift) & 0xFF;
    atomicAdd(&contagens[blockIdx.x * 256 + valor_byte], 1);
}

// ============================================================
//              KERNEL DE PREFIX SUM DE OFFSETS (FASE 2)
// ============================================================
/*
 * PrefixSumKernel: para cada bucket b, calcula o prefix sum exclusivo de
 * contagens[0..num_blocos-1][b], produzindo offsets[bloco][b] e totais[b].
 *
 * Lançamento: <<<1, 256>>> — 1 bloco, 256 threads, uma por bucket.
 *
 * Acesso coalesced: em cada iteração do loop, os 256 threads acessam
 * contagens[bloco*256 + 0..255] — posições consecutivas na memória.
 *
 * Substitui a transferência D→H de contagens (~100MB) + cálculo de offsets
 * na CPU + transferência H→D de offsets (~100MB) por passagem.
 * Apenas totais[256] (1KB) é transferido D→H para calcular o prefixo global.
 */
__global__ void PrefixSumKernel(int *contagens, int *totais, int *offsets, int num_blocos)
{
    int b = threadIdx.x;
    if (b >= 256) return;

    int acc = 0;
    int bloco = 0;
    while (bloco < num_blocos)
    {
        offsets[bloco * 256 + b] = acc;
        acc += contagens[bloco * 256 + b];
        bloco++;
    }
    totais[b] = acc;
}

// ============================================================
//              KERNEL DE DISTRIBUIÇÃO (FASE 3)
// ============================================================
/*
 * DistribuicaoKernel: cada thread calcula seu rank_local via prefix sum
 * em shared memory e escreve seu elemento na posição correta do buffer.
 *
 * Parâmetros:
 * - dados:          vetor de entrada na GPU
 * - buffer:         buffer de saída na GPU
 * - prefixo:        array [256] — posição inicial de cada bucket no buffer
 * - offsets:        matriz [num_blocos][256] — offset acumulado por bloco/bucket
 * - n:              tamanho total do vetor
 * - shift:          byte sendo processado
 *
 * Algoritmo por bloco (em shared memory):
 *   1) Cada thread carrega seu elemento e extrai valor_byte
 *   2) Para cada bucket b (0..255):
 *        - Cada thread marca s_mask[t] = (valor_byte_da_thread == b) ? 1 : 0
 *        - Prefix sum exclusivo de s_mask via Hillis-Steele → s_scan[t]
 *        - s_scan[t] é o rank_local desta thread para o bucket b
 *   3) A thread escreve seu elemento usando rank correspondente ao seu bucket
 *
 * ATENÇÃO — por que iterar sobre todos os 256 buckets:
 *   Cada thread participa do prefix sum de todos os buckets, mas só usa
 *   o resultado do bucket correspondente ao seu próprio valor_byte.
 *   Isso é necessário pois __syncthreads() é uma barreira coletiva —
 *   todas as threads do bloco precisam participar de cada iteração.
 *
 * ATENÇÃO — shared memory necessária:
 *   s_dados[blockDim.x] + s_mask[blockDim.x] + s_scan[blockDim.x]
 *   = 3 × blockDim.x × sizeof(int) bytes por bloco.
 *   Alocado dinamicamente via 'extern __shared__' e passado em shmem_sz.
 */
__global__ void DistribuicaoKernel(int *dados, int *buffer,
                                   int *prefixo, int *offsets,
                                   int n, int shift)
{
    extern __shared__ int shmem[];

    int *s_dados = shmem;                          // [blockDim.x] elementos do bloco
    int *s_mask  = shmem + blockDim.x;             // [blockDim.x] máscara do bucket atual
    int *s_scan  = shmem + 2 * blockDim.x;         // [blockDim.x] prefix sum da máscara

    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int t     = threadIdx.x;
    int bloco = blockIdx.x;

    // Carrega o elemento desta thread (0 se fora dos limites)
    int meu_dado     = (idx < n) ? dados[idx] : -1;
    int valor_byte   = (idx < n) ? ((meu_dado >> shift) & 0xFF) : -1;
    s_dados[t]       = meu_dado;

    __syncthreads();

    int rank_local = 0;

    // Para cada bucket b, calcula o prefix sum exclusivo das threads
    // cujo valor_byte == b, obtendo o rank de cada thread dentro do bucket
    int b = 0;
    while (b < 256)
    {
        // Marca 1 se esta thread pertence ao bucket b
        s_mask[t] = (valor_byte == b && idx < n) ? 1 : 0;
        __syncthreads();

        // Prefix sum exclusivo (Hillis-Steele) em shared memory
        // Após o loop, s_scan[t] = número de threads com byte==b nos índices [0..t-1]
        s_scan[t] = s_mask[t];
        __syncthreads();

        int passo = 1;
        while (passo < (int)blockDim.x)
        {
            int val = (t >= passo) ? s_scan[t - passo] : 0;
            __syncthreads();
            s_scan[t] += val;
            __syncthreads();
            passo <<= 1;
        }

        // Converte para prefix sum EXCLUSIVO (rank = quantos antes de mim, não incluindo eu)
        int rank_b = (t > 0) ? s_scan[t - 1] : 0;
        __syncthreads();

        // Se esta thread pertence ao bucket b, guarda seu rank
        if (valor_byte == b && idx < n)
        {
            rank_local = rank_b;
        }

        b++;
    }

    // Escreve o elemento na posição final do buffer
    if (idx < n)
    {
        int pos_final = prefixo[valor_byte]
                      + offsets[bloco * 256 + valor_byte]
                      + rank_local;
        buffer[pos_final] = meu_dado;
    }
}

// ============================================================
//          FUNÇÃO DE CONTROLE DO RADIX SORT (GPU)
// ============================================================
/*
 * RadixSortCuda: coordena os kernels e o prefix sum para ordenar em 4 passagens.
 *
 * Parâmetros:
 * - dados_device:        ponteiro-para-ponteiro do vetor de entrada na GPU
 * - buffer_device:       ponteiro-para-ponteiro do buffer de saída na GPU
 * - n:                   tamanho do vetor
 * - num_blocos:          número de blocos CUDA
 * - threads_por_bloco:   número de threads por bloco (potência de 2, 32..1024)
 */
void RadixSortCuda(int **dados_device, int **buffer_device,
                   int n, int num_blocos, int threads_por_bloco)
{
    cudaError_t err;

    int *contagens_device = nullptr;
    err = cudaMalloc(&contagens_device, num_blocos * 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar contagens_device: %s\n", cudaGetErrorString(err));
        return;
    }

    int *prefixo_device = nullptr;
    err = cudaMalloc(&prefixo_device, 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar prefixo_device: %s\n", cudaGetErrorString(err));
        cudaFree(contagens_device);
        return;
    }

    int *offsets_device = nullptr;
    err = cudaMalloc(&offsets_device, num_blocos * 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar offsets_device: %s\n", cudaGetErrorString(err));
        cudaFree(contagens_device);
        cudaFree(prefixo_device);
        return;
    }

    int *totais_device = nullptr;
    err = cudaMalloc(&totais_device, 256 * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar totais_device: %s\n", cudaGetErrorString(err));
        cudaFree(contagens_device);
        cudaFree(prefixo_device);
        cudaFree(offsets_device);
        return;
    }

    int totais_host[256];
    int prefixo_host[256];

    // Shared memory por bloco: 3 arrays de blockDim.x inteiros
    size_t shmem_sz = 3 * threads_por_bloco * sizeof(int);

    int shift = 0;
    while (shift < 32)
    {
        // ── Fase 1: Contagem ─────────────────────────────────────────────────
        cudaMemset(contagens_device, 0, num_blocos * 256 * sizeof(int));

        ContagemKernel<<<num_blocos, threads_por_bloco>>>(
            *dados_device, contagens_device, n, shift);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no ContagemKernel (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        // ── Fase 2: Prefix sum de offsets na GPU ─────────────────────────────
        // PrefixSumKernel calcula offsets[num_blocos×256] inteiramente na GPU.
        // Apenas totais[256] (1KB) é transferido D→H para o prefixo global.
        PrefixSumKernel<<<1, 256>>>(contagens_device, totais_device, offsets_device, num_blocos);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no PrefixSumKernel (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(totais_host, totais_device, 256 * sizeof(int), cudaMemcpyDeviceToHost);

        prefixo_host[0] = 0;
        int b = 1;
        while (b < 256)
        {
            prefixo_host[b] = prefixo_host[b - 1] + totais_host[b - 1];
            b++;
        }

        cudaMemcpy(prefixo_device, prefixo_host, 256 * sizeof(int), cudaMemcpyHostToDevice);

        // ── Fase 3: Distribuição (GPU, shared memory) ────────────────────────
        DistribuicaoKernel<<<num_blocos, threads_por_bloco, shmem_sz>>>(
            *dados_device, *buffer_device,
            prefixo_device, offsets_device,
            n, shift);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no DistribuicaoKernel (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        // ── Swap de ponteiros ────────────────────────────────────────────────
        int *temp      = *dados_device;
        *dados_device  = *buffer_device;
        *buffer_device = temp;

        shift += 8;
    }

    cudaFree(contagens_device);
    cudaFree(prefixo_device);
    cudaFree(offsets_device);
    cudaFree(totais_device);
}

// ============================================================
//    FUNÇÃO HostParaDeviceRadix — GERENCIA CÓPIAS HOST/DEVICE
// ============================================================
void HostParaDeviceRadix(int *dados_host, int n, int threads_por_bloco)
{
    cudaError_t err;

    int num_blocos = (n + threads_por_bloco - 1) / threads_por_bloco;

    int *dados_device  = nullptr;
    int *buffer_device = nullptr;

    err = cudaMalloc(&dados_device, n * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar dados_device (n=%d): %s\n", n, cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&buffer_device, n * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar buffer_device (n=%d): %s\n", n, cudaGetErrorString(err));
        cudaFree(dados_device);
        return;
    }

    err = cudaMemcpy(dados_device, dados_host, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Host->Device (n=%d): %s\n", n, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFree(buffer_device);
        return;
    }

    RadixSortCuda(&dados_device, &buffer_device, n, num_blocos, threads_por_bloco);

    // Após 4 passagens (par), dados_device contém o resultado ordenado
    err = cudaMemcpy(dados_host, dados_device, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Device->Host (n=%d): %s\n", n, cudaGetErrorString(err));
    }

    cudaFree(dados_device);
    cudaFree(buffer_device);
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
/*
 * ExecRadixCuda: executa o Radix Sort CUDA para múltiplos arquivos binários.
 *
 * Parâmetros:
 * - entradas:          array de caminhos para arquivos binários
 * - num_entradas:      número de arquivos
 * - threads_por_bloco: threads por bloco CUDA (potência de 2, ex: 128, 256, 512)
 * - csv_saida:         caminho do arquivo CSV de saída
 */
void ExecRadixCuda(const char **entradas, int num_entradas,
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
            fprintf(stderr, "Erro ao alocar pinned memory (N=%ld): %s\n",
                    tamanho, cudaGetErrorString(err));
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
        HostParaDeviceRadix(vetor, tamanho, threads_por_bloco);
        auto fim = chrono::high_resolution_clock::now();

        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("Radix Sort CUDA - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "RadixSort - CUDA,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        cudaFreeHost(vetor);
    }

    fclose(csv);
}