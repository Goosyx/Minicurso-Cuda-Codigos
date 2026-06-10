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
    Radix Sort LSD paralelo na GPU via CUDA, base 256 (1 byte por passagem).

    Cada passagem segue a mesma estrutura dos modos sequencial e threads:

        Fase 1 — ContagemKernel (GPU):
            Cada thread incrementa via atomicAdd em contagens[bloco][256].
            Cada bloco escreve apenas na sua própria linha — sem contenção
            entre blocos. Equivalente à "contagem local" de cada thread.

        Fase 2 — PrefixSumKernel (GPU) + prefixo global (CPU):
            Um único bloco de 256 threads percorre contagens[*][bucket] e produz
            offsets[bloco][bucket] (posição do bloco dentro do bucket).
            Apenas totais[256] (1KB) vai para a CPU calcular prefixo[256]
            e volta. Equivalente ao "Thread 0 faz prefix sum".

        Fase 3 — DistribuicaoKernel (GPU):
            Todos os threads carregam sua faixa em shared memory (leitura
            coalescida). A thread 0 distribui os elementos em ordem, de forma
            estável. O paralelismo vem dos ~100k blocos simultâneos na GPU.
            Equivalente à "distribuição" de cada thread no modo threads.

        Swap de ponteiros após cada passagem.
        Após 4 passagens (par), resultado em dados_device original.

    DECISÃO DE DESIGN — Pinned memory no host:
        cudaMallocHost garante transferências H↔D via DMA direto.

    DECISÃO DE DESIGN — Threads por bloco como parâmetro:
        Configurável via parâmetro em ExecRadixCuda (potência de 2, 32..1024).
*/

// ============================================================
//              KERNEL DE CONTAGEM (FASE 1)
// ============================================================
/*
 * ContagemKernel: cada thread incrementa via atomicAdd o contador do seu
 * byte na linha do seu bloco em contagens[bloco][256].
 */
__global__ void ContagemKernel(int *dados, int *contagens, int num_elementos, int shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elementos) return;
    atomicAdd(&contagens[blockIdx.x * 256 + ((dados[idx] >> shift) & 0xFF)], 1);
}

// ============================================================
//              KERNEL DE PREFIX SUM DE OFFSETS (FASE 2)
// ============================================================
/*
 * PrefixSumKernel: para cada bucket, percorre todos os blocos e calcula
 * offsets[bloco][bucket] (prefix sum exclusivo) e totais[bucket] (contagem global).
 *
 * Lançamento: <<<1, 256>>> — 1 bloco, 256 threads, uma por bucket.
 */
__global__ void PrefixSumKernel(int *contagens, int *totais, int *offsets, int num_blocos)
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
//              KERNEL DE DISTRIBUIÇÃO (FASE 3)
// ============================================================
/*
 * DistribuicaoKernel: todos os threads carregam seus dados em shared memory
 * (leitura coalescida). A thread 0 distribui os elementos em ordem, garantindo
 * estabilidade. Cada bloco escreve em sub-regiões exclusivas do buffer —
 * sem condição de corrida entre blocos.
 *
 * Shared memory: blockDim.x ints (dados) + 256 ints (posições de escrita).
 */
__global__ void DistribuicaoKernel(int *dados, int *buffer,
                                   int *prefixo, int *offsets,
                                   int num_elementos, int shift)
{
    extern __shared__ int shmem[];
    int *s_dados = shmem;               // [blockDim.x]
    int *s_pos   = shmem + blockDim.x;  // [256]

    int id_thread = threadIdx.x;
    int idx       = blockIdx.x * blockDim.x + id_thread;

    // Todos os threads carregam dados em shared memory (leitura coalescida)
    s_dados[id_thread] = (idx < num_elementos) ? dados[idx] : 0;

    // Carrega posição inicial de cada bucket para este bloco
    for (int i = id_thread; i < 256; i += blockDim.x)
        s_pos[i] = prefixo[i] + offsets[blockIdx.x * 256 + i];
    __syncthreads();

    // Thread 0 distribui os elementos desta faixa em ordem (estável)
    if (id_thread == 0)
    {
        int limite = min((int)blockDim.x, num_elementos - (int)(blockIdx.x * blockDim.x));
        for (int i = 0; i < limite; i++)
        {
            int elemento = s_dados[i];
            int bucket   = (elemento >> shift) & 0xFF;
            buffer[s_pos[bucket]++] = elemento;
        }
    }
}

// ============================================================
//          FUNÇÃO DE CONTROLE DO RADIX SORT (GPU)
// ============================================================
void RadixSortCuda(int **dados_device, int **buffer_device,
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

    // Shared memory do DistribuicaoKernel: dados (blockDim.x) + posições (256)
    size_t shmem_sz = (threads_por_bloco + 256) * sizeof(int);

    for (int shift = 0; shift < 32; shift += 8)
    {
        // ── Fase 1: Contagem ─────────────────────────────────────────────────
        cudaMemset(contagens_device, 0, num_blocos * 256 * sizeof(int));

        ContagemKernel<<<num_blocos, threads_por_bloco>>>(
            *dados_device, contagens_device, num_elementos, shift);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no ContagemKernel (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        // ── Fase 2: Prefix sum de offsets na GPU + prefixo global na CPU ─────
        PrefixSumKernel<<<1, 256>>>(contagens_device, totais_device, offsets_device, num_blocos);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro no PrefixSumKernel (shift=%d): %s\n", shift, cudaGetErrorString(err));
            break;
        }
        cudaDeviceSynchronize();

        // Apenas 1KB (totais[256]) atravessa o barramento PCIe por passagem
        cudaMemcpy(totais_host, totais_device, 256 * sizeof(int), cudaMemcpyDeviceToHost);

        prefixo_host[0] = 0;
        for (int bucket = 1; bucket < 256; bucket++)
            prefixo_host[bucket] = prefixo_host[bucket - 1] + totais_host[bucket - 1];

        cudaMemcpy(prefixo_device, prefixo_host, 256 * sizeof(int), cudaMemcpyHostToDevice);

        // ── Fase 3: Distribuição ─────────────────────────────────────────────
        DistribuicaoKernel<<<num_blocos, threads_por_bloco, shmem_sz>>>(
            *dados_device, *buffer_device,
            prefixo_device, offsets_device,
            num_elementos, shift);

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
    }

    cudaFree(contagens_device);
    cudaFree(offsets_device);
    cudaFree(prefixo_device);
    cudaFree(totais_device);
}

// ============================================================
//    FUNÇÃO HostParaDeviceRadix — GERENCIA CÓPIAS HOST/DEVICE
// ============================================================
void HostParaDeviceRadix(int *dados_host, int num_elementos, int threads_por_bloco)
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

    RadixSortCuda(&dados_device, &buffer_device, num_elementos, num_blocos, threads_por_bloco);

    // Após 4 passagens (par), dados_device contém o resultado ordenado
    err = cudaMemcpy(dados_host, dados_device, num_elementos * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        fprintf(stderr, "Erro em cudaMemcpy Device->Host (n=%d): %s\n", num_elementos, cudaGetErrorString(err));

    cudaFree(dados_device);
    cudaFree(buffer_device);
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
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
