#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define BITONIC_THREADS_POR_BLOCO 256

using namespace std;

// ============================================================
//                  Observações gerais
// ============================================================
/*
    Este arquivo implementa o Bitonic Sort na GPU usando CUDA.
    O fluxo geral é:
        1) Ler vetores de arquivos binários (int)
        2) Preencher vetor auxiliar de tamanho P (potência de 2) com INT_MAX
        3) Copiar dados do host para o device (incluso no tempo medido)
        4) Executar os kernels de compare-and-swap para cada passo (k, j)
        5) Copiar resultado de volta para o host (incluso no tempo medido)
        6) Regravar o arquivo com os dados ordenados
        7) Registrar tempos em CSV

    DECISÃO DE DESIGN — Um kernel por passo (k, j):
        Cada kernel lança P threads, onde cada thread trata um par de índices
        (i, l=i^j). A sincronização entre passos é garantida pela sequência
        de lançamento na CPU + cudaDeviceSynchronize, sem necessidade de
        shared memory ou sincronização intra-kernel.

    DECISÃO DE DESIGN — Preenchimento com INT_MAX no host:
        O vetor é preenchido com INT_MAX antes de copiar para o device para
        evitar um kernel extra de inicialização. Os valores INT_MAX ficam
        automaticamente no final após a ordenação ascendente.

    DECISÃO DE DESIGN — Tempo inclui H→D + kernels + D→H:
        Reflete o custo real de uso da GPU para a aplicação.
*/

// ============================================================
//                  KERNEL DO BITONIC SORT (GPU)
// ============================================================
/*
 * BitonicKernel: cada thread realiza um compare-and-swap para o par (i, l=i^j).
 * Apenas a thread com o índice menor (i < l) executa a troca, garantindo
 * que cada par seja tratado exatamente uma vez.
 */
__global__ void BitonicKernel(int *vetor, int P, int k, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;

    int l = i ^ j;
    if (l > i)
    {
        bool crescente = (i & k) == 0;
        if ((crescente && vetor[i] > vetor[l]) || (!crescente && vetor[i] < vetor[l]))
        {
            int tmp = vetor[i];
            vetor[i] = vetor[l];
            vetor[l] = tmp;
        }
    }
}

// ============================================================
//          FUNÇÃO DE CONTROLE DO BITONIC SORT (GPU)
// ============================================================
/*
 * BitonicSortCuda: coordena os lançamentos de kernel para ordenar o vetor de P elementos.
 * O número de blocos é fixo para toda a execução pois P não muda.
 */
void BitonicSortCuda(int *dados, int P)
{
    int blocos = (P + BITONIC_THREADS_POR_BLOCO - 1) / BITONIC_THREADS_POR_BLOCO;

    for (int k = 2; k <= P; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            BitonicKernel<<<blocos, BITONIC_THREADS_POR_BLOCO>>>(dados, P, k, j);

            cudaError_t launchErr = cudaGetLastError();
            if (launchErr != cudaSuccess)
            {
                fprintf(stderr, "Erro de launch (k=%d, j=%d): %s\n", k, j, cudaGetErrorString(launchErr));
                return;
            }

            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                fprintf(stderr, "Erro de execução do kernel (k=%d, j=%d): %s\n", k, j, cudaGetErrorString(syncErr));
                return;
            }
        }
    }
}

// ============================================================
//    FUNÇÃO HostParaDeviceBitonic() - GERENCIA CÓPIAS HOST/DEVICE
// ============================================================
/*
 * HostParaDeviceBitonic: aloca e preenche vetor padded no host (pinned),
 * copia para o device, executa o sort e copia os n elementos ordenados de volta.
 */
void HostParaDeviceBitonic(int *dados_host, int n)
{
    int P = 1;
    while (P < n) P <<= 1;

    cudaError_t err;

    // Aloca buffer pinned no host de tamanho P e preenche com INT_MAX
    int *v_host = nullptr;
    err = cudaMallocHost(&v_host, P * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar pinned memory no host (P=%d): %s\n", P, cudaGetErrorString(err));
        return;
    }
    for (int i = 0; i < n; i++) v_host[i] = dados_host[i];
    for (int i = n; i < P; i++) v_host[i] = INT_MAX;

    int *dados_device = nullptr;
    err = cudaMalloc(&dados_device, P * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar dados_device (P=%d): %s\n", P, cudaGetErrorString(err));
        cudaFreeHost(v_host);
        return;
    }

    err = cudaMemcpy(dados_device, v_host, P * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Host->Device (P=%d): %s\n", P, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFreeHost(v_host);
        return;
    }

    BitonicSortCuda(dados_device, P);

    err = cudaMemcpy(v_host, dados_device, P * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Device->Host (P=%d): %s\n", P, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFreeHost(v_host);
        return;
    }

    // Copia apenas os n elementos originais (os INT_MAX ficaram no final)
    for (int i = 0; i < n; i++) dados_host[i] = v_host[i];

    cudaFree(dados_device);
    cudaFreeHost(v_host);
}

// ============================================================
//             FUNÇÕES AUXILIARES DE ENTRADA/SAÍDA
// ============================================================

void ExecBitonicCuda(const char **entradas, int num_entradas, const char *csv_saida)
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

        int *v = nullptr;
        err = cudaMallocHost(&v, tamanho * sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro ao alocar pinned memory no host (N=%ld): %s\n", tamanho, cudaGetErrorString(err));
            fclose(file);
            continue;
        }

        if (fread(v, sizeof(int), tamanho, file) != (size_t)tamanho)
        {
            perror("Erro ao ler o arquivo");
            fclose(file);
            cudaFreeHost(v);
            continue;
        }

        auto inicio = chrono::high_resolution_clock::now();
        HostParaDeviceBitonic(v, tamanho);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("BitonicSort CUDA - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "BitonicSort - CUDA,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, v, tamanho);
        cudaFreeHost(v);
    }

    fclose(csv);
}
