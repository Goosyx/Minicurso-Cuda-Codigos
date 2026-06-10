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
        2) Preencher vetor auxiliar de tamanho tamanho_padded (potência de 2) com INT_MAX
        3) Copiar dados do host para o device (incluso no tempo medido)
        4) Executar os kernels de compare-and-swap para cada passo (tamanho_seq, j)
        5) Copiar resultado de volta para o host (incluso no tempo medido)
        6) Regravar o arquivo com os dados ordenados
        7) Registrar tempos em CSV

    DECISÃO DE DESIGN — Um kernel por passo (tamanho_seq, j):
        Cada kernel lança tamanho_padded threads, onde cada thread trata um par de índices
        (i, indice_par=i^j). A sincronização entre passos é garantida pela sequência
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
 * BitonicKernel: cada thread realiza um compare-and-swap para o par (i, indice_par=i^j).
 * Apenas a thread com o índice menor (i < indice_par) executa a troca, garantindo
 * que cada par seja tratado exatamente uma vez.
 */
__global__ void BitonicKernel(int *vetor, int tamanho_padded, int tamanho_seq, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tamanho_padded) return;

    int indice_par = i ^ j;
    if (indice_par > i)
    {
        bool crescente = (i & tamanho_seq) == 0;
        if ((crescente  && vetor[i] > vetor[indice_par]) ||
            (!crescente && vetor[i] < vetor[indice_par]))
        {
            int temp = vetor[i];
            vetor[i] = vetor[indice_par];
            vetor[indice_par] = temp;
        }
    }
}

// ============================================================
//          FUNÇÃO DE CONTROLE DO BITONIC SORT (GPU)
// ============================================================
/*
 * BitonicSortCuda: coordena os lançamentos de kernel para ordenar o vetor de tamanho_padded elementos.
 * O número de blocos é fixo para toda a execução pois tamanho_padded não muda.
 */
void BitonicSortCuda(int *dados, int tamanho_padded)
{
    int blocos = (tamanho_padded + BITONIC_THREADS_POR_BLOCO - 1) / BITONIC_THREADS_POR_BLOCO;

    for (int tamanho_seq = 2; tamanho_seq <= tamanho_padded; tamanho_seq <<= 1)
    {
        for (int j = tamanho_seq >> 1; j > 0; j >>= 1)
        {
            BitonicKernel<<<blocos, BITONIC_THREADS_POR_BLOCO>>>(dados, tamanho_padded, tamanho_seq, j);

            cudaError_t launchErr = cudaGetLastError();
            if (launchErr != cudaSuccess)
            {
                fprintf(stderr, "Erro de launch (tamanho_seq=%d, j=%d): %s\n", tamanho_seq, j, cudaGetErrorString(launchErr));
                return;
            }

            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess)
            {
                fprintf(stderr, "Erro de execução do kernel (tamanho_seq=%d, j=%d): %s\n", tamanho_seq, j, cudaGetErrorString(syncErr));
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
 * copia para o device, executa o sort e copia os num_elementos ordenados de volta.
 */
void HostParaDeviceBitonic(int *dados_host, int num_elementos)
{
    int tamanho_padded = 1;
    while (tamanho_padded < num_elementos) tamanho_padded <<= 1;

    cudaError_t err;

    // Aloca buffer pinned no host de tamanho tamanho_padded e preenche com INT_MAX
    int *vetor_host_padded = nullptr;
    err = cudaMallocHost(&vetor_host_padded, tamanho_padded * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar pinned memory no host (tamanho_padded=%d): %s\n", tamanho_padded, cudaGetErrorString(err));
        return;
    }
    for (int i = 0; i < num_elementos; i++) vetor_host_padded[i] = dados_host[i];
    for (int i = num_elementos; i < tamanho_padded; i++) vetor_host_padded[i] = INT_MAX;

    int *dados_device = nullptr;
    err = cudaMalloc(&dados_device, tamanho_padded * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao alocar dados_device (tamanho_padded=%d): %s\n", tamanho_padded, cudaGetErrorString(err));
        cudaFreeHost(vetor_host_padded);
        return;
    }

    err = cudaMemcpy(dados_device, vetor_host_padded, tamanho_padded * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Host->Device (tamanho_padded=%d): %s\n", tamanho_padded, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFreeHost(vetor_host_padded);
        return;
    }

    BitonicSortCuda(dados_device, tamanho_padded);

    err = cudaMemcpy(vetor_host_padded, dados_device, tamanho_padded * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Device->Host (tamanho_padded=%d): %s\n", tamanho_padded, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFreeHost(vetor_host_padded);
        return;
    }

    // Copia apenas os num_elementos originais (os INT_MAX ficaram no final)
    for (int i = 0; i < num_elementos; i++) dados_host[i] = vetor_host_padded[i];

    cudaFree(dados_device);
    cudaFreeHost(vetor_host_padded);
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

        int *vetor = nullptr;
        err = cudaMallocHost(&vetor, tamanho * sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro ao alocar pinned memory no host (N=%ld): %s\n", tamanho, cudaGetErrorString(err));
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
        HostParaDeviceBitonic(vetor, tamanho);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("BitonicSort CUDA - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "BitonicSort - CUDA,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        cudaFreeHost(vetor);
    }

    fclose(csv);
}
