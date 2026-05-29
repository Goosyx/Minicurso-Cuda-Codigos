// ============================================================
//                  Observações gerais
// ============================================================
/*
    Este é o arquivo principal do projeto, responsável por orquestrar a execução
    dos algoritmos de ordenação (Merge Sort, Radix Sort e Bitonic Sort) em diferentes
    versões: sequencial, com threads POSIX e com CUDA (GPU).

    O fluxo geral é:
        1. Define os arquivos de entrada e seus tamanhos
        2. Gera os arquivos binários com dados aleatórios
        3. Executa cada versão dos algoritmos de ordenação
        4. Mede e registra os tempos de execução em um arquivo CSV
        5. Verifica se os arquivos foram ordenados corretamente após cada execução
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "utils/utils.h"
#include "ordenadores/sequencial/merge_sort_seq.h"
#include "ordenadores/threads/merge_sort_threads.h"
#include "ordenadores/cuda/merge_sort_cuda.cu"
#include "ordenadores/sequencial/radix_sort_seq.h"
#include "ordenadores/threads/radix_sort_threads.h"
#include "ordenadores/cuda/radix_sort_cuda.cu"
#include "ordenadores/cuda/radix_sort_cuda_otimizado.cu"
#include "ordenadores/sequencial/bitonic_sort_seq.h"
#include "ordenadores/threads/bitonic_sort_threads.h"
#include "ordenadores/cuda/bitonic_sort_cuda.cu"

// ============================================================
//                  Função principal
// ============================================================
int main()
{
    // ── Parâmetros de execução ───────────────────────────────────────────────

    // Número de threads para os modos paralelos em CPU (threads POSIX)
    const int num_threads = 8;

    // Número de threads por bloco CUDA para o Radix Sort GPU.
    // Deve ser potência de 2 entre 32 e 1024.
    // Valores sugeridos para experimentação: 64, 128, 256, 512, 1024.
    // Impacto: blocos maiores reduzem num_blocos e o tamanho da matriz de
    // contagens, diminuindo o trabalho do PrefixSumKernel.
    // Shared memory usada: 3 × threads_por_bloco_cuda × 4 bytes por bloco.
    // Com 1024: 12KB/bloco — dentro do limite de 48KB do sm_86.
    const int threads_por_bloco_cuda = 1024;

    // ── Arquivos de entrada ──────────────────────────────────────────────────

    const int num_entradas = 11;

    const long tamanho_arquivos[num_entradas] = {
        250000, 500000, 750000, 1000000,
        2500000, 5000000, 7500000, 10000000,
        25000000, 50000000, 100000000
    };

    const char *entradas[num_entradas] = {
        "dados/250k.bin", "dados/500k.bin", "dados/750k.bin", "dados/1m.bin",
        "dados/2m500.bin", "dados/5m.bin", "dados/7m500.bin", "dados/10m.bin",
        "dados/25m.bin", "dados/50m.bin", "dados/100m.bin"
    };

    // ── Inicializa CSV ───────────────────────────────────────────────────────
    FILE *csv = fopen("results/tempos.csv", "w");
    if (!csv) { perror("Erro ao abrir arquivo CSV para escrita"); return 1; }
    fprintf(csv, "Algoritmo,Tamanho,Tempo\n");
    fclose(csv);

    /*
        Para cada algoritmo:
            - gera os arquivos binários com dados aleatórios
            - executa a ordenação
            - verifica se os arquivos estão ordenados corretamente
    */

    // Merge Sort Sequencial
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecMergeSeq(entradas, num_entradas, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Merge Sort com Threads
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecMergeThread(entradas, num_entradas, num_threads, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Merge Sort com CUDA (GPU)
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecMergeCuda(entradas, num_entradas, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Radix Sort Sequencial
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecRadixSeq(entradas, num_entradas, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Radix Sort com Threads
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecRadixThread(entradas, num_entradas, num_threads, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Radix Sort com CUDA (GPU)
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecRadixCuda(entradas, num_entradas, threads_por_bloco_cuda, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Radix Sort com CUDA Otimizado (warp ballot + shared memory atomics)
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecRadixCudaOtim(entradas, num_entradas, threads_por_bloco_cuda, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Bitonic Sort Sequencial
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecBitonicSeq(entradas, num_entradas, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Bitonic Sort com Threads
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecBitonicThread(entradas, num_entradas, num_threads, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    // Bitonic Sort com CUDA (GPU)
    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecBitonicCuda(entradas, num_entradas, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    return 0;
}