#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <time.h>
#include <chrono>
#include <pthread.h>

using namespace std;

// ============================================================
//                  Observações gerais
// ============================================================
/*
    Este arquivo implementa o Bitonic Sort paralelo com threads POSIX.

    DECISÃO DE DESIGN — Threads persistentes com barreira:
        Cada thread executa o mesmo laço (k, j) da versão sequencial, mas
        opera apenas na sua faixa de índices [inicio, fim). Uma barreira
        (pthread_barrier_t) sincroniza todas as threads ao final de cada
        passo j, garantindo que nenhuma thread avance antes que todas as
        trocas do passo atual estejam completas.

        Essa abordagem evita criar e destruir threads a cada iteração —
        as threads são criadas uma vez e encerradas apenas após o sort completo.

    SEGURANÇA DE ESCRITA — Por que não há condição de corrida:
        Para um dado j, cada elemento forma exatamente um par (i, l=i^j).
        A troca é executada apenas pela thread dona do índice menor (i < l).
        Como cada elemento pertence a exatamente uma thread e cada troca é
        executada por exatamente uma thread, não há escrita concorrente
        no mesmo elemento dentro de um mesmo passo j.
*/

// Estrutura de dados passada para cada thread
struct DadosBitonicThread {
    int *vetor;
    int P;                      // tamanho do vetor padded (potência de 2)
    int inicio;                 // índice de início da faixa desta thread
    int fim;                    // índice de fim (exclusivo) da faixa desta thread
    pthread_barrier_t *barreira;
};

// ============================================================
//              FUNÇÃO QUE CADA THREAD EXECUTA
// ============================================================

void* ThreadBitonicWorker(void *arg)
{
    DadosBitonicThread *dados = (DadosBitonicThread *)arg;
    int *v = dados->vetor;
    int P = dados->P;

    for (int k = 2; k <= P; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = dados->inicio; i < dados->fim; i++)
            {
                int l = i ^ j;
                if (l > i)
                {
                    bool crescente = (i & k) == 0;
                    if ((crescente && v[i] > v[l]) || (!crescente && v[i] < v[l]))
                    {
                        int tmp = v[i];
                        v[i] = v[l];
                        v[l] = tmp;
                    }
                }
            }
            // Barreira: todas as threads terminam o passo j antes de avançar
            pthread_barrier_wait(dados->barreira);
        }
    }

    pthread_exit(0);
}

// ============================================================
//              FUNÇÃO PRINCIPAL BITONIC SORT COM THREADS
// ============================================================

void BitonicSortThread(int *vetor, int n, int num_threads)
{
    int P = 1;
    while (P < n) P <<= 1;

    int *v = new int[P];
    for (int i = 0; i < n; i++) v[i] = vetor[i];
    for (int i = n; i < P; i++) v[i] = INT_MAX;

    pthread_barrier_t barreira;
    pthread_barrier_init(&barreira, NULL, num_threads);

    pthread_t *threads = new pthread_t[num_threads];
    DadosBitonicThread *dados_threads = new DadosBitonicThread[num_threads];

    for (int t = 0; t < num_threads; t++)
    {
        dados_threads[t].vetor = v;
        dados_threads[t].P = P;
        dados_threads[t].inicio = (P * t) / num_threads;
        dados_threads[t].fim = (P * (t + 1)) / num_threads;
        dados_threads[t].barreira = &barreira;
        pthread_create(&threads[t], NULL, ThreadBitonicWorker, &dados_threads[t]);
    }

    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    for (int i = 0; i < n; i++) vetor[i] = v[i];

    delete[] v;
    delete[] threads;
    delete[] dados_threads;
    pthread_barrier_destroy(&barreira);
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================

void ExecBitonicThread(const char **entradas, int num_entradas, int num_threads, const char *csv_saida)
{
    FILE *csv = AbrirCSV(csv_saida);
    if (!csv) return;

    for (int i = 0; i < num_entradas; i++)
    {
        long tamanho;
        int *v;
        FILE *file = LerVetor(entradas[i], &v, &tamanho);
        if (!file) continue;

        auto inicio = chrono::high_resolution_clock::now();
        BitonicSortThread(v, tamanho, num_threads);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("BitonicSort Threads - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "BitonicSort - Threads,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, v, tamanho);
        delete[] v;
    }

    fclose(csv);
}
