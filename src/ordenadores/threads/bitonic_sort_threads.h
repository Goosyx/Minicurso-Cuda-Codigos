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
        Cada thread executa o mesmo laço (tamanho_seq, j) da versão sequencial, mas
        opera apenas na sua faixa de índices [inicio, fim). Uma barreira
        (pthread_barrier_t) sincroniza todas as threads ao final de cada
        passo j, garantindo que nenhuma thread avance antes que todas as
        trocas do passo atual estejam completas.

        Essa abordagem evita criar e destruir threads a cada iteração —
        as threads são criadas uma vez e encerradas apenas após o sort completo.

    SEGURANÇA DE ESCRITA — Por que não há condição de corrida:
        Para um dado j, cada elemento forma exatamente um par (i, indice_par=i^j).
        A troca é executada apenas pela thread dona do índice menor (i < indice_par).
        Como cada elemento pertence a exatamente uma thread e cada troca é
        executada por exatamente uma thread, não há escrita concorrente
        no mesmo elemento dentro de um mesmo passo j.
*/

// Estrutura de dados passada para cada thread
struct DadosBitonicThread {
    int *vetor;
    int tamanho_padded;          // tamanho do vetor padded (potência de 2)
    int inicio;                  // índice de início da faixa desta thread
    int fim;                     // índice de fim (exclusivo) da faixa desta thread
    pthread_barrier_t *barreira;
};

// ============================================================
//              FUNÇÃO QUE CADA THREAD EXECUTA
// ============================================================

void* ThreadBitonicWorker(void *arg)
{
    DadosBitonicThread *dados = (DadosBitonicThread *)arg;
    int *vetor_padded  = dados->vetor;
    int tamanho_padded = dados->tamanho_padded;

    for (int tamanho_seq = 2; tamanho_seq <= tamanho_padded; tamanho_seq <<= 1)
    {
        for (int j = tamanho_seq >> 1; j > 0; j >>= 1)
        {
            for (int i = dados->inicio; i < dados->fim; i++)
            {
                int indice_par = i ^ j;
                if (indice_par > i)
                {
                    bool crescente = (i & tamanho_seq) == 0;
                    if ((crescente  && vetor_padded[i] > vetor_padded[indice_par]) ||
                        (!crescente && vetor_padded[i] < vetor_padded[indice_par]))
                    {
                        int temp = vetor_padded[i];
                        vetor_padded[i] = vetor_padded[indice_par];
                        vetor_padded[indice_par] = temp;
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

void BitonicSortThread(int *vetor, int num_elementos, int num_threads)
{
    int tamanho_padded = 1;
    while (tamanho_padded < num_elementos) tamanho_padded <<= 1;

    int *vetor_padded = new int[tamanho_padded];
    for (int i = 0; i < num_elementos; i++) vetor_padded[i] = vetor[i];
    for (int i = num_elementos; i < tamanho_padded; i++) vetor_padded[i] = INT_MAX;

    pthread_barrier_t barreira;
    pthread_barrier_init(&barreira, NULL, num_threads);

    pthread_t *threads = new pthread_t[num_threads];
    DadosBitonicThread *dados_threads = new DadosBitonicThread[num_threads];

    for (int id_thread = 0; id_thread < num_threads; id_thread++)
    {
        dados_threads[id_thread].vetor          = vetor_padded;
        dados_threads[id_thread].tamanho_padded = tamanho_padded;
        dados_threads[id_thread].inicio         = (tamanho_padded * id_thread) / num_threads;
        dados_threads[id_thread].fim            = (tamanho_padded * (id_thread + 1)) / num_threads;
        dados_threads[id_thread].barreira       = &barreira;
        pthread_create(&threads[id_thread], NULL, ThreadBitonicWorker, &dados_threads[id_thread]);
    }

    for (int id_thread = 0; id_thread < num_threads; id_thread++)
    {
        pthread_join(threads[id_thread], NULL);
    }

    for (int i = 0; i < num_elementos; i++) vetor[i] = vetor_padded[i];

    delete[] vetor_padded;
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
        int *vetor;
        FILE *file = LerVetor(entradas[i], &vetor, &tamanho);
        if (!file) continue;

        auto inicio = chrono::high_resolution_clock::now();
        BitonicSortThread(vetor, tamanho, num_threads);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("BitonicSort Threads - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "BitonicSort - Threads,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        delete[] vetor;
    }

    fclose(csv);
}
