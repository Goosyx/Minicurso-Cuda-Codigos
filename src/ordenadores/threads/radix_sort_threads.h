#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <pthread.h>

using namespace std;

// ============================================================
//                  Observações gerais
// ============================================================
/*
    Radix Sort LSD paralelo com threads POSIX, base 256 (1 byte por passagem).

    Cada passagem tem 4 fases separadas por barreiras:

        Fase 1 — Contagem local:
            Cada thread conta os bytes da sua faixa em contagens[id][256].

        Fase 2 — Prefix sum global (Thread 0):
            Agrega as contagens de todas as threads e calcula prefixo[256],
            a posição inicial de cada bucket no vetor de saída.

        Fase 3 — Distribuição:
            Cada thread distribui sua faixa de trás para frente em
            sub-regiões exclusivas de cada bucket — sem condição de corrida.

        Fase 4 — Swap de ponteiros (Thread 0):
            Troca entrada ↔ saída para a próxima passagem.
            Após 4 passagens (número par), o resultado está em 'vetor'.
*/

// ============================================================
//              ESTRUTURA DE DADOS DAS THREADS
// ============================================================
struct DadosThreadRadix {
    int **p_entrada;        // ponteiro para o ponteiro de leitura desta passagem
    int **p_saida;          // ponteiro para o ponteiro de escrita desta passagem
    int **contagens;        // contagens[id][256] — contagem local por byte
    int  *prefixo;          // prefixo[256] — posição inicial de cada bucket
    pthread_barrier_t *barreira;
    int num_elementos;
    int id;
    int num_threads;
};

// ============================================================
//              FUNÇÃO EXECUTADA POR CADA THREAD
// ============================================================
void* TrabalhadorRadix(void *arg)
{
    DadosThreadRadix *dados = (DadosThreadRadix *)arg;

    int id             = dados->id;
    int num_elementos  = dados->num_elementos;
    int num_threads_total = dados->num_threads;

    int inicio = (num_elementos * id) / num_threads_total;
    int fim    = (num_elementos * (id + 1)) / num_threads_total;

    for (int shift = 0; shift < 32; shift += 8)
    {
        // ── Fase 1: Contagem local ───────────────────────────────────────────
        for (int bucket = 0; bucket < 256; bucket++) dados->contagens[id][bucket] = 0;

        int *leitura = *(dados->p_entrada);
        for (int i = inicio; i < fim; i++)
            dados->contagens[id][(leitura[i] >> shift) & 0xFF]++;

        pthread_barrier_wait(dados->barreira);

        // ── Fase 2: Prefix sum global (Thread 0) ─────────────────────────────
        if (id == 0)
        {
            int total[256] = {0};
            for (int id_thread = 0; id_thread < num_threads_total; id_thread++)
                for (int bucket = 0; bucket < 256; bucket++)
                    total[bucket] += dados->contagens[id_thread][bucket];

            dados->prefixo[0] = 0;
            for (int bucket = 1; bucket < 256; bucket++)
                dados->prefixo[bucket] = dados->prefixo[bucket - 1] + total[bucket - 1];
        }

        pthread_barrier_wait(dados->barreira);

        // ── Fase 3: Distribuição ─────────────────────────────────────────────
        // Calcula posição inicial de cada bucket para esta thread:
        // prefixo[bucket] + soma das contagens das threads anteriores para bucket
        int posicao[256];
        for (int bucket = 0; bucket < 256; bucket++)
        {
            int offset = 0;
            for (int id_thread_anterior = 0; id_thread_anterior < id; id_thread_anterior++)
                offset += dados->contagens[id_thread_anterior][bucket];
            posicao[bucket] = dados->prefixo[bucket] + offset + dados->contagens[id][bucket] - 1;
        }

        // Distribui de trás para frente (estável)
        int *src = *(dados->p_entrada);
        int *dst = *(dados->p_saida);
        for (int i = fim - 1; i >= inicio; i--)
        {
            int bucket = (src[i] >> shift) & 0xFF;
            dst[posicao[bucket]--] = src[i];
        }

        pthread_barrier_wait(dados->barreira);

        // ── Fase 4: Swap de ponteiros (Thread 0) ──────────────────────────────
        if (id == 0)
        {
            int *temp           = *(dados->p_entrada);
            *(dados->p_entrada) = *(dados->p_saida);
            *(dados->p_saida)   = temp;
        }

        pthread_barrier_wait(dados->barreira);
    }

    pthread_exit(0);
}

// ============================================================
//                  RADIX SORT COM THREADS
// ============================================================
void RadixSortThread(int *vetor, int num_elementos, int num_threads)
{
    if (num_elementos <= 1) return;

    int *buffer  = new int[num_elementos];
    int *entrada = vetor;
    int *saida   = buffer;

    int **contagens = new int*[num_threads];
    for (int id_thread = 0; id_thread < num_threads; id_thread++)
        contagens[id_thread] = new int[256];

    int *prefixo = new int[256];

    pthread_barrier_t barreira;
    pthread_barrier_init(&barreira, NULL, num_threads);

    pthread_t        *threads       = new pthread_t[num_threads];
    DadosThreadRadix *dados_threads = new DadosThreadRadix[num_threads];

    for (int id_thread = 0; id_thread < num_threads; id_thread++)
    {
        dados_threads[id_thread].p_entrada    = &entrada;
        dados_threads[id_thread].p_saida      = &saida;
        dados_threads[id_thread].contagens    = contagens;
        dados_threads[id_thread].prefixo      = prefixo;
        dados_threads[id_thread].barreira     = &barreira;
        dados_threads[id_thread].num_elementos = num_elementos;
        dados_threads[id_thread].id           = id_thread;
        dados_threads[id_thread].num_threads  = num_threads;
        pthread_create(&threads[id_thread], NULL, TrabalhadorRadix, &dados_threads[id_thread]);
    }

    for (int id_thread = 0; id_thread < num_threads; id_thread++)
        pthread_join(threads[id_thread], NULL);

    pthread_barrier_destroy(&barreira);
    for (int id_thread = 0; id_thread < num_threads; id_thread++)
        delete[] contagens[id_thread];
    delete[] contagens;
    delete[] prefixo;
    delete[] buffer;
    delete[] threads;
    delete[] dados_threads;
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
void ExecRadixThread(const char **entradas, int num_entradas, int num_threads, const char *csv_saida)
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
        RadixSortThread(vetor, tamanho, num_threads);
        auto fim = chrono::high_resolution_clock::now();

        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("Radix Sort Threads - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "RadixSort - Threads,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        delete[] vetor;
    }

    fclose(csv);
}
