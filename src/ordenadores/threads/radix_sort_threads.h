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
    int n;
    int id;
    int num_threads;
};

// ============================================================
//              FUNÇÃO EXECUTADA POR CADA THREAD
// ============================================================
void* TrabalhadorRadix(void *arg)
{
    DadosThreadRadix *d = (DadosThreadRadix *)arg;

    int id = d->id;
    int n  = d->n;
    int nt = d->num_threads;

    int inicio = (n * id) / nt;
    int fim    = (n * (id + 1)) / nt;

    for (int shift = 0; shift < 32; shift += 8)
    {
        // ── Fase 1: Contagem local ───────────────────────────────────────────
        for (int b = 0; b < 256; b++) d->contagens[id][b] = 0;

        int *leitura = *(d->p_entrada);
        for (int i = inicio; i < fim; i++)
            d->contagens[id][(leitura[i] >> shift) & 0xFF]++;

        pthread_barrier_wait(d->barreira);

        // ── Fase 2: Prefix sum global (Thread 0) ─────────────────────────────
        if (id == 0)
        {
            int total[256] = {0};
            for (int t = 0; t < nt; t++)
                for (int b = 0; b < 256; b++)
                    total[b] += d->contagens[t][b];

            d->prefixo[0] = 0;
            for (int b = 1; b < 256; b++)
                d->prefixo[b] = d->prefixo[b - 1] + total[b - 1];
        }

        pthread_barrier_wait(d->barreira);

        // ── Fase 3: Distribuição ─────────────────────────────────────────────
        // Calcula posição inicial de cada bucket para esta thread:
        // prefixo[b] + soma das contagens das threads anteriores para b
        int posicao[256];
        for (int b = 0; b < 256; b++)
        {
            int offset = 0;
            for (int t = 0; t < id; t++) offset += d->contagens[t][b];
            posicao[b] = d->prefixo[b] + offset + d->contagens[id][b] - 1;
        }

        // Distribui de trás para frente (estável)
        int *src = *(d->p_entrada);
        int *dst = *(d->p_saida);
        for (int i = fim - 1; i >= inicio; i--)
        {
            int b = (src[i] >> shift) & 0xFF;
            dst[posicao[b]--] = src[i];
        }

        pthread_barrier_wait(d->barreira);

        // ── Fase 4: Swap de ponteiros (Thread 0) ──────────────────────────────
        if (id == 0)
        {
            int *temp       = *(d->p_entrada);
            *(d->p_entrada) = *(d->p_saida);
            *(d->p_saida)   = temp;
        }

        pthread_barrier_wait(d->barreira);
    }

    pthread_exit(0);
}

// ============================================================
//                  RADIX SORT COM THREADS
// ============================================================
void RadixSortThread(int *vetor, int n, int num_threads)
{
    if (n <= 1) return;

    int *buffer  = new int[n];
    int *entrada = vetor;
    int *saida   = buffer;

    int **contagens = new int*[num_threads];
    for (int t = 0; t < num_threads; t++) contagens[t] = new int[256];

    int *prefixo = new int[256];

    pthread_barrier_t barreira;
    pthread_barrier_init(&barreira, NULL, num_threads);

    pthread_t        *threads       = new pthread_t[num_threads];
    DadosThreadRadix *dados_threads = new DadosThreadRadix[num_threads];

    for (int t = 0; t < num_threads; t++)
    {
        dados_threads[t].p_entrada   = &entrada;
        dados_threads[t].p_saida     = &saida;
        dados_threads[t].contagens   = contagens;
        dados_threads[t].prefixo     = prefixo;
        dados_threads[t].barreira    = &barreira;
        dados_threads[t].n           = n;
        dados_threads[t].id          = t;
        dados_threads[t].num_threads = num_threads;
        pthread_create(&threads[t], NULL, TrabalhadorRadix, &dados_threads[t]);
    }

    for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);

    pthread_barrier_destroy(&barreira);
    for (int t = 0; t < num_threads; t++) delete[] contagens[t];
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
