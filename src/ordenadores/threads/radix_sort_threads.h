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
    Este arquivo implementa o algoritmo Radix Sort LSD com paralelismo via
    threads POSIX (pthreads).

    OTIMIZAÇÃO — Alternância de ponteiros (ping-pong) sem cópia final:
        Em vez de copiar o buffer de saída de volta para o vetor ao final de
        cada passagem (O(n) sequencial por Thread 0), os papéis de entrada e
        saída são alternados trocando os ponteiros após cada passagem.
        Após 4 passagens (número par de trocas), o resultado está em 'vetor'.

    DECISÃO DE DESIGN — Ordem das fases por passagem:
        Fase 1 — Contagem:       cada thread conta os bytes de sua fatia de *p_entrada
        Barreira 1
        Fase 2 — Prefix sum:     Thread 0 agrega e calcula prefixo[256]
        Barreira 2
        Fase 3 — Distribuição:   cada thread escreve de *p_entrada para *p_saida
        Barreira 3
        Swap de ponteiros:       Thread 0 troca *p_entrada ↔ *p_saida
        Barreira 4

    ATENÇÃO — por que o swap ocorre APÓS a Fase 3:
        A Fase 3 lê de *p_entrada e escreve em *p_saida da passagem atual.
        O swap só pode acontecer depois que a distribuição está completa.
        Fazer o swap antes (na Fase 2) causaria a Fase 3 ler do buffer vazio
        e escrever no vetor original — resultado incorreto e segfault.

    DECISÃO DE DESIGN — Buffer único compartilhado:
        Cada thread escreve em sub-região exclusiva de cada bucket (calculada
        via offset das threads anteriores) — sem condição de corrida.
*/

// ============================================================
//              ESTRUTURA DE DADOS DAS THREADS
// ============================================================
struct DadosThreadRadix {
    int **p_entrada;       // ponteiro para o ponteiro de leitura desta passagem
    int **p_saida;         // ponteiro para o ponteiro de escrita desta passagem
    int **contagens;       // matriz [num_threads][256] — contagem local por byte
    int  *prefixo;         // array [256] — posição inicial de cada bucket
    pthread_barrier_t *barreira;
    int n;
    int id_thread;
    int num_threads;
};

// ============================================================
//              FUNÇÃO EXECUTADA POR CADA THREAD
// ============================================================
void* TrabalhadorRadix(void *arg)
{
    DadosThreadRadix *dados = (DadosThreadRadix *)arg;

    int id = dados->id_thread;
    int n  = dados->n;
    int nt = dados->num_threads;

    int inicio = (n * id) / nt;
    int fim    = (n * (id + 1)) / nt;

    int shift = 0;
    while (shift < 32)
    {
        // ── Fase 1: Contagem local ───────────────────────────────────────────
        int b = 0;
        while (b < 256) { dados->contagens[id][b] = 0; b++; }

        int *leitura = *(dados->p_entrada);
        int i = inicio;
        while (i < fim)
        {
            dados->contagens[id][(leitura[i] >> shift) & 0xFF]++;
            i++;
        }

        // ── Barreira 1 ───────────────────────────────────────────────────────
        pthread_barrier_wait(dados->barreira);

        // ── Fase 2: Prefix sum (Thread 0) ────────────────────────────────────
        if (id == 0)
        {
            int total[256] = {0};
            int t = 0;
            while (t < nt)
            {
                b = 0;
                while (b < 256) { total[b] += dados->contagens[t][b]; b++; }
                t++;
            }

            dados->prefixo[0] = 0;
            b = 1;
            while (b < 256)
            {
                dados->prefixo[b] = dados->prefixo[b - 1] + total[b - 1];
                b++;
            }
        }

        // ── Barreira 2 ───────────────────────────────────────────────────────
        pthread_barrier_wait(dados->barreira);

        // ── Fase 3: Distribuição ─────────────────────────────────────────────
        // Lê de *p_entrada (dados desta passagem) e escreve em *p_saida.
        // Sub-região exclusiva por thread: sem condição de corrida.
        int *src = *(dados->p_entrada);
        int *dst = *(dados->p_saida);

        int posicao_local[256];
        b = 0;
        while (b < 256)
        {
            int offset = 0;
            int t = 0;
            while (t < id) { offset += dados->contagens[t][b]; t++; }
            posicao_local[b] = dados->prefixo[b] + offset + dados->contagens[id][b] - 1;
            b++;
        }

        i = fim - 1;
        while (i >= inicio)
        {
            int valor_byte = (src[i] >> shift) & 0xFF;
            dst[posicao_local[valor_byte]] = src[i];
            posicao_local[valor_byte]--;
            i--;
        }

        // ── Barreira 3: distribuição completa ────────────────────────────────
        pthread_barrier_wait(dados->barreira);

        // ── Swap de ponteiros (Thread 0) ──────────────────────────────────────
        // Feito APÓS a Fase 3: saída desta passagem vira entrada da próxima.
        // Após 4 trocas (número par), *p_entrada == vetor (resultado correto).
        if (id == 0)
        {
            int *temp           = *(dados->p_entrada);
            *(dados->p_entrada) = *(dados->p_saida);
            *(dados->p_saida)   = temp;
        }

        // ── Barreira 4: swap visível a todas antes do próximo shift ───────────
        pthread_barrier_wait(dados->barreira);

        shift += 8;
    }

    pthread_exit(0);
}

// ============================================================
//                  FUNÇÃO PRINCIPAL RADIX SORT THREADS
// ============================================================
void RadixSortThread(int *vetor, int n, int num_threads)
{
    if (n <= 1) return;

    int *buffer = new int[n];

    // Ponteiros compartilhados alternados a cada passagem
    int *entrada = vetor;
    int *saida   = buffer;

    int **contagens = new int*[num_threads];
    int t = 0;
    while (t < num_threads) { contagens[t] = new int[256]; t++; }

    int *prefixo = new int[256];

    pthread_barrier_t barreira;
    pthread_barrier_init(&barreira, NULL, num_threads);

    pthread_t        *threads       = new pthread_t[num_threads];
    DadosThreadRadix *dados_threads = new DadosThreadRadix[num_threads];

    t = 0;
    while (t < num_threads)
    {
        dados_threads[t].p_entrada   = &entrada;
        dados_threads[t].p_saida     = &saida;
        dados_threads[t].contagens   = contagens;
        dados_threads[t].prefixo     = prefixo;
        dados_threads[t].barreira    = &barreira;
        dados_threads[t].n           = n;
        dados_threads[t].id_thread   = t;
        dados_threads[t].num_threads = num_threads;

        pthread_create(&threads[t], NULL, TrabalhadorRadix, &dados_threads[t]);
        t++;
    }

    t = 0;
    while (t < num_threads) { pthread_join(threads[t], NULL); t++; }

    // Após 4 passagens (par de swaps), 'entrada' == vetor — resultado em vetor. ✓

    pthread_barrier_destroy(&barreira);
    t = 0;
    while (t < num_threads) { delete[] contagens[t]; t++; }
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