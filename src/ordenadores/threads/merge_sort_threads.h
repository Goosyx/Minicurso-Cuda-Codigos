#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <pthread.h>

using namespace std;

// Estrutura para passar dados para cada thread
struct ThreadDados {
    int *vetor;
    int *buffer;             // Buffer auxiliar pré-alocado exclusivo desta thread
    int inicio_merge;        // Índice do primeiro merge que esta thread deve fazer
    int fim_merge;           // Índice do último merge que esta thread deve fazer
    int tamanho_sublista;    // Tamanho atual das sublistas a serem mescladas
    int num_elementos;       // Tamanho total do vetor
};

/*
 * MergeThread: mescla dois subvetores adjacentes já ordenados em um único segmento ordenado.
 *
 * Parâmetros:
 * - vetor: ponteiro para o array de inteiros a ser ordenado
 * - buffer: ponteiro para buffer auxiliar pré-alocado exclusivo desta thread
 * - começo: índice inicial do primeiro subvetor
 * - meio: índice final do primeiro subvetor (inclusivo)
 * - fim: índice final do segundo subvetor (inclusivo)
 *
 * ATENÇÃO — motivação do parâmetro buffer:
 *   A versão anterior alocava `new int[]` e `delete[]` a cada chamada de MergeThread.
 *   Com múltiplas threads executando simultaneamente, o allocator de heap do C++ usa
 *   locks internos, criando contenção entre threads e serializando as alocações —
 *   o oposto do que se espera de código paralelo. O buffer agora é alocado uma única
 *   vez por thread em MergeSortThread e reutilizado em todas as iterações,
 *   eliminando essa contenção e tornando a comparação com os outros modos mais justa.
 */
void MergeThread(int *vetor, int *buffer, int começo, int meio, int fim){
    int tam_esquerda = meio - começo + 1;
    int tam_direita  = fim - meio;

    // Copia os dois subvetores para o buffer auxiliar pré-alocado.
    // buffer[0..tam_esquerda-1]                         ← metade esquerda
    // buffer[tam_esquerda..tam_esquerda+tam_direita-1]  ← metade direita
    int i = 0;
    while(i < tam_esquerda){
        buffer[i] = vetor[começo + i];
        i++;
    }

    int j = 0;
    while(j < tam_direita){
        buffer[tam_esquerda + j] = vetor[meio + 1 + j];
        j++;
    }

    // Merge principal: compara elementos das duas metades no buffer
    // e escreve o resultado ordenado de volta em vetor
    i = 0;
    j = 0;
    int idx_atual = começo;
    while(i < tam_esquerda && j < tam_direita){
        if(buffer[i] <= buffer[tam_esquerda + j]){
            vetor[idx_atual] = buffer[i];
            i++;
        } else {
            vetor[idx_atual] = buffer[tam_esquerda + j];
            j++;
        }
        idx_atual++;
    }

    // Copia elementos restantes da esquerda
    while(i < tam_esquerda){
        vetor[idx_atual] = buffer[i];
        i++;
        idx_atual++;
    }

    // Copia elementos restantes da direita
    while(j < tam_direita){
        vetor[idx_atual] = buffer[tam_esquerda + j];
        j++;
        idx_atual++;
    }
}

// Função que cada thread executa
void* ThreadMergeWorker(void *arg)
{
    ThreadDados *dados = (ThreadDados *)arg;

    // Cada thread processa uma faixa de merges usando seu buffer exclusivo.
    // O buffer foi alocado pelo chamador (MergeSortThread) com tamanho num_elementos,
    // garantindo espaço suficiente para qualquer merge desta thread.
    int idx_merge = dados->inicio_merge;
    while(idx_merge < dados->fim_merge)
    {
        int inicio = idx_merge * dados->tamanho_sublista * 2;

        if(inicio >= dados->num_elementos - 1)
        {
            break;
        }

        int começo = inicio;
        int meio   = min(inicio + dados->tamanho_sublista - 1, dados->num_elementos - 1);
        int fim    = min(inicio + 2 * dados->tamanho_sublista - 1, dados->num_elementos - 1);

        if(meio < fim)
        {
            MergeThread(dados->vetor, dados->buffer, começo, meio, fim);
        }

        idx_merge++;
    }

    delete dados;
    pthread_exit(0);
}

// MergeSort Bottom-Up com Threads
void MergeSortThread(int *vetor, int num_elementos, int num_threads)
{
    if(num_elementos <= 1)
    {
        return;
    }

    // Pré-aloca um buffer exclusivo por thread, cada um de tamanho num_elementos.
    // Tamanho num_elementos é suficiente pois o maior merge possível envolve o vetor inteiro.
    // Cada thread usa seu próprio buffer sem compartilhamento, evitando contenção.
    // Os buffers são alocados uma única vez aqui e reutilizados em todas as
    // iterações do loop externo, eliminando as alocações por merge da versão anterior.
    //
    // ATENÇÃO — por que um buffer por thread e não um único buffer compartilhado:
    //   Threads executam merges simultaneamente em regiões distintas do vetor.
    //   Um buffer único compartilhado exigiria sincronização ou particionamento
    //   explícito, aumentando a complexidade. Um buffer por thread é a solução
    //   mais simples e elimina qualquer risco de condição de corrida no buffer.
    int **buffers = new int*[num_threads];
    int id_thread = 0;
    while(id_thread < num_threads)
    {
        buffers[id_thread] = new int[num_elementos];
        id_thread++;
    }

    // Loop externo: tamanho das sublistas (1, 2, 4, 8, ...)
    int tamanho = 1;
    while(tamanho < num_elementos)
    {
        int total_merges  = (num_elementos + (tamanho * 2 - 1)) / (tamanho * 2);
        int threads_usadas = min(num_threads, total_merges);

        pthread_t *threads = new pthread_t[threads_usadas];

        id_thread = 0;
        while(id_thread < threads_usadas)
        {
            ThreadDados *dados = new ThreadDados;
            dados->vetor            = vetor;
            dados->buffer           = buffers[id_thread]; // buffer exclusivo desta thread
            dados->tamanho_sublista = tamanho;
            dados->num_elementos    = num_elementos;
            dados->inicio_merge     = (total_merges * id_thread) / threads_usadas;
            dados->fim_merge        = (total_merges * (id_thread + 1)) / threads_usadas;

            pthread_create(&threads[id_thread], NULL, ThreadMergeWorker, dados);
            id_thread++;
        }

        id_thread = 0;
        while(id_thread < threads_usadas)
        {
            pthread_join(threads[id_thread], NULL);
            id_thread++;
        }

        delete[] threads;
        tamanho *= 2;
    }

    // Libera todos os buffers após o sort completo
    id_thread = 0;
    while(id_thread < num_threads)
    {
        delete[] buffers[id_thread];
        id_thread++;
    }
    delete[] buffers;
}

void ExecMergeThread(const char **entradas, int num_entradas, int num_threads, const char *csv_saida)
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
        MergeSortThread(vetor, tamanho, num_threads);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("MergeSort Threads - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "MergeSort - Threads,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        delete[] vetor;
    }

    fclose(csv);
}
