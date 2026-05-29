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
    int *buffer;           // Buffer auxiliar pré-alocado exclusivo desta thread
    int inicio_merge;      // Índice do primeiro merge que esta thread deve fazer
    int fim_merge;         // Índice do último merge que esta thread deve fazer
    int tamanho_sublista;  // Tamanho atual das sublistas a serem mescladas
    int n;                 // Tamanho total do vetor
};

/*
 * MergeThread: mescla dois subvetores adjacentes já ordenados em um único segmento ordenado.
 *
 * Parâmetros:
 * - v: ponteiro para o array de inteiros a ser ordenado
 * - buffer: ponteiro para buffer auxiliar pré-alocado exclusivo desta thread
 * - p: índice inicial do primeiro subvetor
 * - q: índice final do primeiro subvetor (inclusivo)
 * - r: índice final do segundo subvetor (inclusivo)
 *
 * ATENÇÃO — motivação do parâmetro buffer:
 *   A versão anterior alocava `new int[]` e `delete[]` a cada chamada de MergeThread.
 *   Com múltiplas threads executando simultaneamente, o allocator de heap do C++ usa
 *   locks internos, criando contenção entre threads e serializando as alocações —
 *   o oposto do que se espera de código paralelo. O buffer agora é alocado uma única
 *   vez por thread em MergeSortThread e reutilizado em todas as iterações,
 *   eliminando essa contenção e tornando a comparação com os outros modos mais justa.
 */
void MergeThread(int *v, int *buffer, int p, int q, int r){
    int n1 = q - p + 1;
    int n2 = r - q;

    // Copia os dois subvetores para o buffer auxiliar pré-alocado.
    // buffer[0..n1-1]      ← metade esquerda
    // buffer[n1..n1+n2-1]  ← metade direita
    int i = 0;
    while(i < n1){
        buffer[i] = v[p + i];
        i++;
    }
    
    int j = 0;
    while(j < n2){
        buffer[n1 + j] = v[q + 1 + j];
        j++;
    }

    // Merge principal: compara elementos das duas metades no buffer
    // e escreve o resultado ordenado de volta em v
    i = 0;
    j = 0;
    int k = p;
    while(i < n1 && j < n2){
        if(buffer[i] <= buffer[n1 + j]){
            v[k] = buffer[i];
            i++;
        } else {
            v[k] = buffer[n1 + j];
            j++;
        }
        k++;
    }

    // Copia elementos restantes da esquerda
    while(i < n1){
        v[k] = buffer[i];
        i++;
        k++;
    }

    // Copia elementos restantes da direita
    while(j < n2){
        v[k] = buffer[n1 + j];
        j++;
        k++;
    }
}

// Função que cada thread executa
void* ThreadMergeWorker(void *arg)
{
    ThreadDados *dados = (ThreadDados *)arg;
    
    // Cada thread processa uma faixa de merges usando seu buffer exclusivo.
    // O buffer foi alocado pelo chamador (MergeSortThread) com tamanho n,
    // garantindo espaço suficiente para qualquer merge desta thread.
    int idx_merge = dados->inicio_merge;
    while(idx_merge < dados->fim_merge)
    {
        int inicio = idx_merge * dados->tamanho_sublista * 2;
        
        if(inicio >= dados->n - 1)
        {
            break;
        }
            
        int p = inicio;
        int q = min(inicio + dados->tamanho_sublista - 1, dados->n - 1);
        int r = min(inicio + 2 * dados->tamanho_sublista - 1, dados->n - 1);
        
        if(q < r)
        {
            MergeThread(dados->vetor, dados->buffer, p, q, r);
        }
        
        idx_merge++;
    }
    
    delete dados;
    pthread_exit(0);
}

// MergeSort Bottom-Up com Threads 
void MergeSortThread(int *vetor, int n, int num_threads)
{
    if(n <= 1)
    {
        return;
    }
    
    // Pré-aloca um buffer exclusivo por thread, cada um de tamanho n.
    // Tamanho n é suficiente pois o maior merge possível envolve o vetor inteiro.
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
    int t = 0;
    while(t < num_threads)
    {
        buffers[t] = new int[n];
        t++;
    }

    // Loop externo: tamanho das sublistas (1, 2, 4, 8, ...)
    int tamanho = 1;
    while(tamanho < n)
    {
        int total_merges = (n + (tamanho * 2 - 1)) / (tamanho * 2);
        int threads_usadas = min(num_threads, total_merges);
        
        pthread_t *threads = new pthread_t[threads_usadas];
        
        t = 0;
        while(t < threads_usadas)
        {
            ThreadDados *dados = new ThreadDados;
            dados->vetor = vetor;
            dados->buffer = buffers[t]; // buffer exclusivo desta thread
            dados->tamanho_sublista = tamanho;
            dados->n = n;
            dados->inicio_merge = (total_merges * t) / threads_usadas;
            dados->fim_merge = (total_merges * (t + 1)) / threads_usadas;
            
            pthread_create(&threads[t], NULL, ThreadMergeWorker, dados);
            t++;
        }
        
        t = 0;
        while(t < threads_usadas)
        {
            pthread_join(threads[t], NULL);
            t++;
        }
        
        delete[] threads;
        tamanho *= 2;
    }

    // Libera todos os buffers após o sort completo
    t = 0;
    while(t < num_threads)
    {
        delete[] buffers[t];
        t++;
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
        int *v;
        FILE *file = LerVetor(entradas[i], &v, &tamanho);
        if (!file) continue;

        auto inicio = chrono::high_resolution_clock::now();
        MergeSortThread(v, tamanho, num_threads);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("MergeSort Threads - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "MergeSort - Threads,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, v, tamanho);
        delete[] v;
    }

    fclose(csv);
}