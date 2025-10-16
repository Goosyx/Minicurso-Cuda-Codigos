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
    int inicio_merge;      // Índice do primeiro merge que esta thread deve fazer
    int fim_merge;         // Índice do último merge que esta thread deve fazer
    int tamanho_sublista;  // Tamanho atual das sublistas a serem mescladas
    int n;                 // Tamanho total do vetor
};

// Função de merge (padronizada com while, similar ao CUDA)
void MergeThread(int *v, int p, int q, int r){
    int n1 = q - p + 1;
    int n2 = r - q;

    int *esq = new int[n1];
    int *dir = new int[n2];

    // Copia elementos para arrays temporários usando while
    int i = 0;
    while(i < n1){
        esq[i] = v[p + i];
        i++;
    }
    
    int j = 0;
    while(j < n2){
        dir[j] = v[q + 1 + j];
        j++;
    }

    // Merge principal
    i = 0;
    j = 0;
    int k = p;
    while(i < n1 && j < n2){
        if(esq[i] <= dir[j]){
            v[k] = esq[i];
            i++;
        } else {
            v[k] = dir[j];
            j++;
        }
        k++;
    }

    // Copia elementos restantes da esquerda
    while(i < n1){
        v[k] = esq[i];
        i++;
        k++;
    }

    // Copia elementos restantes da direita
    while(j < n2){
        v[k] = dir[j];
        j++;
        k++;
    }

    delete[] esq;
    delete[] dir;
}

// Função que cada thread executa
void* ThreadMergeWorker(void *arg)
{
    ThreadDados *dados = (ThreadDados *)arg;
    
    // Cada thread processa uma faixa de merges
    int idx_merge = dados->inicio_merge;
    while(idx_merge < dados->fim_merge)
    {
        // Calcula os índices p, q, r para este merge específico
        int inicio = idx_merge * dados->tamanho_sublista * 2;
        
        if(inicio >= dados->n - 1)
        {
            break;
        }
            
        int p = inicio;
        int q = min(inicio + dados->tamanho_sublista - 1, dados->n - 1);
        int r = min(inicio + 2 * dados->tamanho_sublista - 1, dados->n - 1);
        
        // Só faz merge se houver dois sub-arrays
        if(q < r)
        {
            MergeThread(dados->vetor, p, q, r);
        }
        
        idx_merge++;
    }
    
    delete dados;
    pthread_exit(0);
}

// MergeSort Bottom-Up com Threads (similar ao CUDA mas com threads POSIX)
void MergeSortThread(int *vetor, int n, int num_threads)
{
    if(n <= 1)
        return;
    
    // Loop externo: tamanho das sublistas (1, 2, 4, 8, ...)
    int tamanho = 1;
    while(tamanho < n)
    {
        // Calcula quantos merges precisam ser feitos nesta iteração
        int total_merges = (n + (tamanho * 2 - 1)) / (tamanho * 2);
        
        // Limita o número de threads ao número de merges disponíveis
        int threads_usadas = min(num_threads, total_merges);
        
        // Distribui os merges entre as threads
        pthread_t *threads = new pthread_t[threads_usadas];
        
        int t = 0;
        while(t < threads_usadas)
        {
            ThreadDados *dados = new ThreadDados;
            dados->vetor = vetor;
            dados->tamanho_sublista = tamanho;
            dados->n = n;
            
            // Divide os merges igualmente entre as threads
            dados->inicio_merge = (total_merges * t) / threads_usadas;
            dados->fim_merge = (total_merges * (t + 1)) / threads_usadas;
            
            pthread_create(&threads[t], NULL, ThreadMergeWorker, dados);
            t++;
        }
        
        // Aguarda todas as threads terminarem antes de passar para o próximo tamanho
        t = 0;
        while(t < threads_usadas)
        {
            pthread_join(threads[t], NULL);
            t++;
        }
        
        delete[] threads;
        tamanho *= 2;
    }
}

void ExecMergeThread(const char **entradas, int num_entradas, int num_threads, const char *csv_saida)
{
    FILE *csv = fopen(csv_saida, "a");
    if (!csv)
    {
        perror("Erro ao abrir arquivo CSV");
        return;
    }

    for(int i = 0; i < num_entradas; i++)
    {
        FILE *file = fopen(entradas[i], "rb+");
        if(!file)
        {
            perror(entradas[i]);
            return;
        }

        fseek(file, 0, SEEK_END);
        long tamanho = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        int *v = new int[tamanho];
        if(fread(v, sizeof(int), tamanho, file) != (size_t)tamanho)
        {
            perror("Erro ao ler o arquivo");
            fclose(file);
            delete[] v;
            continue;
        }

        auto start = chrono::high_resolution_clock::now();
        MergeSortThread(v, tamanho, num_threads);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        double tempo = elapsed.count();

        printf("MergeSort Threads - Tempo para ordenar %s: %f s\n", entradas[i], tempo);

        fprintf(csv, "MergeSort - Threads,%ld,%f\n", tamanho, tempo);

        fseek(file, 0, SEEK_SET);
        if(fwrite(v, sizeof(int), tamanho, file) != (size_t)tamanho)
        {
            perror("Erro ao escrever no arquivo");
            fclose(file);
            delete[] v;
            continue;
        }

        fclose(file);
        delete[] v;
    }

    fclose(csv);
}