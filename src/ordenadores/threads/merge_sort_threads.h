#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <pthread.h>


using namespace std;

struct ThreadDados {
    int *vetor;
    int esquerda;
    int direita;
    int num_threads;
};

void MergeThread(int *v, int p, int q, int r){
    int n1 = q - p + 1;
    int n2 = r - q;

    int *esq = new int[n1];
    int *dir = new int[n2];

    for(int i = 0; i < n1; i++)
        esq[i] = v[p + i];
    for(int j = 0; j < n2; j++)
        dir[j] = v[q + 1 + j];

    int i = 0, j = 0, k = p;
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

    while(i < n1){
        v[k] = esq[i];
        i++;
        k++;
    }

    while(j < n2){
        v[k] = dir[j];
        j++;
        k++;
    }

    delete[] esq;
    delete[] dir;
}

int count = 0;
int count2 = 0;
void* MergeSortThreadWrapper(void *arg);

void MergeSort(int *vetor, int esquerda, int direita, int num_threads)
{

    if (esquerda < direita)
    {

        int meio = (esquerda + direita) / 2;

        if (num_threads > 1)
        {


            pthread_t thread_esq;
            pthread_t thread_dir;

            ThreadDados *dados_esq = new ThreadDados{vetor, esquerda, meio, num_threads / 2};
            ThreadDados *dados_dir = new ThreadDados{vetor, meio + 1, direita, num_threads / 2};

            pthread_create(&thread_esq, NULL, MergeSortThreadWrapper, dados_esq);
            pthread_create(&thread_dir, NULL, MergeSortThreadWrapper, dados_dir);

            pthread_join(thread_esq, NULL);
            pthread_join(thread_dir, NULL);


        }
        else
        {
            count2++;
            MergeSort(vetor, esquerda, meio, 1);
            MergeSort(vetor, meio + 1, direita, 1);
        }
        MergeThread(vetor, esquerda, meio, direita);

    }

}

void* MergeSortThreadWrapper(void *arg)
{   count++;

    ThreadDados *dados = (ThreadDados *)arg;
    MergeSort(dados->vetor, dados->esquerda, dados->direita, dados->num_threads);
    delete dados;
    pthread_exit(0);
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
            continue;
        }

        fseek(file, 0, SEEK_END);
        long tamanho = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        int *v = new int[tamanho];
        if(fread(v, sizeof(int), tamanho, file) != tamanho)
        {
            perror("Erro ao ler o arquivo");
            fclose(file);
            delete[] v;
            continue;
        }

        auto start = chrono::high_resolution_clock::now();
        MergeSort(v, 0, tamanho - 1, num_threads);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        double cpu_time_used = elapsed.count();

        printf("MergeSort Threads - Tempo para ordenar %s: %f segundos\n", entradas[i], cpu_time_used);

        fprintf(csv, "MergeSort - Threads,%ld,%f\n", tamanho, cpu_time_used);

        fseek(file, 0, SEEK_SET);
        if(fwrite(v, sizeof(int), tamanho, file) != tamanho)
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

    printf("Threads: %d\n", count);
    printf("SEQUENCIAL: %d\n", count2);
}