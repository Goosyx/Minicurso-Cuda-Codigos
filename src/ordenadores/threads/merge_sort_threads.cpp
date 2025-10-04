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

void VerificarOrdenado(const char *arquivo) {
    FILE *file = fopen(arquivo, "rb");
    if (!file) {
        perror("Erro ao abrir arquivo");
        return;
    }

    int atual, anterior;
    size_t lidos;

    // lê o primeiro elemento
    lidos = fread(&anterior, sizeof(int), 1, file);
    if (lidos != 1) {
        fclose(file);
        return; // arquivo vazio ou erro de leitura
    }

    long pos = 1;
    while (fread(&atual, sizeof(int), 1, file) == 1) {
        if (atual < anterior) {
            printf("Erro: arquivo %s está desordenado na posição %ld (%d > %d)\n",
                   arquivo, pos, anterior, atual);
            fclose(file);
            return;
        }
        anterior = atual;
        pos++;
    }

        printf("\nO arquivo %s está ordenado.\n", arquivo);    

    fclose(file);
}

void ImprimirVetor(const char **entrada, int num_entradas)
{
    for(int i = 0; i < num_entradas; i++)
    {
        FILE *file = fopen(entrada[i], "rb");
        if(!file)
        {
            perror(entrada[i]);
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
        fclose(file);


        printf("Vetor %s:\n", entrada[i]);
    
        for(long j = 0; j < tamanho; j++)
        {
            printf("%d ", v[j]);
        }

        printf("\n\n");

        delete[] v;
    }
}

void Merge(int *v, int p, int q, int r){
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
        Merge(vetor, esquerda, meio, direita);

    }

}

void* MergeSortThreadWrapper(void *arg)
{   count++;
    int *id; // ponteiro para o id da thread
    printf("Thread[%lu] contador: %d\n", (long int) pthread_self(), count);

    ThreadDados *dados = (ThreadDados *)arg;
    MergeSort(dados->vetor, dados->esquerda, dados->direita, dados->num_threads);
    delete dados;
    pthread_exit(0);
}

void ExecMerge(const char **entradas, int num_entradas, int num_threads, const char *csv_saida)
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
}

int main()
{
    int num_threads = 8;

    int num_entradas = 1;

    const char *entradas[num_entradas] =
    {
        "dados/250k.bin", "dados/500k.bin", "dados/750k.bin", "dados/1m.bin",
        "dados/2m500.bin", "dados/5m.bin", "dados/7m500.bin", "dados/10m.bin",
        "dados/25m.bin", "dados/50m.bin", "dados/100m.bin"

    };


    FILE *csv = fopen("results/tempos.csv", "w");
    if (csv) {
        fprintf(csv, "Algoritmo,Tamanho,Tempo\n");
        fclose(csv);
    }

    //VerificarOrdenado(entradas[0]);
    ExecMerge(entradas, num_entradas, num_threads,"results/tempos.csv");
    VerificarOrdenado(entradas[0]);

    printf("Threads: %d\n", count);
    printf("SEQUENCIAL: %d\n", count2);


}