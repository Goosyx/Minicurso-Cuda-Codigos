#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;

void MergeSeq(int *v, int p, int q, int r){
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

void MergeSort(int *v, int p, int r)
{
    if(p < r){
        int m = (p+r)/2;
        MergeSort(v, p, m);
        MergeSort(v, m+1, r);
        MergeSeq(v, p, m, r);
    }
}

void ExecMergeSeq(const char **entradas, int num_entradas, const char *csv_saida)
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
        MergeSort(v, 0, tamanho - 1);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        double cpu_time_used = elapsed.count();

        printf("Merge Sort Sequencial - Tempo para ordenar %s: %f segundos\n", entradas[i], cpu_time_used);

        fprintf(csv, "MergeSort - Sequencial,%ld,%f\n", tamanho, cpu_time_used);

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
