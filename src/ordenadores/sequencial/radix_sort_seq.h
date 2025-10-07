#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <iostream>

using namespace std;

void countingSort(int *vetor, int tamanho, int exp) 
{
    int *output = new int[tamanho]; 
    int count[10] = {0};

    for (int i = 0; i < tamanho; i++)
        count[(vetor[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = tamanho - 1; i >= 0; i--) {
        output[count[(vetor[i] / exp) % 10] - 1] = vetor[i];
        count[(vetor[i] / exp) % 10]--;
    }

    for (int i = 0; i < tamanho; i++)
        vetor[i] = output[i];

    delete[] output;
}

void RadixSort(int *vetor, int tamanho) 
{
    int max = vetor[0];
    for (int i = 1; i < tamanho; i++)
        if (vetor[i] > max)
            max = vetor[i];

    for (int exp = 1; max / exp > 0; exp *= 10)
        countingSort(vetor, tamanho, exp);
}

void exec_radix(const char **entradas, int num_entradas, const char *csv_saida)
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
        RadixSort(v, tamanho);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        double cpu_time_used = elapsed.count();

        printf("Radix Sort Sequencial - Tempo para ordenar %s: %f segundos\n", entradas[i], cpu_time_used);

        fprintf(csv, "RadixSort Sequencial,%ld,%f\n", tamanho, cpu_time_used);

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
