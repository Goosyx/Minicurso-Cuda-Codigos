#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <iostream>

using namespace std;

void countingSort(int *v, int tamanho, int exp) 
{
    int *output = new int[tamanho]; // vetor de saída
    int count[10] = {0}; // vetor de contagem para dígitos (0-9)

    // Conta a ocorrência de cada dígito no vetor de entrada
    for (int i = 0; i < tamanho; i++)
        count[(v[i] / exp) % 10]++;

    // Atualiza o vetor de contagem para armazenar a posição real dos dígitos no vetor de saída
    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    // Constrói o vetor de saída
    for (int i = tamanho - 1; i >= 0; i--) {
        output[count[(v[i] / exp) % 10] - 1] = v[i];
        count[(v[i] / exp) % 10]--;
    }

    // Copia o vetor de saída de volta para o vetor original
    for (int i = 0; i < tamanho; i++)
        v[i] = output[i];

    delete[] output;
}

void RadixSort(int *v, int tamanho) 
{
    // Encontra o maior número para determinar o número de dígitos
    int max = v[0];
    for (int i = 1; i < tamanho; i++)
        if (v[i] > max)
            max = v[i];

    // Aplica o counting sort para cada dígito
    for (int exp = 1; max / exp > 0; exp *= 10)
        countingSort(v, tamanho, exp);
}

void exec_radix(const char **entradas, int num_entradas, const char *csv_saida)
{
    FILE *csv = fopen(csv_saida, "w");
    if (!csv) {
        perror(csv_saida);
        return;
    }
    fprintf(csv, "Algoritmo,Tamanho,Tempo\n");

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

        printf("Radix Sort - Tempo para ordenar %s: %f segundos\n", entradas[i], cpu_time_used);

        fprintf(csv, "RadixSort,%ld,%f\n", tamanho, cpu_time_used);

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

void imprimir_vetor(const char **entrada, int num_entradas)
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

int main()
{
    int num_entradas = 11;
    const char *entradas[num_entradas] =
    {
        "dados/250k.bin", "dados/500k.bin", "dados/750k.bin", "dados/1m.bin",
        "dados/2m500.bin", "dados/5m.bin", "dados/7m500.bin", "dados/10m.bin",
        "dados/25m.bin", "dados/50m.bin", "dados/100m.bin"

    };

    //imprimir_vetor(entradas, num_entradas);
    exec_radix(entradas, num_entradas, "results/tempos.csv");
    //imprimir_vetor(entradas, num_entradas);

}