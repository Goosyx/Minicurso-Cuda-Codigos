#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "gerador_dados.h"
#include "utils/utils.h"
#include "ordenadores/sequencial/merge_sort_seq.h"
#include "ordenadores/threads/merge_sort_threads.h"



int main()
{
    const int num_threads = 8;

    const int num_entradas = 11;

    const long tamanho_arquivos[num_entradas] = {
        250000, 500000, 750000, 1000000,
        2500000, 5000000, 7500000, 10000000,
        25000000, 50000000, 100000000
    };

    const char *entradas[num_entradas] =
    {
        "dados/250k.bin", "dados/500k.bin", "dados/750k.bin", "dados/1m.bin",
        "dados/2m500.bin", "dados/5m.bin", "dados/7m500.bin", "dados/10m.bin",
        "dados/25m.bin", "dados/50m.bin", "dados/100m.bin"

    };

    FILE *csv = fopen("results/tempos.csv", "w");
    if (!csv) {
        perror("Erro ao abrir arquivo CSV para escrita");
        return 1;
    }
    fprintf(csv, "Algoritmo,Tamanho,Tempo\n");
    fclose(csv);


    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecMergeSeq(entradas, num_entradas, "results/tempos.csv");
    VerificarOrdenado(entradas, num_entradas);

    GerarArquivos(tamanho_arquivos, entradas, num_entradas);
    ExecMergeThread(entradas, num_entradas, num_threads, "results/tempos.csv");
    VerificarOrdenado(entradas,num_entradas);
    


}
