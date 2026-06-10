#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <time.h>
#include <chrono>

using namespace std;

// ============================================================
//                  Observações gerais
// ============================================================
/*
    Este arquivo implementa o algoritmo Bitonic Sort de forma sequencial na CPU.
    O fluxo geral é:
        1) Ler vetores de arquivos binários (int)
        2) Ordenar os dados usando Bitonic Sort
        3) Medir o tempo de execução
        4) Regravar o arquivo com os dados ordenados
        5) Registrar tempos em CSV

    O Bitonic Sort é um algoritmo de ordenação baseado em redes de comparação.
    Ele constrói sequências bitônicas (crescente seguida de decrescente) e as ordena
    usando trocas do tipo compare-and-swap. Requer que o tamanho do vetor seja
    uma potência de 2 — o vetor auxiliar é preenchido com INT_MAX para o alinhamento
    necessário. Os INT_MAX vão para o final do vetor após a ordenação.
*/

// ============================================================
//               FUNÇÃO PRINCIPAL DO BITONIC SORT
// ============================================================

void BitonicSortSeq(int *vetor, int num_elementos)
{
    // Calcula a próxima potência de 2 >= num_elementos
    int tamanho_padded = 1;
    while (tamanho_padded < num_elementos) tamanho_padded <<= 1;

    // Aloca vetor auxiliar preenchido com INT_MAX (elemento neutro: vai para o fim)
    int *vetor_padded = new int[tamanho_padded];
    for (int i = 0; i < num_elementos; i++) vetor_padded[i] = vetor[i];
    for (int i = num_elementos; i < tamanho_padded; i++) vetor_padded[i] = INT_MAX;

    // tamanho_seq: tamanho da sequência bitônica atual (2, 4, 8, ..., tamanho_padded)
    for (int tamanho_seq = 2; tamanho_seq <= tamanho_padded; tamanho_seq <<= 1)
    {
        // j: distância de comparação dentro do passo (tamanho_seq/2, ..., 1)
        for (int j = tamanho_seq >> 1; j > 0; j >>= 1)
        {
            for (int i = 0; i < tamanho_padded; i++)
            {
                int indice_par = i ^ j;  // índice do par de comparação via XOR
                if (indice_par > i)
                {
                    // Direção de ordenação: crescente quando o bit tamanho_seq do índice i é 0
                    bool crescente = (i & tamanho_seq) == 0;
                    if ((crescente  && vetor_padded[i] > vetor_padded[indice_par]) ||
                        (!crescente && vetor_padded[i] < vetor_padded[indice_par]))
                    {
                        int temp = vetor_padded[i];
                        vetor_padded[i] = vetor_padded[indice_par];
                        vetor_padded[indice_par] = temp;
                    }
                }
            }
        }
    }

    // Copia os num_elementos primeiros elementos (os INT_MAX ficaram no final) de volta ao vetor original
    for (int i = 0; i < num_elementos; i++) vetor[i] = vetor_padded[i];

    delete[] vetor_padded;
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================

void ExecBitonicSeq(const char **entradas, int num_entradas, const char *csv_saida)
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
        BitonicSortSeq(vetor, tamanho);
        auto fim = chrono::high_resolution_clock::now();

        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("Bitonic Sort Sequencial - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "BitonicSort - Sequencial,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        delete[] vetor;
    }

    fclose(csv);
}
