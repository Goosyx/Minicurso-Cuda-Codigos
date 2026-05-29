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

void BitonicSortSeq(int *vetor, int n)
{
    // Calcula a próxima potência de 2 >= n
    int P = 1;
    while (P < n) P <<= 1;

    // Aloca vetor auxiliar preenchido com INT_MAX (elemento neutro: vai para o fim)
    int *v = new int[P];
    for (int i = 0; i < n; i++) v[i] = vetor[i];
    for (int i = n; i < P; i++) v[i] = INT_MAX;

    // k: tamanho da sequência bitônica atual (2, 4, 8, ..., P)
    for (int k = 2; k <= P; k <<= 1)
    {
        // j: distância de comparação dentro do passo (k/2, k/4, ..., 1)
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = 0; i < P; i++)
            {
                int l = i ^ j;  // índice do par de comparação via XOR
                if (l > i)
                {
                    // Direção de ordenação: crescente quando o bit k do índice i é 0
                    bool crescente = (i & k) == 0;
                    if ((crescente && v[i] > v[l]) || (!crescente && v[i] < v[l]))
                    {
                        int tmp = v[i];
                        v[i] = v[l];
                        v[l] = tmp;
                    }
                }
            }
        }
    }

    // Copia os n primeiros elementos (os INT_MAX ficaram no final) de volta ao vetor original
    for (int i = 0; i < n; i++) vetor[i] = v[i];

    delete[] v;
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
