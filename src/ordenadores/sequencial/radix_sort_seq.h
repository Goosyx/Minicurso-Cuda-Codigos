#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;

// ============================================================
//                  Observações gerais
// ============================================================
/*
    Radix Sort LSD sequencial com base 256 (1 byte por passagem).

    Estrutura de cada passagem:
        1) Contagem:    conta quantos elementos têm cada valor de byte
        2) Prefix sum:  converte contagem em posição inicial de cada bucket
        3) Distribuição: copia elementos para a posição correta (de trás
                         para frente, garantindo estabilidade)

    Alternância de ponteiros (ping-pong):
        Após cada passagem, entrada e saída trocam de papel.
        Com 4 passagens (número par), o resultado sempre volta para 'vetor'.
*/

// ============================================================
//              COUNTING SORT POR BYTE (UMA PASSAGEM)
// ============================================================
void CountingSortByte(int *entrada, int *saida, int n, int shift)
{
    int contagem[256] = {0};

    // Fase 1: conta ocorrências de cada valor de byte
    for (int i = 0; i < n; i++)
        contagem[(entrada[i] >> shift) & 0xFF]++;

    // Fase 2: prefix sum — contagem[b] passa a ser a última posição do bucket b
    for (int i = 1; i < 256; i++)
        contagem[i] += contagem[i - 1];

    // Fase 3: distribui de trás para frente (estável)
    for (int i = n - 1; i >= 0; i--)
    {
        int b = (entrada[i] >> shift) & 0xFF;
        saida[--contagem[b]] = entrada[i];
    }
}

// ============================================================
//                  RADIX SORT SEQUENCIAL
// ============================================================
void RadixSortSeq(int *vetor, int n)
{
    int *buffer  = new int[n];
    int *entrada = vetor;
    int *saida   = buffer;

    for (int shift = 0; shift < 32; shift += 8)
    {
        CountingSortByte(entrada, saida, n, shift);

        // Troca os ponteiros: saída desta passagem vira entrada da próxima
        int *temp = entrada;
        entrada   = saida;
        saida     = temp;
    }
    // Após 4 trocas (par), 'entrada' == 'vetor' — resultado já está em vetor

    delete[] buffer;
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
void ExecRadixSeq(const char **entradas, int num_entradas, const char *csv_saida)
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
        RadixSortSeq(vetor, tamanho);
        auto fim = chrono::high_resolution_clock::now();

        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("Radix Sort Sequencial - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "RadixSort - Sequencial,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        delete[] vetor;
    }

    fclose(csv);
}
