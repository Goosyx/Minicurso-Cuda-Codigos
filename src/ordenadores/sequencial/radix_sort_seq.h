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
    Este arquivo implementa o algoritmo Radix Sort LSD de forma sequencial na CPU.
    O fluxo geral é:
        1) Ler vetores de arquivos binários (int)
        2) Ordenar os dados usando Radix Sort LSD (base 256)
        3) Medir o tempo de execução
        4) Regravar o arquivo com os dados ordenados
        5) Registrar tempos em CSV

    O Radix Sort é um algoritmo de ordenação não-comparativo que ordena inteiros
    processando cada byte individualmente, do menos significativo para o mais
    significativo (LSD — Least Significant Digit first).

    DECISÃO DE DESIGN — Base 256 (1 byte por passagem):
        Inteiros de 32 bits possuem exatamente 4 bytes, logo o algoritmo sempre
        realiza exatamente 4 passagens, independentemente dos valores presentes
        no vetor. Isso garante comportamento previsível e comparável entre os
        modos sequencial, threads e CUDA.

    OTIMIZAÇÃO — Alternância de ponteiros (ping-pong) sem cópia final:
        A versão anterior copiava o buffer de volta para o vetor ao final de
        cada passagem (O(n) de cópia por passagem = 4n cópias extras no total).
        A versão atual alterna os papéis de 'entrada' e 'saida' a cada passagem:
        passagem 1: lê de vetor,  escreve em buffer
        passagem 2: lê de buffer, escreve em vetor
        passagem 3: lê de vetor,  escreve em buffer
        passagem 4: lê de buffer, escreve em vetor  ← resultado final em vetor
        Após 4 passagens (número par), o resultado está sempre em 'vetor',
        sem nenhuma cópia adicional.
*/

// ============================================================
//              FUNÇÃO COUNTING SORT POR BYTE (AUXILIAR)
// ============================================================
/*
 * CountingSortByte: ordena os elementos de 'entrada' para 'saida' considerando
 * apenas o byte indicado por 'shift'.
 *
 * Parâmetros:
 * - entrada: ponteiro para o array de origem (somente leitura nesta passagem)
 * - saida:   ponteiro para o array de destino (resultado desta passagem)
 * - n:       número de elementos
 * - shift:   número de bits a deslocar para isolar o byte (0, 8, 16 ou 24)
 *
 * Não copia de volta para entrada — isso é responsabilidade do chamador
 * via alternância de ponteiros.
 */
void CountingSortByte(int *entrada, int *saida, int n, int shift)
{
    int contagem[256] = {0};

    // Conta a ocorrência de cada valor do byte na posição 'shift'
    int i = 0;
    while (i < n)
    {
        contagem[(entrada[i] >> shift) & 0xFF]++;
        i++;
    }

    // Prefix sum: contagem[b] passa a indicar a posição inicial do bucket b
    i = 1;
    while (i < 256)
    {
        contagem[i] += contagem[i - 1];
        i++;
    }

    // Distribui de trás para frente para preservar estabilidade
    i = n - 1;
    while (i >= 0)
    {
        int valor_byte = (entrada[i] >> shift) & 0xFF;
        saida[--contagem[valor_byte]] = entrada[i];
        i--;
    }
}

// ============================================================
//                  FUNÇÃO PRINCIPAL RADIX SORT
// ============================================================
/*
 * RadixSortSeq: ordena um vetor de inteiros usando Radix Sort LSD com base 256.
 *
 * Parâmetros:
 * - vetor: ponteiro para o array de inteiros a ser ordenado
 * - n:     número de elementos no array
 *
 * Funcionamento:
 * - Aloca um único buffer auxiliar de tamanho n.
 * - Alterna os papéis de entrada e saída a cada passagem (ping-pong),
 *   eliminando a cópia buffer→vetor da versão anterior.
 * - Após 4 passagens (número par), o resultado está sempre em 'vetor'.
 *
 * Complexidade: O(4n) tempo, O(n) espaço extra.
 */
void RadixSortSeq(int *vetor, int n)
{
    int *buffer = new int[n];

    // Ponteiros que alternam entre vetor e buffer a cada passagem
    int *entrada = vetor;
    int *saida   = buffer;

    int shift = 0;
    while (shift < 32)
    {
        CountingSortByte(entrada, saida, n, shift);

        // Troca os papéis: saída desta passagem vira entrada da próxima
        int *temp = entrada;
        entrada   = saida;
        saida     = temp;

        shift += 8;
    }
    // Após 4 trocas (par), entrada == vetor → resultado já está em vetor

    delete[] buffer;
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
/*
 * ExecRadixSeq: executa o Radix Sort sequencial para múltiplos arquivos binários
 * contendo inteiros, mede o tempo de ordenação e registra os resultados em CSV.
 */
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