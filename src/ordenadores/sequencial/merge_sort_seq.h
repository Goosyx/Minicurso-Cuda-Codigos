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
    Este arquivo implementa o algoritmo Merge Sort Botton Up de forma sequencial na CPU.
    O fluxo geral é:
        1) Ler vetores de arquivos binários (int)
        2) Ordenar os dados usando Merge Sort (bottom-up)
        3) Medir o tempo de execução
        4) Regravar o arquivo com os dados ordenados
        5) Registrar tempos em CSV

    O Merge Sort é um algoritmo de ordenação baseado na técnica "dividir para conquistar".
    Ele divide o vetor em subvetores, ordena cada subvetor e depois mescla (merge) os resultados.
    Esta implementação utiliza a versão iterativa (bottom-up), que evita chamadas recursivas.
*/

// ============================================================
//                  FUNÇÃO DE MESCLAGEM (MERGE)
// ============================================================
/*
 * MergeSeq: mescla dois subvetores adjacentes já ordenados em um único segmento ordenado.
 *
 * Parâmetros:
 * - vetor: ponteiro para o array de inteiros a ser ordenado
 * - buffer: ponteiro para buffer auxiliar pré-alocado (tamanho >= n pelo chamador)
 * - começo: índice inicial do primeiro subvetor
 * - meio: índice final do primeiro subvetor (inclusivo)
 * - fim: índice final do segundo subvetor (inclusivo)
 *
 * Funcionamento:
 * - Utiliza o buffer auxiliar pré-alocado pelo chamador (MergeSortSeq) para
 *   realizar o merge sem alocações dinâmicas por chamada.
 * - Copia os elementos do intervalo [começo, fim] para o buffer, faz o merge
 *   e escreve o resultado de volta no vetor original.
 * - Garante estabilidade na ordenação.
 *
 * ATENÇÃO — motivação desta assinatura:
 *   A versão anterior alocava `new int[]` e `delete[]` a cada chamada de MergeSeq.
 *   Para N=100M elementos, isso representa milhões de alocações de heap durante o sort,
 *   penalizando o modo sequencial artificialmente em relação ao CUDA (que usa um único
 *   buffer pré-alocado em GPU) e ao modo threads. O buffer é agora alocado uma única
 *   vez em MergeSortSeq e reutilizado em todas as chamadas, tornando a comparação justa.
 */
void MergeSeq(int *vetor, int *buffer, int começo, int meio, int fim)
{
    int tam_esquerda = meio - começo + 1;
    int tam_direita = fim - meio;

    // Copia os dois subvetores para o buffer auxiliar pré-alocado.
    // buffer[0..tam_esquerda-1]              ← metade esquerda
    // buffer[tam_esquerda..tam_esquerda+tam_direita-1] ← metade direita
    int idx_esq = 0;
    while(idx_esq < tam_esquerda)
    {
        buffer[idx_esq] = vetor[começo + idx_esq];
        idx_esq++;
    }
    
    int idx_dir = 0;
    while(idx_dir < tam_direita)
    {
        buffer[tam_esquerda + idx_dir] = vetor[meio + 1 + idx_dir];
        idx_dir++;
    }

    // Mescla as duas metades do buffer de volta ao vetor principal
    idx_esq = 0;
    idx_dir = 0;
    int idx = começo;
    while(idx_esq < tam_esquerda && idx_dir < tam_direita)
    {
        if(buffer[idx_esq] <= buffer[tam_esquerda + idx_dir])
        {
            vetor[idx] = buffer[idx_esq];
            idx_esq++;
        } else {
            vetor[idx] = buffer[tam_esquerda + idx_dir];
            idx_dir++;
        }
        idx++;
    }

    // Copia o restante dos elementos, se houver
    while(idx_esq < tam_esquerda)
    {
        vetor[idx] = buffer[idx_esq];
        idx_esq++;
        idx++;
    }

    while(idx_dir < tam_direita)
    {
        vetor[idx] = buffer[tam_esquerda + idx_dir];
        idx_dir++;
        idx++;
    }
}

// ============================================================
//                  FUNÇÃO PRINCIPAL MERGE SORT
// ============================================================
/*
 * MergeSortSeq: ordena um vetor de inteiros usando o algoritmo Merge Sort iterativo (bottom-up).
 *
 * Parâmetros:
 * - vetor: ponteiro para o array de inteiros a ser ordenado
 * - n: número de elementos no array
 *
 * Funcionamento:
 * - Aloca um único buffer auxiliar de tamanho n antes do loop principal.
 *   Este buffer é reutilizado em todas as chamadas a MergeSeq, eliminando
 *   milhões de alocações de heap que ocorriam na versão anterior (onde cada
 *   chamada a MergeSeq fazia new/delete internamente).
 * - Começa com subvetores de tamanho 1 e vai dobrando o tamanho a cada iteração.
 * - Para cada par de subvetores adjacentes, chama MergeSeq para mesclar.
 * - Repete até que todo o vetor esteja ordenado.
 */
void MergeSortSeq(int *vetor, int n)
{
    // Buffer auxiliar pré-alocado: reutilizado em todas as chamadas a MergeSeq.
    // Tamanho n é suficiente pois o maior merge possível envolve o vetor inteiro.
    // Analogia direta com o buffer_device do CUDA, que também é alocado uma única vez.
    int *buffer = new int[n];

    int tamanho = 1;
    while(tamanho < n)
    {
        int inicio = 0;
        while(inicio < n - 1)
        {
            int começo = inicio;
            int meio = min(inicio + tamanho - 1, n - 1);
            int fim = min(inicio + 2 * tamanho - 1, n - 1);
            
            if(meio < fim)
                MergeSeq(vetor, buffer, começo, meio, fim);
            
            inicio += 2 * tamanho;
        }
        
        tamanho *= 2;
    }

    delete[] buffer;
}

// ============================================================
//             FUNÇÃO DE EXECUÇÃO E MEDIÇÃO DE TEMPO
// ============================================================
/*
 * ExecMergeSeq: executa o Merge Sort sequencial para múltiplos arquivos binários
 * contendo inteiros, mede o tempo de ordenação e registra os resultados em CSV.
 *
 * Parâmetros:
 * - entradas: array de caminhos (const char*) para arquivos binários
 * - num_entradas: número de entradas no array
 * - csv_saida: caminho do arquivo CSV de saída onde serão registrados os tempos
 *
 * Funcionamento:
 * - Para cada arquivo:
 *     - Abre o arquivo e determina o número de inteiros
 *     - Lê os dados para um vetor alocado dinamicamente
 *     - Mede o tempo de ordenação usando chrono
 *     - Ordena os dados com MergeSortSeq
 *     - Registra o tempo no arquivo CSV
 *     - Regrava o arquivo com os dados ordenados
 *     - Libera memória utilizada
 */
void ExecMergeSeq(const char **entradas, int num_entradas, const char *csv_saida)
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
        MergeSortSeq(vetor, tamanho);
        auto fim = chrono::high_resolution_clock::now();

        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("Merge Sort Sequencial - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "MergeSort - Sequencial,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, vetor, tamanho);
        delete[] vetor;
    }

    fclose(csv);
}