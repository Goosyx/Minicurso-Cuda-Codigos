#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define MERGE_THREADS_POR_BLOCO 1

/*
    * MERGE_THREADS_POR_BLOCO: número de threads por bloco CUDA quando lançamos um kernel.
        - É uma configuração de desempenho que normalmente depende da GPU.
        - não necessariamente mais threads significa mais desempenho.
*/ 

using namespace std;


// ============================================================
//                  Observações gerais
// ============================================================
/* 
    Este arquivo implementa um merge sort paralelo executado na GPU usando CUDA. O fluxo geral é:
        1) Ler vetores de arquivos binários (int)
        2) Copiar os dados para a GPU
        3) Executar várias etapas de "merge" (tamanho da sublista dobrando a cada iteração)
        4) Copiar resultado de volta para o host e regravar o arquivo
        5) Registrar tempos em CSV
*/ 




// ============================================================
//                  KERNEL DO MERGE SORT (GPU)
// ============================================================

/*
 * MergeKernel: kernel CUDA que faz o "merge" entre dois blocos
 * já ordenados do array.
 *
 * Parâmetros:
 * - dados: ponteiro para o vetor principal (na memória do device)
 * - buffer_temp:  ponteiro para área temporária (na memória do device)
 * - N: tamanho total do array (número de elementos)
 * - tamanho_atual: tamanho atual da sublista ordenada. A cada iteração do
 *   MergeSortCuda esta variável dobra (1,2,4,8,...)
 *
 * Cada thread é responsável por mesclar duas sublistas adjacentes
 * de tamanho tamanho_atual (ou menores se estivermos nas bordas).
 */
__global__ void MergeKernel(int* dados, int* buffer_temp, int N, int tamanho_atual)
{
    // thread_id: índice global da thread (0..num_threads-1)
    long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcula os índices (em elementos) das duas metades a serem mescladas:
    // Cast explícito para long antes de qualquer multiplicação para evitar overflow de int
    long esquerda  = (long)thread_id * (long)tamanho_atual * 2L;       // esquerda  = início da primeira metade
    long meio      = esquerda + (long)tamanho_atual;                    // meio      = início da segunda metade
    long direita   = (long)(thread_id + 1L) * (long)tamanho_atual * 2L; // direita   = fim (exclusivo) da segunda metade

    // Verificações de segurança para evitar acessar além do array.
    if (esquerda >= N) return;        // nada a fazer se o início estiver fora
    if (meio > N)  meio = N;          // ajusta meio se ultrapassar
    if (direita > N) direita = N;     // ajusta direita se ultrapassar
    if (esquerda >= direita) return;  // segmento inválido (tamanho 0)


    long idx_esq = esquerda;     // idx_esq: ponteiro corrente na metade esquerda
    long idx_dir = meio;         // idx_dir: ponteiro corrente na metade direita
    long idx_atual = esquerda;   // idx_atual: posição corrente onde iremos colocar no buffer_temp


    /* 
        Merge clássico: 
        * Compara o elemento mais à frente das duas metades e escreve o menor em buffer_temp. 
        * Continua até que uma das metades acabe ou até preencher todo o intervalo [esquerda,direita).
    */ 
    while (idx_esq < meio && idx_dir < direita && idx_atual < direita) 
    {
        int a = dados[idx_esq];
        int b = dados[idx_dir];
        if (a <= b) 
        {
            buffer_temp[idx_atual++] = a;
            idx_esq++;
        } else {
            buffer_temp[idx_atual++] = b;
            idx_dir++;
        }
    }

    // Se sobrou elementos na metade esquerda, copia para buffer_temp
    while (idx_esq < meio && idx_atual < direita)
    {
        buffer_temp[idx_atual++] = dados[idx_esq++];
    }

    // Se sobrou elementos na metade direita, copia para buffer_temp
    while (idx_dir < direita && idx_atual < direita)
    {
        buffer_temp[idx_atual++] = dados[idx_dir++];
    }
        

    // Copia o intervalo mesclado de volta para o array original `dados`.
    // Fazemos isso aqui no final do kernel para manter `dados` sempre consistente
    // para iterações futuras (essa escolha evita manter alternância entre dois
    // buffers na CPU; no entanto, dependências de escrita/leitura precisam ser
    // consideradas — aqui cada thread escreve apenas no seu segmento).
    for (long i = esquerda; i < direita && i < N; ++i)
    {
        dados[i] = buffer_temp[i];
    }
       
}

// ============================================================
//          FUNÇÃO DE CONTROLE DO MERGE SORT (GPU)
// ============================================================

/*
 * MergeSortCuda: coordena as chamadas ao kernel para ordenar todo o vetor.
 *
 * Parâmetros:
 * - dados: ponteiro para o vetor no device (GPU)
 * - buffer_temp:  ponteiro para buffer temporário no device
 * - N: tamanho do vetor
 *
 * Funcionamento:
 * - A ideia do merge sort bottom-up: começamos com sublistas de tamanho 1,
 *   depois 2, 4, 8,... até cobrir todo o vetor. Em cada passo, lançamos
 *   um kernel onde cada thread junta (merge) duas sublistas adjacentes de
 *   tamanho `tamanho`.
 */
void MergeSortCuda(int* dados, int* buffer_temp, int N)
{
    if (N <= 1) 
    {
        return;
    }

    // 'tamanho' é o tamanho atual das sublistas ordenadas (1,2,4,8...)
    for (int tamanho = 1; tamanho < N; tamanho <<= 1) 
    {
        // Calcula quantas threads precisamos: cada thread faz o merge de
        // duas sublistas de tamanho `tamanho` => cobre `tamanho*2` elementos.
        long total_threads = (N + (tamanho * 2L - 1L)) / (tamanho * 2L);

        // Converte número de threads em número de blocos considerando
        // MERGE_THREADS_POR_BLOCO por bloco.
        int blocos = (int)((total_threads + MERGE_THREADS_POR_BLOCO - 1) / MERGE_THREADS_POR_BLOCO);

        // Lança o kernel com a configuração (blocos, MERGE_THREADS_POR_BLOCO).
        MergeKernel<<<blocos, MERGE_THREADS_POR_BLOCO>>>(dados, buffer_temp, N, tamanho);

        // Verifica erro de configuração/launch imediatamente após o lançamento.
        // cudaGetLastError() captura erros síncronos (ex: parâmetros inválidos,
        // recursos insuficientes) que ocorrem antes da execução do kernel na GPU.
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess)
        {
            fprintf(stderr, "Erro de launch na iteração tamanho=%d: %s\n", tamanho, cudaGetErrorString(launchErr));
            return;
        }

        // cudaDeviceSynchronize bloqueia a CPU até o kernel terminar na GPU.
        // Qualquer erro de execução do kernel (ex: acesso fora dos limites) é
        // reportado aqui, pois a execução do kernel é assíncrona em relação à CPU.
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) 
        {
            fprintf(stderr, "Erro de execução do kernel na iteração tamanho=%d: %s\n", tamanho, cudaGetErrorString(syncErr));
            return;
        }
    }
}

// ============================================================
//    FUNÇÃO HostParaDeviceMerge() - GERENCIA COPIAS HOST/DEVICE
// ============================================================

/*
 * HostParaDeviceMerge: interface que recebe os dados no host (CPU), aloca memória na GPU,
 * copia os dados para lá, executa MergeSortCuda na GPU e copia o resultado
 * de volta para o host.
 *
 * Parâmetros:
 * - dados_host: ponteiro para array de inteiros na memória do host
 * - N: número de elementos no array
 */
void HostParaDeviceMerge(int* dados_host, int N)
{
    int *dados_device = nullptr;
    int *buffer_device = nullptr; // ponteiros para device (GPU)

    cudaError_t err;

    // Aloca memória para `dados_device` na GPU: espaço para N inteiros
    err = cudaMalloc(&dados_device, N * sizeof(int));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Erro ao alocar dados_device (N=%d): %s\n", N, cudaGetErrorString(err));
        return; // aborta se não conseguiu alocar
    }

    // Aloca memória para buffer temporário `buffer_device` na GPU
    err = cudaMalloc(&buffer_device, N * sizeof(int));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Erro ao alocar buffer_device (N=%d): %s\n", N, cudaGetErrorString(err));
        cudaFree(dados_device);
        return;
    }

    // Copia dados do host (CPU) para o device (GPU)
    err = cudaMemcpy(dados_device, dados_host, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Erro em cudaMemcpy Host->Device (N=%d): %s\n", N, cudaGetErrorString(err));
        cudaFree(dados_device);
        cudaFree(buffer_device);
        return;
    }

    // Chama a rotina que executa o merge sort na GPU (controlada por MergeSortCuda)
    MergeSortCuda(dados_device, buffer_device, N);

    // Copia o resultado ordenado de volta para o host
    err = cudaMemcpy(dados_host, dados_device, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro em cudaMemcpy Device->Host (N=%d): %s\n", N, cudaGetErrorString(err));
    }

    // Libera memória alocada no device
    cudaFree(dados_device);
    cudaFree(buffer_device);
}

// ============================================================
//             FUNÇÕES AUXILIARES DE ENTRADA/SAÍDA
// ============================================================

/*
 * ExecMergeCuda: dado um conjunto de caminhos para arquivos binários contendo
 * inteiros, realiza a ordenação utilizando a GPU e grava tempos em CSV.
 *
 * Parâmetros:
 * - entradas: array de caminhos (const char*) para arquivos binários
 * - num_entradas: número de entradas no array
 * - csv_saida: caminho do arquivo CSV de saída onde serão registrados os tempos
 *
 * Comportamento importante sobre leitura de arquivo:
 * - Usa fseek/ftell para descobrir o tamanho do arquivo (em bytes) e divide
 *   por sizeof(int) para obter o número de inteiros.
 * - Lê todo o arquivo para um vetor alocado dinamicamente (new int[tamanho]).
 */
void ExecMergeCuda(const char **entradas, int num_entradas, const char *csv_saida)
{
    FILE *csv = AbrirCSV(csv_saida);
    if (!csv) return;

    cudaError_t err;

    for (int i = 0; i < num_entradas; i++)
    {
        FILE *file = fopen(entradas[i], "rb+");
        if (!file) { perror(entradas[i]); continue; }

        fseek(file, 0, SEEK_END);
        long tamanho = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        int *v = nullptr;
        err = cudaMallocHost(&v, tamanho * sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Erro ao alocar pinned memory no host (N=%ld): %s\n", tamanho, cudaGetErrorString(err));
            fclose(file);
            continue;
        }

        if (fread(v, sizeof(int), tamanho, file) != (size_t)tamanho)
        {
            perror("Erro ao ler o arquivo");
            fclose(file);
            cudaFreeHost(v);
            continue;
        }

        auto inicio = chrono::high_resolution_clock::now();
        HostParaDeviceMerge(v, tamanho);
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double> decorrido = fim - inicio;
        double tempo = decorrido.count();

        printf("MergeSort CUDA - Tempo para ordenar %s: %f s\n", entradas[i], tempo);
        fprintf(csv, "MergeSort - CUDA,%ld,%f\n", tamanho, tempo);

        GravarEFechar(file, v, tamanho);
        cudaFreeHost(v);
    }

    fclose(csv);
}