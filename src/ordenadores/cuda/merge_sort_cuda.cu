// merge_cuda.cu
// Compilar: nvcc -O3 -arch=sm_80 merge_cuda.cu -o merge_cuda
// (ajuste -arch=sm_XX conforme sua GPU; se preferir, remova -arch e deixe o nvcc decidir)

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

// -----------------------------
// Merge-Path helpers (device)
// -----------------------------
//
// A função merge_path_partition encontra (i, j) tal que
// i é o número de elementos tomados de A (começando em 0) para formar
// a prefix output de tamanho 'diag' dentre A (length m) e B (length n).
//
// Parâmetros:
//  A, m  -> primeiro array (tamanho m)
//  B, n  -> segundo array (tamanho n)
//  diag  -> índice no array resultado combinado (0..m+n)
//
// Retorna i (número de elementos de A no prefix), e j = diag - i para B.
//
// Observação: limites considered: i ∈ [max(0, diag - n), min(diag, m)]
// Algoritmo: binary search sobre i.
// -----------------------------
__device__ inline int merge_path_partition(const int* A, int m, const int* B, int n, int diag) {
    int low = max(0, diag - n);
    int high = min(diag, m);
    // Busca binária em [low, high]
    while (low < high) {
        int mid = (low + high) >> 1; // (low+high)/2
        int a_val = A[mid];          // A[mid]
        int b_val = B[diag - mid - 1]; // B[diag-mid-1]
        // Se A[mid] <= B[diag-mid-1], então podemos aumentar low
        // (isso tenta colocar mais elementos de A no prefix)
        if (a_val <= b_val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// -----------------------------
// Kernel de merge de pares (cada bloco cuida de um par de runs).
//
// Parâmetros:
//  src      : ponteiro para buffer de entrada (contém runs ordenados de tamanho width)
//  dst      : ponteiro para buffer de saída (escreve runs de tamanho 2*width)
//  N        : tamanho total do array
//  width    : tamanho da run esquerda (direita também terá até width, pode ser menor no final)
// -----------------------------
__global__ void merge_pairs_kernel(const int* src, int* dst, int N, int width) {
    // índice do par que esse bloco deve processar
    // (cada par cobre [left, left+2*width) )
    int pairIdx = blockIdx.x;
    int left = pairIdx * (2 * width);
    if (left >= N) return; // sem trabalho

    // limites dos subarrays
    int mid = min(left + width, N);            // início do segundo run
    int right = min(left + 2 * width, N);      // fim (exclusivo) do segundo run

    int m = mid - left;    // tamanho do primeiro run (A)
    int n = right - mid;   // tamanho do segundo run (B)
    if (m <= 0) return;    // nada a fazer
    // pointers para as subarrays na memória src
    const int* A = src + left;
    const int* B = src + mid;
    int total = m + n;

    // cada thread no bloco processa um "chunk" do output [0..total)
    int T = blockDim.x;
    int tid = threadIdx.x;

    // chunk size (divisão simples dos total entre threads)
    int chunk = (total + T - 1) / T;
    int start_diag = tid * chunk;
    int end_diag = min(total, (tid + 1) * chunk);
    if (start_diag >= end_diag) return; // sem trabalho para esta thread

    // Determina partições de A/B para start_diag e end_diag:
    int i_start = merge_path_partition(A, m, B, n, start_diag);
    int j_start = start_diag - i_start;

    int i_end = merge_path_partition(A, m, B, n, end_diag);
    int j_end = end_diag - i_end;

    // Posições de escrita no buffer de saída:
    int out_start = left + start_diag; // offset global onde esta thread começa a escrever
    int write_pos = out_start;

    // Agora faremos uma fusão sequencial entre A[i_start..i_end) e B[j_start..j_end)
    // e escreveremos para dst[write_pos ... write_pos + (end_diag-start_diag) - 1]
    int ia = i_start;
    int jb = j_start;

    while (ia < i_end && jb < j_end) {
        if (A[ia] <= B[jb]) {
            dst[write_pos++] = A[ia++];
        } else {
            dst[write_pos++] = B[jb++];
        }
    }
    // copia restos de A
    while (ia < i_end) {
        dst[write_pos++] = A[ia++];
    }
    // copia restos de B
    while (jb < j_end) {
        dst[write_pos++] = B[jb++];
    }
    // fim da thread
}

// -----------------------------
// Função host: mergeSortCuda
// - Aloca buffers no device
// - Copia dados para device
// - Executa passes: width = 1,2,4,... até >=N
// - Usa merge_pairs_kernel para cada passada (cada bloco = 1 par)
// - Faz ping-pong entre d_src e d_dst
// - Copia resultado de volta para host
// -----------------------------
void mergeSortCuda(int *h_arr, long N) {
    if (N <= 1) return;

    int *d_src = nullptr;
    int *d_dst = nullptr;

    size_t bytes = N * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, bytes));

    // Copia array de entrada para d_src
    CUDA_CHECK(cudaMemcpy(d_src, h_arr, bytes, cudaMemcpyHostToDevice));

    // Parâmetros de execução:
    // threads por bloco (por par): heurística inicial -> 256 (ajustável)
    const int threadsPerBlock = 256;

    // Ping-pong: src->dst, swap, repetidamente
    int *src = d_src;
    int *dst = d_dst;

    // Número de passes = ceil(log2(N)), mas iteramos até width >= N
    for (int width = 1; width < N; width <<= 1) {
        // número de pares (cada par cobre 2*width elementos)
        int numPairs = (N + 2 * width - 1) / (2 * width);

        // grid: um bloco por par
        dim3 grid(numPairs);
        dim3 block(threadsPerBlock);

        // Lançamento do kernel
        merge_pairs_kernel<<<grid, block>>>(src, dst, (int)N, width);

        // Checar erro de lançamento
        CUDA_CHECK(cudaGetLastError());
        // Sincroniza para garantir término antes de swap e próxima iteração (poderia usar streams/assíncrono)
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        int *tmp = src;
        src = dst;
        dst = tmp;
    }

    // Após o loop, 'src' contém o array ordenado (pode ser d_src ou d_dst dependendo do número de passes)
    // Copia de volta para host
    CUDA_CHECK(cudaMemcpy(h_arr, src, bytes, cudaMemcpyDeviceToHost));

    // libera
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// -----------------------------
// Código de suporte: funções do seu código base adaptadas
// -----------------------------
void merge_cpu(int *v, int p, int q, int r) {
    int n1 = q - p + 1;
    int n2 = r - q;

    int *esq = new int[n1];
    int *dir = new int[n2];

    for (int i = 0; i < n1; i++) esq[i] = v[p + i];
    for (int j = 0; j < n2; j++) dir[j] = v[q + 1 + j];

    int i = 0, j = 0, k = p;
    while (i < n1 && j < n2) {
        if (esq[i] <= dir[j]) {
            v[k] = esq[i++];
        } else {
            v[k] = dir[j++];
        }
        k++;
    }
    while (i < n1) v[k++] = esq[i++];
    while (j < n2) v[k++] = dir[j++];

    delete[] esq;
    delete[] dir;
}

void MergeSort_cpu(int *v, int p, int r) {
    if (p < r) {
        int m = (p + r) / 2;
        MergeSort_cpu(v, p, m);
        MergeSort_cpu(v, m + 1, r);
        merge_cpu(v, p, m, r);
    }
}

// Função para execução e medição (adaptada do seu exec_merge)
void exec_merge_cuda(const char **entradas, int num_entradas, const char *csv_saida) {
    FILE *csv = fopen(csv_saida, "a");
    if (!csv) {
        perror("Erro ao abrir arquivo CSV");
        return;
    }

    for (int i = 0; i < num_entradas; i++) {
        FILE *file = fopen(entradas[i], "rb+");
        if (!file) {
            perror(entradas[i]);
            continue;
        }

        fseek(file, 0, SEEK_END);
        long tamanho = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        if (tamanho <= 0) {
            fprintf(stderr, "Arquivo %s vazio ou inválido\n", entradas[i]);
            fclose(file);
            continue;
        }

        int *v = new int[tamanho];
        if (fread(v, sizeof(int), tamanho, file) != (size_t)tamanho) {
            perror("Erro ao ler o arquivo");
            fclose(file);
            delete[] v;
            continue;
        }

        // Medição GPU
        auto start = std::chrono::high_resolution_clock::now();
        mergeSortCuda(v, tamanho);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double gpu_time = elapsed.count();

        printf("Merge Sort CUDA - Tempo para ordenar %s (N=%ld): %f segundos\n", entradas[i], tamanho, gpu_time);
        fprintf(csv, "MergeSort CUDA,%ld,%f\n", tamanho, gpu_time);

        // escreve resultado ordenado de volta no arquivo
        fseek(file, 0, SEEK_SET);
        if (fwrite(v, sizeof(int), tamanho, file) != (size_t)tamanho) {
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

// -----------------------------
// Programa principal (main) - adaptado do seu main
// -----------------------------
int main() {
    // Ajuste: coloque aqui o número correto de entradas que você pretende usar
    const int num_entradas = 11;
    const char *entradas[num_entradas] = {
        "dados/250k.bin", "dados/500k.bin", "dados/750k.bin", "dados/1m.bin",
        "dados/2m500.bin", "dados/5m.bin", "dados/7m500.bin", "dados/10m.bin",
        "dados/25m.bin", "dados/50m.bin", "dados/100m.bin"
    };

    // Prepara CSV
    FILE *csv = fopen("results/tempos.csv", "w");
    if (csv) {
        fprintf(csv, "Algoritmo,Tamanho,Tempo\n");
        fclose(csv);
    }

    // Executa a versão CUDA (substitui exec_merge)
    exec_merge_cuda(entradas, num_entradas, "results/tempos.csv");

    return 0;
}
