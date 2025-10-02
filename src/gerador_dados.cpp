#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

/**
 * Gera três arquivos binários com números aleatórios:
 * - dados/pequeno.bin 20.000 elementos
 * - dados/medio.bin   100.000 elementos
 * - dados/grande.bin  265.000 elementos
 */

void gerar_arquivo(const long *tamanho_arquivos, const char **nomes_arquivos, const int num_arquivos) 
{
    for (int i = 0; i < num_arquivos; i++) 
    {
        long n = tamanho_arquivos[i];
        const char *path = nomes_arquivos[i];

        FILE *file = fopen(path, "wb");
        if (!file) 
        {
            perror(path);
            continue;
        }

        for (long j = 0; j < n; j++) 
        {
            int num = rand() % 100000000;  // gera número aleatório entre 0 e o maior inteiro
            if (fwrite(&num, sizeof(int), 1, file) != 1) 
            {
                perror("Erro ao escrever no arquivo");
                fclose(file);
                break;
            }
        }

        fclose(file);
        printf("Gerado: %s com %ld inteiros\n", path, n);
    }

    printf("\n");

}

int main() 
{
    int num_arquivos = 11;

    long tamanho_arquivos[num_arquivos] = {
        250000, 500000, 750000, 1000000,
        2500000, 5000000, 7500000, 10000000,
        25000000, 50000000, 100000000
    };
    const char *nomes_arquivos[num_arquivos]  = {
        "dados/250k.bin", "dados/500k.bin", "dados/750k.bin", "dados/1m.bin",
        "dados/2m500.bin", "dados/5m.bin", "dados/7m500.bin", "dados/10m.bin",
        "dados/25m.bin", "dados/50m.bin", "dados/100m.bin"

    };

    gerar_arquivo(tamanho_arquivos, nomes_arquivos, num_arquivos);

}
