#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

void GerarArquivos(const long *tamanho_arquivos, const char **nomes_arquivos, const int num_arquivos) 
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
