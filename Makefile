.PHONY: all profile plot clean clean_dados

all:
	mkdir -p dados
	mkdir -p results
	nvcc -O3 -arch=sm_86 -std=c++14 -o main src/main.cu -lpthread
	./main
	python3 plot_tempos.py
	rm -f dados/*.bin
	rm -f main


# Compila e executa o programa para visualiza-lo no Nsight Systems.
# Tambem exibe no terminal algumas informacoes sobre a execucao
# O arquivo de perfil é salvo em results/perfil.nsys-rep.
profile:
	mkdir -p dados
	mkdir -p results
	nvcc -O3 -arch=sm_86 -std=c++14 -o main src/main.cu -lpthread
	nsys profile --trace=cuda,osrt --output results/perfil ./main
	nsys stats results/perfil.nsys-rep
	rm -f dados/*.bin
	rm -f main


plot:
	python3 plot_tempos.py


clean_dados:
	rm -f dados/*.bin


clean:
	rm -f dados/*.bin
	rm -f results/tempos.csv
	rm -f results/grafico_comparacao.png
	rm -f results/perfil.nsys-rep
	rm -f main

teste:         # só compila, sem profiling
	nsys profile --trace=cuda,osrt --force-overwrite true --output results/perfil ./main

	# 2. Depois, separadamente, gera as estatísticas
	nsys stats --force-export true results/perfil.nsys-rep