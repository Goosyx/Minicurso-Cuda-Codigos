.PHONY: dados clean

dados:
	mkdir -p dados
	g++ -std=c++17 -O2 -o gerador_dados src/gerador_dados.cpp
	./gerador_dados
	rm gerador_dados

clean_dados:
	rm -f dados/*.bin

merge_seq:
	mkdir -p results
	g++ -std=c++17 -O2 -o merge_sort_seq src/ordenadores/sequencial/merge_sort_seq.cpp
	./merge_sort_seq
	rm merge_sort_seq

merge_thread:
	mkdir -p results
	g++ -std=c++17 -O2 -o merge_sort_threads src/ordenadores/threads/merge_sort_threads.cpp
	./merge_sort_threads
	rm merge_sort_threads



plot:
	python3 plot_tempos.py

radix_seq:
	mkdir -p results
	g++ -std=c++17 -O2 -o radix_sort_seq src/ordenadores/sequencial/radix_sort_seq.cpp
	./radix_sort_seq
	rm radix_sort_seq

cuda:
	nvcc -O3 -arch=sm_75 -std=c++14 src/ordenadores/cuda/merge_sort_cuda.cu -o merge_sort_cuda
	./merge_sort_cuda
	rm merge_sort_cuda