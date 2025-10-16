.PHONY: dados clean

all:
	mkdir -p dados
	mkdir -p results
	nvcc -O3 -arch=sm_75 -std=c++14 -o main src/main.cu
	./main
	python3 plot_tempos.py
	rm -f dados/*.bin
	rm main


clean_dados:
	rm -f dados/*.bin


plot:
	python3 plot_tempos.py