LIBBLAS_SO=libblas.so
CXXFLAGS=-Wall -g -O2 -std=c++11 -lstdc++ -DLIBBLAS_SO=$(LIBBLAS_SO) -fopenmp
LDLIBS=-lm -lrt -ldl

all: seq blas packed opt

seq: neuralnetwork_seq
neuralnetwork_seq: timer.o

blas: neuralnetwork_blas
neuralnetwork_blas: timer.o

packed: neuralnetwork_packed
neuralnetwork_packed: timer.o

opt: neuralnetwork_opt
neuralnetwork_opt: timer.o

clean:
	rm -f timer.o neuralnetwork_seq neuralnetwork_blas neuralnetwork_packed neuralnetwork_opt
