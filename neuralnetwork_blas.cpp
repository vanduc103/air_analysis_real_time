#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <time.h>
#include <sys/time.h>
#include <malloc.h>
#include <cstdlib>
#include <ctime>
#include "timer.h"

using namespace std;

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

#ifndef LIBBLAS_SO
#define LIBBLAS_SO libblas.so
#endif /* !LIBBLAS_SO */

typedef void (*func_sgemm)(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);

void* GetLibrarayFunction(string function)
{
  void* handle = NULL;

  handle = dlopen(STR(LIBBLAS_SO), RTLD_LAZY);

  if(!handle)
    throw "Could not load library";

  void* Func = dlsym(handle, function.c_str());

  char* result = dlerror();

  if(result)
     throw result;

  return Func;
}

// load gemm function
func_sgemm f = (func_sgemm)GetLibrarayFunction("sgemm_");
char no_trans('n');
float zero(0);
float one(1.0);


#define ReLU(x) (((x)>0)?(x):0)

static void forward_propagation( float *inputs, float *outputs, float *weights, float *biases, int m, int D1, int D2 ) {
    int i, j;

    memset(outputs, 0, sizeof(float) * m * D2);
    int M = m;
    int N = D2;
    int K = D1;
    f(&no_trans, &no_trans, &M, &N, &K, &one, inputs, &M, weights, &K, &zero, outputs, &M);

    for (i = 0; i < D2; i++) {
        float * output = outputs + m * i;
        float bias = biases[i];
        for (j = 0; j < m; j++) {
            output[j] = ReLU(output[j] + bias);
        }
    }
}

void random_generate( int size, float *arr ) {
    srand (static_cast <unsigned> (0));
    for (int i=0; i < size; i++) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        *arr++ = r;
    }
}

void print(float * a, int n) {
    for (int k=0; k < n; k++) cout << a[k] << ",";
    cout << endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <batch size>\n", argv[0]);
        fprintf(stderr, " e.g., %s 200\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    int batch_size = atoi(argv[1]);
    int num_of_features = 244;
    int num_of_output = 1000;

    // read inputs data
    int num_of_rows = 1000; //436424;
    float* inputs = (float *) malloc( num_of_rows * num_of_features * sizeof(float));

    std::ifstream file("train_set.data");
    std::string line;
    for( int i = 0; i < num_of_rows; i++ ) {
        getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);
        for (int j = 0; j < num_of_features; ++j)
        {
            std::string val;
            std::getline(iss, val, ' ');
            if ( !iss.good() ) 
                break;

            std::stringstream convertor(val);
            convertor >> inputs[i*num_of_features + j];
        }
    }
    file.close();

    // randomizely generate weights and biases
    float *weights = (float *) malloc( num_of_features * num_of_output * sizeof(float));
    float *biases = (float *) malloc( num_of_output * sizeof(float));
    random_generate( num_of_features * num_of_output, weights);
    random_generate( num_of_output, biases);
    //print(weights, 10);
    //print(biases, 10);

    // compute forward propagation
    float *outputs = (float *) malloc( batch_size * num_of_output * sizeof(float));
    
    timer_start(0);
    for (int i = 0; i < num_of_rows; i += batch_size) {
        float *input = inputs + i * num_of_features;
        int bi = min( num_of_rows-i, batch_size );

        forward_propagation( input, outputs, weights, biases, bi, num_of_features, num_of_output );

        for (int k=0; k < 10; k++) cout << outputs[k] << ",";
        cout << endl;
    }
    double time_elapsed = timer_end(0);
    printf("Elapsed time: %f sec\n", time_elapsed);
    printf("Average time: %f sec/row\n", time_elapsed/num_of_rows);

    free(inputs);
    free(outputs);
    free(weights);
    free(biases);
    
    return 0;
}

