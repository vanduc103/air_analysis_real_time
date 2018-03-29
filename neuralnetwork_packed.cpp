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

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Block sizes */
int mc = 512;
int kc = 256;

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot4x4( int, float *, int, float *, int, float *, int );

void PackMatrixA( int, float *, int, float * );
void PackMatrixB( int, float *, int, float * );
void InnerKernel( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc );

int MR = 4;
int NR = 4;

static void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, p, pb, ib; 
  printf("m = %d, n = %d, k = %d\n", m, n, k);

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */

  for ( p=0; p<k; p+=kc ){
    pb = min( k-p, kc );
    for ( i=0; i<m; i+=mc ){
      ib = min( m-i, mc );
      InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc );
    }
  }
}

void InnerKernel( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc )
{
  int i, j;
  float
    *packedA = ( float * ) malloc( (m*k) * sizeof( float ) );
  float
    *packedB = ( float * ) malloc( (n*k) * sizeof( float ) );
  
  for ( j=0; j<n; j+=NR ) {        /* Loop over the columns of C, unrolled by 4 */
    PackMatrixB( k, &B( 0, j ), NR, &packedB[ k*j ] );

    for ( i=0; i<m; i+=MR ) {        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if ( j == 0 ) {
        PackMatrixA( k, &A( i, 0 ), MR, &packedA[ i*k ] );
      }
      AddDot4x4( k, &packedA[ i*k ], lda, &packedB[ k*j ], ldb, &C( i,j ), ldc );
    }
  }
}

void PackMatrixA( int k, float *a, int lda, float *a_to )
{
  int j, i;

  for( j=0; j<k; j++){  /* loop over columns of A */
    float 
      *a_ij_pntr = &A( 0, j );

    for (i=0; i<lda; i++) {
        *a_to++ = *(a_ij_pntr+i);
    }
  }
}

void PackMatrixB( int k, float *b, int ldb, float *b_to )
{
  int i, j;
  float *b_pntr[ ldb ];
  for( j=0; j<ldb; j++) {
    b_pntr[ j ] = &B( 0, j );
  }

  for( i=0; i<k; i++) {  /* loop over rows of B */
    for( j=0; j<ldb; j++) {
        *b_to++ = *( b_pntr[ j ]++ );
    }
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128 v;
  float d[4];
} v4df_t;

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix C

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */

  int p;
  v4df_t
    c_00_c_30_vreg,    c_01_c_31_vreg,    c_02_c_32_vreg,    c_03_c_33_vreg,
    a_0p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;
  /*float 
    *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 

  b_p0_pntr = &B( 0, 0 );
  b_p1_pntr = &B( 0, 1 );
  b_p2_pntr = &B( 0, 2 );
  b_p3_pntr = &B( 0, 3 );*/

  c_00_c_30_vreg.v = _mm_setzero_ps();   
  c_01_c_31_vreg.v = _mm_setzero_ps();
  c_02_c_32_vreg.v = _mm_setzero_ps(); 
  c_03_c_33_vreg.v = _mm_setzero_ps(); 

  for ( p=0; p<k; p++ ) {
    // load 4 words from a
    a_0p_a_3p_vreg.v = _mm_load_ps( (float *) ( a ) );
    a += 4;

    //b_p0_vreg.v = _mm_load1_ps( (float *) b_p0_pntr++ );   /* load and duplicate */
    //b_p1_vreg.v = _mm_load1_ps( (float *) b_p1_pntr++ );   /* load and duplicate */
    //b_p2_vreg.v = _mm_load1_ps( (float *) b_p2_pntr++ );   /* load and duplicate */
    //b_p3_vreg.v = _mm_load1_ps( (float *) b_p3_pntr++ );   /* load and duplicate */
    b_p0_vreg.v = _mm_load1_ps( (float *) (b) );   /* load and duplicate */
    b_p1_vreg.v = _mm_load1_ps( (float *) (b+1) );   /* load and duplicate */
    b_p2_vreg.v = _mm_load1_ps( (float *) (b+2) );   /* load and duplicate */
    b_p3_vreg.v = _mm_load1_ps( (float *) (b+3) );   /* load and duplicate */
    b+= 4;

    /* All 4 rows */
    c_00_c_30_vreg.v += a_0p_a_3p_vreg.v * b_p0_vreg.v;
    c_01_c_31_vreg.v += a_0p_a_3p_vreg.v * b_p1_vreg.v;
    c_02_c_32_vreg.v += a_0p_a_3p_vreg.v * b_p2_vreg.v;
    c_03_c_33_vreg.v += a_0p_a_3p_vreg.v * b_p3_vreg.v;

  }

  C( 0, 0 ) += c_00_c_30_vreg.d[0];  C( 0, 1 ) += c_01_c_31_vreg.d[0];  
  C( 0, 2 ) += c_02_c_32_vreg.d[0];  C( 0, 3 ) += c_03_c_33_vreg.d[0]; 

  C( 1, 0 ) += c_00_c_30_vreg.d[1];  C( 1, 1 ) += c_01_c_31_vreg.d[1];  
  C( 1, 2 ) += c_02_c_32_vreg.d[1];  C( 1, 3 ) += c_03_c_33_vreg.d[1]; 

  C( 2, 0 ) += c_00_c_30_vreg.d[2];  C( 2, 1 ) += c_01_c_31_vreg.d[2];  
  C( 2, 2 ) += c_02_c_32_vreg.d[2];  C( 2, 3 ) += c_03_c_33_vreg.d[2]; 

  C( 3, 0 ) += c_00_c_30_vreg.d[3];  C( 3, 1 ) += c_01_c_31_vreg.d[3];  
  C( 3, 2 ) += c_02_c_32_vreg.d[3];  C( 3, 3 ) += c_03_c_33_vreg.d[3]; 
}

#define ReLU(x) (((x)>0)?(x):0)

static void forward_propagation( float *inputs, float *outputs, float *weights, float *biases, int m, int D1, int D2 ) {
    int i, j;

    memset(outputs, 0, sizeof(float) * m * D2);
    int M_packed = m;
    int N_packed = D2;
    int K_packed = D1;
    MY_MMult( M_packed, N_packed, K_packed, inputs, M_packed, 
                                    weights, K_packed,
                                    outputs, M_packed );

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

