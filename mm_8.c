#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "time.h"
#include "cblas.h"

#define ROUND 5

int main(int argc, char* argv[])
{
	if (argc < 4) {
		printf("Input error, ./mm M N K\n");
		return 1;
	}

	openblas_set_num_threads(8);

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	int sizeofa = m * k;
	int sizeofb = k * n;
	int sizeofc = m * n;

	struct timeval start, finish;
	long duration = 0;
	long accumulate_duration = 0;

	double *A = (double*)malloc(sizeof(double) * sizeofa);
	double *B = (double*)malloc(sizeof(double) * sizeofb);
	double *C = (double*)malloc(sizeof(double) * sizeofc);

	srand(113);
	double alpha = ((double)rand()) / RAND_MAX;
	double beta = ((double)rand()) / RAND_MAX;
	for (int i = 0; i < sizeofa; i++)	A[i] = ((double)rand()) / RAND_MAX;
	for (int i = 0; i < sizeofb; i++)	B[i] = ((double)rand()) / RAND_MAX;
	for (int i = 0; i < sizeofc; i++)	C[i] = ((double)rand()) / RAND_MAX;
	
	printf("M=%d N=%d K=%d\n", m, n, k);
	for (int round = 0; round < ROUND; round++) {
		gettimeofday(&start, NULL);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
		gettimeofday(&finish, NULL);

		duration = ((long)(finish.tv_sec - start.tv_sec))*1000000 + (long)(finish.tv_usec - start.tv_usec);
		printf("Round %d: %ld ms\n", round, duration/1000);
		accumulate_duration += duration;
	}
	long average_duration = accumulate_duration / ROUND; // 1 us = 1.0e-6 s
	printf("M * N * (1+3*K) = %ld flops, time = %ld us\n", (long)m*(long)n*(long)(1+3*k), average_duration);
	printf("Performance: %.3f GFLOPs/s\n", 1.0e-3*m*n*(1+3*k) / average_duration);
	
	free(A);
	free(B);
	free(C);
	return 0;
}

