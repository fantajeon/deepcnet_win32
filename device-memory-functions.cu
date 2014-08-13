#include "device-memory-functions.h"

void cublasError(cublasStatus_t error)
{
	using namespace std;
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		break;

	case CUBLAS_STATUS_NOT_INITIALIZED:
		cout <<  "CUBLAS_STATUS_NOT_INITIALIZED\n";
		break;

	case CUBLAS_STATUS_ALLOC_FAILED:
		cout <<  "CUBLAS_STATUS_ALLOC_FAILED\n";
		break;

	case CUBLAS_STATUS_INVALID_VALUE:
		cout <<  "CUBLAS_STATUS_INVALID_VALUE\n";
		break;

	case CUBLAS_STATUS_ARCH_MISMATCH:
		cout <<  "CUBLAS_STATUS_ARCH_MISMATCH\n";
		break;

	case CUBLAS_STATUS_MAPPING_ERROR:
		cout <<  "CUBLAS_STATUS_MAPPING_ERROR\n";
		break;

	case CUBLAS_STATUS_EXECUTION_FAILED:
		cout <<  "CUBLAS_STATUS_EXECUTION_FAILED\n";
		break;

	case CUBLAS_STATUS_INTERNAL_ERROR:
		cout <<  "CUBLAS_STATUS_INTERNAL_ERROR\n";
		break;

	default:
		cout<< "CUBLAS_UNKNOWN_ERROR\n";
		break;
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. /////////////////////////////////////////////////////////// //////////////////////////////////////////////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (cublasHandle_t handle,
									float* A, float* B, float* C,
									int l, int m, int r,
									float alpha, float beta)
{
	cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N,r,l,m,&alpha,B,r,A,m,&beta,C,r));
	//cudaDeviceSynchronize();
}
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (cublasHandle_t handle,
									 float* A, float* B, float* C,
									 int l, int m, int r,
									 float alpha, float beta)
{
	cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T,r,l,m,&alpha,B,r,A,l,&beta,C,r));
	//cudaDeviceSynchronize();
}
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (cublasHandle_t handle,
									 float* A, float* B, float* C,
									 int l, int m, int r,
									 float alpha, float beta)
{
	cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_N,r,l,m,&alpha,B,m,A,m,&beta,C,r));
	//cudaDeviceSynchronize();
}
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (cublasHandle_t handle,
									  float* A, float* B, float* C,
									  int l, int m, int r,
									  float alpha, float beta)
{
	cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_T,r,l,m,&alpha,B,m,A,l,&beta,C,r));
	//cudaDeviceSynchronize();
}

float peek(float*d ) 
{
	float val;
	d2hMemcopy<float>(d,&val,1);
	return val;
}

void poke(float* d, float val) 
{
	h2dMemcopy<float>(&val,d,1);
}

void canary(const char* file, int linenumber) 
{
	for (int i=0;i<10;i++) {
		float* a=d_allocateArrayZeroed<float>(100,file,linenumber);
		cudaFree(a);
	}
}

