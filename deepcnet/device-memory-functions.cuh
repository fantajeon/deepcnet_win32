#ifndef __DEFINE_DEVICEMEMORY_FUNCTIONS_H__
#define __DEFINE_DEVICEMEMORY_FUNCTIONS_H__

#pragma disable(warning:4819)

#pragma once

#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>



using namespace std;

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}


template <typename t> __global__ void check_memory(t* d_ptr)
{
	if( isnan(d_ptr[threadIdx.x]) || isinf(d_ptr[threadIdx.x]) ) {
		d_ptr[threadIdx.x] = 0.0f;
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename t> t* d_allocateArray(int size, const char* file = 0, int linenumber = 0)
{
	//cout << file << " " << linenumber<<endl;
	t* d_ptr;
	if (cudaSuccess != cudaMalloc((void**) &d_ptr, sizeof(t)*size)) {
		cout<< "cudaMalloc error with size("<<size<<")";
		if (file != 0) 
			cout << " Called from file: " << file << " linenumber: " << linenumber << endl;
		cout << endl;
		exit(1);
	}
	return d_ptr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename t> void d_zeroArray(t* d_ptr, int count) {
	cudaMemset(d_ptr,  0, sizeof(t)*count);
}


template <typename t> t* d_allocateArrayZeroed(int size, const char* file = 0, int linenumber = 0)
{
	t* d_ptr = d_allocateArray<t>(size, file, linenumber);
	d_zeroArray<t>(d_ptr, size);
	return d_ptr;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename t> void h2dMemcopy(t* h_ptr, t* d_ptr, int size)
{
	int copylen = sizeof(t)*size;
	if( cudaSuccess!=cudaMemcpy(d_ptr, h_ptr, copylen, cudaMemcpyHostToDevice)) {
		__debugbreak();
	}
}

template <typename t> void d2hMemcopy(t* d_ptr, t* h_ptr, int size)
{
	int copylen = sizeof(t)*size;
	cudaMemcpy(h_ptr, d_ptr, copylen, cudaMemcpyDeviceToHost);
}

template <typename t> t* d_allocateArrayFromVector( std::vector<t> &source, const char* file = 0, int linenumber = 0) 
{
	t* d_ptr = d_allocateArray<t>(source.size(), file, linenumber);
	h2dMemcopy<t>( &source[0], d_ptr, source.size());
	return d_ptr;
}

void cublasError(cublasStatus_t error);

//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. /////////////////////////////////////////////////////////// //////////////////////////////////////////////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (cublasHandle_t handle,
									float* A, float* B, float* C,
									int l, int m, int r,
									float alpha, float beta);
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (cublasHandle_t handle,
									 float* A, float* B, float* C,
									 int l, int m, int r,
									 float alpha, float beta);
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (cublasHandle_t handle,
									 float* A, float* B, float* C,
									 int l, int m, int r,
									 float alpha, float beta);
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (cublasHandle_t handle,
									  float* A, float* B, float* C,
									  int l, int m, int r,
									  float alpha, float beta);

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename t>  void printMatrix(vector<t> &m, int rows, int cols, int maxVal=10, std::string s="")
{
	cout << s.c_str() << "-----------" << std::endl;
	for (int r=0; r<rows && r<maxVal;r++) {
		for (int c=0;c<cols && c<maxVal;c++) {
			cout << m[r*cols+c] << "\t ";}
		cout <<"\n";}
	cout <<"--------------------------------------------------------------------------\n";
}

template <typename t> void d_printMatrix(t* d_ptr, int rows, int cols, int maxVal=10, std::string s="")
{
	vector<t> m(rows*cols);
	d2hMemcopy<t>(d_ptr,&m[0],rows*cols);
	printMatrix<t>(m,rows,cols,maxVal,s);
}

template <typename t> void printV(std::vector<t> v) 
{
	for(int i=0;i<v.size();i++)
		cout << v[i] << " ";
	cout <<endl;
}


template <typename t> void printVV(std::vector<vector<t> > v) 
{
	cout << "<<"<<endl;
	for(int i=0;i<v.size();i++)
		printV(v[i]);
	cout <<">>"<<endl;
}


float peek(float*d );
void poke(float* d, float val);
void canary(const char* file = 0, int linenumber = 0);


#endif // #define __DEFINE_DEVICEMEMORY_FUNCTIONS_H__