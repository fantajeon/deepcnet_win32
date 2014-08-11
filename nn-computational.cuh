#ifndef __DEFINE_CUDA_NN_COMPUTATIONAL_H__
#define __DEFINE_CUDA_NN_COMPUTATIONAL_H__

template <typename t> __global__ void dReplicateArray(t* d_src, t* d_dest, int size, int nCopies) {
  for (int i=threadIdx.x;i<nCopies;i+=thisManyThreads) {
    for (int j=0;j<size;j++) d_dest[i*size+j]=d_src[j];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dPropForwardToMatrixMultiplyInput(float* d_features, float* d_sgemm, int* rules, int count, int nIn, int fs2);
__global__ void dPropBackwardFromMatrixMultiplyOutput(float* d_deltaGrid, float* d_sgemm, int* rules, int count, int nIn, int fs2);
__global__ void dDropoutFeatures(float* d_features, int* d_featureSampleNumbers, int count, int nIn, float* d_featureWeight);
__global__ void dClassify(float* d_features, int* d_predictions, int batchSize, int nOut);

///////////////////////////////////////////////////////////////////////////////////////////////
//count is a multiple of nOut
__global__ void dMaxout(float* g1a, float* g1b, int count, int kMaxout, unsigned char* d_choice);

__global__ void dMaxoutBackprop(float* d1a, float* d1b, int count, int kMaxout, unsigned char* d_choice);

///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dMaxPool
(float* g1, float* g2, int* rules, int count, int ps2, int nOut, unsigned char* d_choice);

__global__ void dMaxPoolBackprop
(int* rules, float* d1, float* d2, int count, int ps2, int nOut, unsigned char* d_choice);



#endif // #define __DEFINE_CUDA_NN_COMPUTATIONAL_H__