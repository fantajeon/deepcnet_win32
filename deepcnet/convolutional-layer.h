#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include "device-memory-functions.h"
#include "RNG.h"


//extern const int thisManyThreads;

enum sigmoidType             {NOSIGMOID,   LOGISTIC,   RECTIFIEDLINEAR,   TANH,   SOFTMAX,   BLOCKY };


//kMaxout==1 when applying a nonlinear function (applied after maxout anyway)
__global__ void dSigmoidLogistic(float* g, int count, int nOut);
__global__ void dSigmoidRectifiedLinear(float* g, int count, int nOut);
__global__ void dSigmoidBlocky(float* g, int count, int nOut);
__global__ void dSigmoidTanh(float* g, int count, int nOut);
__global__ void dSigmoidSoftmax(float* g, int count, int nOut);

//SOFTMAX only occurs at the top layer;
//derivative contained in calculation of initial d_delta.
__global__ void dSigmoidBackpropLogistic(float* d, float* g, int count, int N);
__global__ void dSigmoidBackpropRectifiedLinear(float* d, float* g, int count, int N);
__global__ void dSigmoidBackpropBlocky(float* d, float* g, int count, int N);
__global__ void dSigmoidBackpropTanh(float* d, float* g, int count, int N);

////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dColumnSum(float* matrix, float* target, int nRows, int nColumns);

////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate softmaxProbability(i) - indicator(i=label)
// for i=0,1,...N-1 with N the number of character classes.
__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights(int batchSize, float* topDelta, float* topGrid,
	int* labels, int N);
//
/////////////////////////////////////////////////////////////////////////////////////////////////
//__global__  void dGradientDescent //momentum
//	(float* d_delta, float* d_momentum, float* d_weights, int N,
//	float learningRate, float momentumDecayRate, float weightDecayRate);

///////////////////////////////////////////////////////////////////////////////////////////////////
//   _____                      _       _   _                   _ _
//  / ____|                    | |     | | (_)                 | | |
// | |     ___  _ ____   _____ | |_   _| |_ _  ___  _ __   __ _| | |     __ _ _   _  ___ _ __
// | |    / _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| '_ \ / _` | | |    / _` | | | |/ _ \ '__|
// | |___| (_) | | | \ V / (_) | | |_| | |_| | (_) | | | | (_| | | |___| (_| | |_| |  __/ |
//  \_____\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|\__,_|_|______\__,_|\__, |\___|_|
//                                                                             __/ |
//                                                                            |___/
class ConvolutionalLayer 
{
public:
	int filterSize;
	int poolSize;
	int s0,s1,s2;
	int nIn;
	int nOut;
	float dropoutProbability;
	int kMaxout;
	sigmoidType sigmoid;
	int fs2;
	int ps2;
	std::vector<float> W;
	std::vector<float> B;
	float* d_W;
	float* d_B;
	float* d_momentumW;
	float* d_momentumB;
	float learningRateScale;
	float* d_lrW;
	float* d_lrB;
	float* d_W1;
	float* d_B1;
	float* d_dW1;
	float *d_dB1;
	float *d_weightDecayRateW;
	float *d_weightDecayRateB;
	int nloop;

	__host__ void loadWeightsFromStream(std::ifstream &f);
	__host__ void putWeightsToStream(std::ofstream &f);
	__host__ ConvolutionalLayer(int fs, int ps, int s0, int s1, int s2, int in, int out, float learningRateScale, sigmoidType sig, float dropoutProbability=0, int kMaxout=1);
	__host__ void applyDerivatives(float* d_deltaW, float* d_deltaB, float learningRate, float momentumDecayRate, float weightDecayRate);
	__host__ void checkapplyDerivatives(float* d_deltaW, float* d_deltaB, float learningRate, float momentumDecayRate, float weightDecayRate);
	__host__ void constraintWeight();
};
