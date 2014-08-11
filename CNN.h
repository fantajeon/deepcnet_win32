#pragma once


#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
//#include <unistd.h>
#include <vector>
//#include <boost/assign/list_of.hpp>
//#include <boost/bind.hpp>
//#include <boost/format.hpp>
//#include <boost/random.hpp>
//#include <boost/thread.hpp>
using namespace std;
//using namespace boost::assign;
#include "RNG.h"
#include "nn-computational.h"

extern const int scale_N;
extern const float learningRate;
extern const float learningRateDecayRate;
extern const int trainingBatchSize;
extern const int nInputFeatures;
extern const int startEpoch;
extern const int nCharacters;
extern const char weightFileNameFormat[];
extern const int decreasing_check_epoch;




class SparseCnnInterface {
public:
	batchType type; //TRAINBATCH, TESTBATCH or UNLABELLEDBATCH
	int batchSize;
	bool batchIsEof;
	int count; // Number of feature vectors, _including_ zero-vectors
	vector<int> featureSampleNumbers; //length count, numbers from 0 to batchSize-1.
	vector<float> features; // size count*nInputFeatures,
	vector<int> labels; // only used if labels are known, i.e. during training and testing

	// Each vector<int> represents an NxN array
	vector<vector<int> > grids;
	vector<int> backgroundNullVectorNumbers;
	int nMistakes;
	vector<vector<int> > topGuesses; //used when labels are unknown; find 10 best matches

	SparseCnnInterface (batchType type) : type(type), batchSize(0), count(0), nMistakes(0), batchIsEof(false) {}
};

class Picture {
public:
	virtual void codifyInputData (SparseCnnInterface &interface)=0;
	virtual Picture* distort () {return this;}
	int label; //-1 for unknown
	virtual ~Picture() {}
};

extern "C" std::vector<Picture*> *g_ptrainCharacters;
extern "C" std::vector<Picture*> *g_ptestCharacters;
extern "C" std::vector<float> *g_pregularizingConstants;


void loadData(); //Application specific loading mechanism
void replaceTestSetWithValidationSet(float p = 0.8);
void smallerTestSet(float p = 0.03);
void calculateRegularizingConstants(int nInputFeatures);



#include "cuda.h"
#include <cublas_v2.h>
#include "device-memory-functions.h"
#include "convolutional-layer.h"


//extern boost::mutex CNNmutex;
extern cublasHandle_t cublasHandle;
extern cudaStream_t cublasstream;

class CNN {
public:
	vector<ConvolutionalLayer> L;
	float learningRate;
	float momentumDecayRate;
	float weightDecayRate;
	deque<float> trainError;
	deque<float> testError;
	int nInputFeatures;
	int nInputSpatialSize;
	int epoch;

	CNN(int nInputFeatures, int nInputSpatialSize, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch);
	~CNN();

	void saveWeights();
	void loadWeights();	
	void addLayer(int filterSize, int poolSize, int nFilters, float learningRateScale, sigmoidType sigmoidFunction, float dropoutProbability=0, int kMaxout=1);
	void initialize();
};


class DeepCNet : public CNN {
public:
	DeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
		float momentumDecayRate, float weightDecayRate, int epoch,
		vector<float>	learningRateScales,
		vector<float> dropoutProbabilities = vector<float>(),
		vector<int> kMaxouts = vector<int>());
};

class DeepDeepCNet : public CNN {
public:
	DeepDeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
		float momentumDecayRate, float weightDecayRate, int epoch,
		vector<float> dropoutProbabilities = vector<float>(),
		vector<int> kMaxouts = vector<int>());
};


class FlatDeepCNet : public CNN {
public:
	FlatDeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
		float momentumDecayRate, float weightDecayRate, int epoch,
		vector<float> dropoutProbabilities = vector<float>(),
		vector<int> kMaxouts = vector<int>());
};


class LeNet5 : public CNN { //scale_N=28 or 32
public:
	LeNet5(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch);
};

class LeNet7 : public CNN {//scale_N=96
public:
	LeNet7(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch, vector<float>	learningRateScales);
};

class LeNetJ : public CNN {//scale_N=96
public:
	LeNetJ(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch, vector<float>	learningRateScales);
};


//#include "nn-computational.h"

class ComputationalCNN {
public:
	RNG rng;
	CNN &nn;
	vector<ConvolutionalComputationalLayerBase*> CL;
	SparseCnnInterface* data;
	ConvolutionalComputationalLayerInterface input;

	ComputationalCNN(CNN &nn);
	void buildSparseProcessingRulesOnCPU();
	void copySparseDataToGPU();
	void forwardPropagate();
	void test();
	void findTopTenGuesses();
	void backwardPropagate();
	void applyDerivatives();
	void cleanUp();
	void cleanUPFoward();
	void processBatch(SparseCnnInterface *d);
	void processRandomBatch(SparseCnnInterface *d);

	void backup(double cost);
	void applyBestCost();
	void randomWeight(double range);
	void restoreWeight(int idx);
};



void init_cnn();
void deinit_cnn();