#pragma once


#include "convolutional-layer.h"

enum batchType {TRAINBATCH, TESTBATCH, UNLABELLEDBATCH};

///////////////////////////////////////////////////////////////////////////////////////////////
//   _____                             _             _
//  / ____|                           | |           (_)
// | (___  _ __   __ _ _ __ ___  ___  | | ___   __ _ _  ___
//  \___ \| '_ \ / _` | '__/ __|/ _ \ | |/ _ \ / _` | |/ __|
//  ____) | |_) | (_| | |  \__ \  __/ | | (_) | (_| | | (__
// |_____/| .__/ \__,_|_|  |___/\___| |_|\___/ \__, |_|\___|
//        | |                                   __/ |
//        |_|                                  |___/

class ConvolutionalComputationalLayerInterface 
{
public:
	float* d_features;
	batchType type;
	int batchSize;
	vector<int> featureSampleNumbers;
	vector<int> backgroundNullVectorNumbers;
	vector<vector<int> > grids;
	int count;
	ConvolutionalComputationalLayerInterface () : count(0) {}
};


class ConvolutionalComputationalLayerBase 
{
public:
	RNG rng;
	ConvolutionalLayer& L;
	int level;
	float* d_deltaW;//     (d cost)/(d W)
	float* d_deltaB;//     (d cost)/(d B)
	ConvolutionalComputationalLayerInterface &input;
	ConvolutionalComputationalLayerInterface middle;
	ConvolutionalComputationalLayerInterface output;
	unsigned char* d_maxPoolChoice;
	float* d_featuresToMaxout;
	unsigned char* d_maxoutChoice;
	int* d_featureSampleNumbers; //Used for
	float* d_featureWeight;        //dropout
	
	int epoch;

	typedef vector<float> vector_float;
	vector<vector_float>	backup_W;
	vector<vector_float>	backup_B;
	vector<double>	backup_Cost;

	vector<float>		s_W;
	vector<float>		s_B;
	double				s_Cost;

	vector<float>		best_W;
	vector<float>		best_B;
	double				best_Cost;

	ConvolutionalComputationalLayerBase
		(ConvolutionalLayer &L, int level, ConvolutionalComputationalLayerInterface &input) :
	L(L), level(level), input(input), epoch(0) {}
	virtual void initialize() {}
	virtual void copyDataToGPU() {}
	virtual void forwards() {}
	virtual void backwards(float* &d_delta) {}
	virtual void applyDerivatives(float learningRate, float momentumDecayRate, float weightDecayRate, bool bupdate) {}
	virtual void cleanUp() {}
	virtual void cleanUPforward() {}
	virtual void backup(double cost){}
	virtual void RandomWeight(float range){}
	virtual void applyBestCost(){}
	virtual void restoreWeight(int idx) {}
};

class ConvolutionalComputationalLayer : public ConvolutionalComputationalLayerBase 
{
public:
	float* d_sgemm;
	int* d_cRules;
	int* d_pRules;
	vector<int> cRules;
	vector<int> pRules;

	ConvolutionalComputationalLayer(ConvolutionalLayer &L, int level, ConvolutionalComputationalLayerInterface &input);
	~ConvolutionalComputationalLayer();
	bool nullVectorSurvivesConvolution(int item);
	void gridConvolutionRules(vector<int>& g0, vector<int>& g1, int bg0, int bg1);
	bool nullVectorSurvivesPooling(int item);
	void gridPoolingRules(vector<int>& g1, vector<int>& g2, int bg1, int bg2);
	void initialize();
	void copyDataToGPU();
	void forwards();
	void backwards(float* &d_delta);
	void applyDerivatives(float learningRate, float momentumDecayRate, float weightDecayRate, bool bupdate);
	void cleanUp();
	void cleanUPforward();


	void backup(double cost);
	void RandomWeight(float range);
	void applyBestCost();
	void restoreWeight(int idx);
};
