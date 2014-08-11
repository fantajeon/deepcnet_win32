//#include "nn-computational.cuh"
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
//#include <cula.h>

#include "nn-computational.h"


cublasHandle_t cublasHandle = NULL;
cudaStream_t cublasstream = NULL;
const int thisManyThreads = 1024;


double temperature(int k, int kmax)
{
	return (double)(kmax-k)/(double)kmax;
}

double prob(double e, double e1, double T)
{
	if( e > e1 )
		return 1.0;
	else
		return exp(-(e1-e)/T);
}


template <typename t> __global__ void dReplicateArray(t* d_src, t* d_dest, int size, int nCopies) 
{
	for (int i=threadIdx.x;i<nCopies;i+=thisManyThreads) {
		for (int j=0;j<size;j++) {
			d_dest[i*size+j]=d_src[j];
		}
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




///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dPropForwardToMatrixMultiplyInput(float* d_features, float* d_sgemm, int* rules, int count, int nIn, int fs2) 
{
	for (int i = threadIdx.x;i<count*fs2;i+=thisManyThreads) {
		for (int k=0;k<nIn;k++) {
			d_sgemm[i*nIn+k]=d_features[rules[i]*nIn+k]; 
		}
	}
}

__global__ void dPropBackwardFromMatrixMultiplyOutput(float* d_deltaGrid, float* d_sgemm, int* rules, int count, int nIn, int fs2) 
{
	for (int i = threadIdx.x;i<count*fs2;i+=thisManyThreads) {
		for (int k=0;k<nIn;k++) {
			atomicAdd(d_deltaGrid+rules[i]*nIn+k,d_sgemm[i*nIn+k]); 
		}
	}
}


__global__ void dDeltaRescaling(float *d_delta, int count, int nIn, float scaling)
{
	const float s = powf(sqrt(2.0f),2.0f);
	for(int i=0; i < count; i+=thisManyThreads) {
		float mx = 0.0f;
		float sum = 0.0f;
		for(int k=0; k < nIn; ++k) {
			float v;
			//d_delta[i*nIn + k] *= s;
			v = fabs(d_delta[i*nIn + k]);
			if( v < 1e-15f ) {
				d_delta[i*nIn + k] = 0.0f;
				v = 0.0f;
			}
			
			sum += v;
			if(mx < v ) {
				mx = v;
			}
		}
		
		if( mx > 1e-15f ) {
			float acc=0.0f;
			for(int k=0; k < nIn; ++k) {
				acc += expf(-powf(d_delta[i*nIn + k]-mx,2.0f)/s);
			}

			//acc /= sqrt(2.0f);
			for(int k=0; k < nIn; ++k) {
				if( d_delta[i*nIn + k] >= 0.0f ) {
					d_delta[i*nIn + k] = expf(-powf(d_delta[i*nIn + k]-mx,2.0f)/s)/acc;
				}
				else {
					d_delta[i*nIn + k] = -expf(-powf(-d_delta[i*nIn + k]-mx,2.0f)/s)/acc;
				}
			}
		}
	}
}

__global__ void dDropoutFeatures(float* d_features, int* d_featureSampleNumbers,
								 int count, int nIn, float* d_featureWeight) 
{
	for (int i=threadIdx.x; i<count*nIn; i+=thisManyThreads) {
		int item=d_featureSampleNumbers[i/nIn];
		d_features[i]*=d_featureWeight[item*nIn+(i%nIn)];
	}
}

__global__ void dClassify(float* d_features, int* d_predictions, int batchSize, int nOut) 
{
	int i = threadIdx.x;
	for (/*int i = threadIdx.x*/; i<batchSize; i+=thisManyThreads) {
		int prediction=0;
		float maxP=d_features[i*nOut];
		for (int k=1;k<nOut;k++) {
			if (d_features[i*nOut+k]>maxP) {
				prediction=k;
				maxP=d_features[i*nOut+k];
			}
		}
		d_predictions[i]=prediction;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////
//count is a multiple of nOut
__global__ void dMaxout(float* g1a, float* g1b, int count, int kMaxout, unsigned char* d_choice) 
{
	for (int i=threadIdx.x; i<count; i+=thisManyThreads) {
		g1b[i]=g1a[i*kMaxout];
		d_choice[i]=0;
		for (int j=1;j<kMaxout;j++) {
			if (g1b[i]<g1a[i*kMaxout+j]) {
				g1b[i]=g1a[i*kMaxout+j];
				d_choice[i]=j;
			}
		}
	}
}

__global__ void dMaxoutBackprop(float* d1a, float* d1b, int count, int kMaxout, unsigned char* d_choice) 
{
	for (int i=threadIdx.x; i<count; i+=thisManyThreads) {
		d1a[i*kMaxout+d_choice[i]]=d1b[i];
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dMaxPool(float* g1, float* g2, int* rules, int count, int ps2, int nOut, unsigned char* d_choice) 
{
	for (int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int j = 0;j<nOut;j++) {
			float max = g1[rules[i*ps2+0]*nOut+j];
			d_choice[i*nOut+j]=0;

			for (int k=1; k<ps2; k++) {
				if (max<g1[rules[i*ps2+k]*nOut+j]) {
					max=g1[rules[i*ps2+k]*nOut+j];
					d_choice[i*nOut+j]=k;
				}
			}
			g2[i*nOut+j] = max;
		}
	}
}

__global__ void dMaxPoolBackprop
	(int* rules, float* d1, float* d2, int count, int ps2, int nOut, unsigned char* d_choice) 
{
	for (int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int j=0; j<nOut; j++) {
			d1[rules[i*ps2+d_choice[i*nOut+j]]*nOut+j]=d2[i*nOut+j];
		}
	}
}

ConvolutionalComputationalLayer::ConvolutionalComputationalLayer(ConvolutionalLayer &L, int level, ConvolutionalComputationalLayerInterface &input) :
	ConvolutionalComputationalLayerBase(L,level,input) 
{
	d_featureSampleNumbers = NULL;
	d_featuresToMaxout = NULL;
	d_maxPoolChoice = NULL;
	d_maxoutChoice = NULL;
	d_featureWeight = NULL;

	d_deltaB=d_allocateArrayZeroed<float>(L.B.size(),__FILE__,__LINE__);
	d_deltaW=d_allocateArrayZeroed<float>(L.W.size(),__FILE__,__LINE__);

	s_W = L.W;
	s_B = L.B;
	s_Cost = 1.0;

	this->best_W = s_W;
	this->best_B = s_B;
	this->best_Cost = s_Cost;

	cout<<"ConvolutionalComputationalLayer()"<<endl;
}

ConvolutionalComputationalLayer::~ConvolutionalComputationalLayer() 
{
	safeCudaFree(d_deltaW);
	safeCudaFree(d_deltaB);

	cout<<"~ConvolutionalComputationalLayer()"<<endl;
}


bool ConvolutionalComputationalLayer::nullVectorSurvivesConvolution(int item) {
	for (int i=0; i<L.s1;i++) {
		for (int j=0; j<L.s1;j++) {
			int ctr=0;
			for (int ii=0;ii<L.filterSize;ii++) {
				for (int jj=0;jj<L.filterSize;jj++) {
					int n0=(i+ii)*L.s0+(j+jj);
					if (input.grids[item][n0]==input.backgroundNullVectorNumbers[item])
						ctr++;
				}
			}
			if (ctr==L.fs2) return true;
		}
	}
	return false;
}

void ConvolutionalComputationalLayer::gridConvolutionRules(vector<int>& g0, vector<int>& g1, int bg0, int bg1) {
	if (bg1 != -1)
		for (int i=0; i<L.fs2; i++)
			cRules.push_back(bg0);
	g1.resize(L.s1*L.s1,bg1);
	for (int i=0;i<L.s1;i++) {
		for (int j=0;j<L.s1;j++) {
			int n1=i*L.s1+j;
			for (int ii=0;ii<L.filterSize;ii++) {
				for (int jj=0;jj<L.filterSize;jj++) {
					int n0=(i+ii)*L.s0+(j+jj);
					if (g0[n0]!=bg0 && g1[n1]==bg1)
						g1[n1]=middle.count++;
				}
			}
			if (g1[n1]!=bg1) {
				for (int ii=0;ii<L.filterSize;ii++) {
					for (int jj=0;jj<L.filterSize;jj++) {
						int n0=(i+ii)*L.s0+(j+jj);
						cRules.push_back(g0[n0]);
					}
				}
			}
		}
	}
}
bool ConvolutionalComputationalLayer::nullVectorSurvivesPooling(int item) {
	for (int i=0; i<L.s2;i++) {
		for (int j=0; j<L.s2;j++) {
			int ctr=0;
			for (int ii=0;ii<L.poolSize;ii++) {
				for (int jj=0;jj<L.poolSize;jj++) {
					int n1=(i*L.poolSize+ii)*L.s1+(j*L.poolSize+jj);
					//            cout << level << " " << n1 << " " << middle.grids[item][n1] << " " << middle.backgroundNullVectorNumbers[item]<< " " << ctr << endl;
					if (middle.grids[item][n1]==middle.backgroundNullVectorNumbers[item])
						ctr++;
				}
			}
			if (ctr==L.ps2) return true;
		}
	}
	return false;
}
void ConvolutionalComputationalLayer::gridPoolingRules(vector<int>& g1, vector<int>& g2, int bg1, int bg2) {
	if (bg2 != -1)
		for (int i=0; i<L.ps2; i++)
			pRules.push_back(bg1);
	g2.resize(L.s2*L.s2,bg2);
	for (int i=0;i<L.s2;i++) {
		for (int j=0;j<L.s2;j++) {
			int n2=i*L.s2+j;
			for (int ii=0;ii<L.poolSize;ii++) {
				for (int jj=0;jj<L.poolSize;jj++) {
					int n1=(i*L.poolSize+ii)*L.s1+(j*L.poolSize+jj);
					if (g1[n1]!=bg1 && g2[n2]==bg2)
						g2[n2]=output.count++;
				}
			}
			if (g2[n2]!=bg2) {
				for (int ii=0;ii<L.poolSize;ii++) {
					for (int jj=0;jj<L.poolSize;jj++) {
						int n1=(i*L.poolSize+ii)*L.s1+(j*L.poolSize+jj);
						pRules.push_back(g1[n1]);
					}
				}
			}
		}
	}
}
void ConvolutionalComputationalLayer::initialize() 
{
	cRules.reserve(L.fs2*input.batchSize*L.s1*L.s1);
	pRules.reserve(L.ps2*input.batchSize*L.s2*L.s2);
	output.type=middle.type=input.type;
	output.batchSize=middle.batchSize=input.batchSize; //All the same
	middle.backgroundNullVectorNumbers.resize(middle.batchSize);
	output.backgroundNullVectorNumbers.resize(output.batchSize);
	middle.grids.resize(middle.batchSize);
	output.grids.resize(output.batchSize);
	for (int item=0; item<input.batchSize; item++) {
		if (nullVectorSurvivesConvolution(item))
			middle.backgroundNullVectorNumbers[item]=middle.count++;
		else
			middle.backgroundNullVectorNumbers[item]=-1;
		gridConvolutionRules(input.grids[item],
			middle.grids[item],
			input.backgroundNullVectorNumbers[item],
			middle.backgroundNullVectorNumbers[item]);
		while (middle.featureSampleNumbers.size() < middle.count)
			middle.featureSampleNumbers.push_back(item);

		if (nullVectorSurvivesPooling(item))
			output.backgroundNullVectorNumbers[item]=output.count++;
		else
			output.backgroundNullVectorNumbers[item]=-1;
		gridPoolingRules(middle.grids[item],
			output.grids[item],
			middle.backgroundNullVectorNumbers[item],
			output.backgroundNullVectorNumbers[item]);
		while (output.featureSampleNumbers.size() < output.count)
			output.featureSampleNumbers.push_back(item);
	}
	// size_t Free, Total;
	// cudaError_t result=cudaMemGetInfo(&Free, &Total);
	// cout << Free<< " " << Total<<endl;
	// cout << cRules.size()/L.fs2 << " "  << pRules.size()/L.ps2<<endl;
	// cout <<input.count <<" " << middle.count << " " << output.count <<endl;
}

void ConvolutionalComputationalLayer::copyDataToGPU() 
{
	d_cRules=d_allocateArrayFromVector<int>(cRules,__FILE__,__LINE__);
	d_pRules=d_allocateArrayFromVector<int>(pRules,__FILE__,__LINE__);
	d_sgemm=d_allocateArray<float>(middle.count*L.fs2*L.nIn,__FILE__,__LINE__);
	d_featuresToMaxout=d_allocateArray<float>(middle.count*L.nOut*L.kMaxout,__FILE__,__LINE__);
	if (L.kMaxout>1) {
		d_maxoutChoice=d_allocateArray<unsigned char>(middle.count*L.nOut,__FILE__,__LINE__);
		middle.d_features=d_allocateArray<float>(middle.count*L.nOut,__FILE__,__LINE__);
	} else
		middle.d_features=d_featuresToMaxout;
	if (L.poolSize>1) {
		d_maxPoolChoice=d_allocateArray<unsigned char>(output.count*L.nOut,__FILE__,__LINE__);
		output.d_features=d_allocateArray<float>(output.count*L.nOut,__FILE__,__LINE__);
	} else
		output.d_features=middle.d_features;

	//cudaDeviceSynchronize();

	h2dMemcopy<float>(&(L.W[0]), L.d_W, L.W.size());
	h2dMemcopy<float>(&(L.B[0]), L.d_B, L.B.size());
}


void ConvolutionalComputationalLayer::cleanUp()
{
	backup_W.clear();
	backup_B.clear();
	backup_Cost.clear();

	safeCudaFree(d_cRules);
	safeCudaFree(d_pRules);
	safeCudaFree(d_sgemm);
	cRules.clear();
	pRules.clear();
	safeCudaFree(d_featuresToMaxout);
	if (L.kMaxout>1) {
		safeCudaFree(d_maxoutChoice);
		safeCudaFree(middle.d_features);
	}
	if (L.poolSize>1) {
		safeCudaFree(d_maxPoolChoice);
		safeCudaFree(output.d_features);
	}
	middle.featureSampleNumbers.clear();
	middle.backgroundNullVectorNumbers.clear();
	middle.grids.clear();
	middle.count=0;
	output.featureSampleNumbers.clear();
	output.backgroundNullVectorNumbers.clear();
	output.grids.clear();
	output.count=0;

	if (L.dropoutProbability>0) {
		safeCudaFree(d_featureWeight); 
		safeCudaFree(d_featureSampleNumbers); 
	}

	cleanUPforward();
}

void ConvolutionalComputationalLayer::cleanUPforward()
{
	safeCudaFree(d_featureWeight);
	safeCudaFree(d_featureSampleNumbers);
}

void ConvolutionalComputationalLayer::forwards() 
{
	//Dropout
	if (L.dropoutProbability>0) {
		vector<float> featureWeights(input.batchSize*L.nIn,1.0f-L.dropoutProbability);
		if (input.type==TRAINBATCH)
			for (int i=0;i<featureWeights.size(); i++)
				featureWeights[i]=rng.bernoulli(1.0f-L.dropoutProbability);
		d_featureWeight=d_allocateArrayFromVector<float>(featureWeights,__FILE__,__LINE__);
		d_featureSampleNumbers=d_allocateArrayFromVector<int>(input.featureSampleNumbers,__FILE__,__LINE__);

		dDropoutFeatures<<<1,thisManyThreads>>>(input.d_features, d_featureSampleNumbers, input.count, L.nIn, d_featureWeight);
	}

	__cudaCheckError(__FILE__, __LINE__);
	//convolution
	dPropForwardToMatrixMultiplyInput<<<1,thisManyThreads>>>(input.d_features, d_sgemm, d_cRules, middle.count,L.nIn,L.fs2);
	//cudaDeviceSynchronize();


	__cudaCheckError(__FILE__, __LINE__);
	dReplicateArray<float><<<1,thisManyThreads>>>       //set bias
		(L.d_B, d_featuresToMaxout, L.nOut*L.kMaxout,middle.count);
	//cudaDeviceSynchronize();


	__cudaCheckError(__FILE__, __LINE__);
	d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
		d_sgemm, L.d_W, d_featuresToMaxout,
		middle.count, L.fs2*L.nIn, L.nOut*L.kMaxout,
		1.0f, 1.0f);
	//cudaDeviceSynchronize();

	//int al_cnt = middle.count*L.nOut*L.kMaxout;
	//float *pfeatureToMax = (float*)malloc(sizeof(float)*al_cnt);

	//d2hMemcopy<float>( d_featuresToMaxout, pfeatureToMax, al_cnt);
	//char sztmp[2048];
	//sprintf(sztmp, "L%d.txt", this->level);
	//FILE *fp = fopen(sztmp, "wt");
	//for(int c=0; c < al_cnt; c++) {
	//	if( (c%middle.count) == 0 ) {
	//		fprintf(fp, "\n★");
	//	}
	//	fprintf(fp, "%f,", pfeatureToMax[c]);
	//}
	//fclose(fp);
	//free(pfeatureToMax);

	//Maxout
	if (L.kMaxout>1) {
		dMaxout<<<1,thisManyThreads>>>
		(d_featuresToMaxout, middle.d_features, middle.count*L.nOut, L.kMaxout, d_maxoutChoice);
		__cudaCheckError(__FILE__, __LINE__);
	}

	//maxpooling
	if (L.poolSize>1) {
		__cudaCheckError(__FILE__, __LINE__);
		dMaxPool<<<1,thisManyThreads>>>(middle.d_features,output.d_features,d_pRules,output.count,L.ps2,L.nOut,d_maxPoolChoice);
		//cudaDeviceSynchronize();

		//int al_cnt = output.count*L.nOut;
		//float *ee_d_features = (float*)malloc(sizeof(float)*al_cnt);

		//d2hMemcopy<float>( output.d_features, ee_d_features, al_cnt);
		//char sztmp[2048];
		//sprintf(sztmp, "L%d-d_feature.txt", this->level);
		//FILE *fp = fopen(sztmp, "wt");
		//for(int c=0; c < al_cnt; c++) {
		//	if( (c%middle.count) == 0 ) {
		//		fprintf(fp, "\n★");
		//	}
		//	fprintf(fp, "%f,", ee_d_features[c]);
		//}
		//fclose(fp);
		//free(ee_d_features);
	}
	__cudaCheckError(__FILE__, __LINE__);
	switch(L.sigmoid) {
	case RECTIFIEDLINEAR: dSigmoidRectifiedLinear<<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
	case LOGISTIC:     dSigmoidLogistic    <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
	case BLOCKY:       dSigmoidBlocky      <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
	case TANH:         dSigmoidTanh        <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
	case SOFTMAX:      dSigmoidSoftmax     <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
	}
	__cudaCheckError(__FILE__, __LINE__);
	//cudaDeviceSynchronize();


}

void ConvolutionalComputationalLayer::backwards(float* &d_delta)  
{
	//  cout << "B" << L.nIn<<endl;
	//adjust d_delta post to pre sigmoid using outputGrid. Does nothing to top softmax layer.
	switch(L.sigmoid) {
	case RECTIFIEDLINEAR: dSigmoidBackpropRectifiedLinear<<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
	case LOGISTIC:     dSigmoidBackpropLogistic    <<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
	case BLOCKY:       dSigmoidBackpropBlocky      <<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
	case TANH:         dSigmoidBackpropTanh        <<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
	}

	//Undo max-pooling.
	if (L.poolSize>1) {
		float* d_delta_=d_allocateArrayZeroed<float>(middle.count*L.nOut,__FILE__,__LINE__);
		dMaxPoolBackprop<<<1,thisManyThreads>>>(d_pRules, d_delta_, d_delta, output.count, L.ps2, L.nOut, d_maxPoolChoice);
		cudaFree(d_delta);
		d_delta=d_delta_;
	}
	//Undo maxout
	if (L.kMaxout>1) {
		float* d_delta_=d_allocateArrayZeroed<float>(middle.count*L.nOut*L.kMaxout,__FILE__,__LINE__);
		dMaxoutBackprop<<<1,thisManyThreads>>>(d_delta_, d_delta, middle.count*L.nOut, L.kMaxout, d_maxoutChoice);
		cudaFree(d_delta);
		d_delta=d_delta_;
	}

	//int sz = middle.count*L.nOut*L.kMaxout;
	//vector<float> W2;
	//W2.resize(sz);
	//d2hMemcopy<float>( d_delta, &(W2[0]), W2.size() );

	//calculate d_deltaB
	dColumnSum<<<1,thisManyThreads>>>
		(d_delta,d_deltaB,middle.count,L.nOut*L.kMaxout);

	//Calculate delta_W
	d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
		d_sgemm, d_delta, d_deltaW,
		L.nIn*L.fs2, middle.count, L.nOut*L.kMaxout,
		1.0, 1.0);
	cudaThreadSynchronize();


	if (level>0) {
		//Undo convolution
		float* d_deltaSgemm;
		d_deltaSgemm=d_allocateArray<float>(middle.count*L.nIn*L.fs2,__FILE__,__LINE__);
		d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
			d_delta, L.d_W, d_deltaSgemm,
			middle.count,L.nOut*L.kMaxout,L.nIn*L.fs2,
			1.0, 0.0);
		safeCudaFree(d_delta);
		d_delta=d_allocateArrayZeroed<float>(input.count*L.nIn,__FILE__,__LINE__);
		dPropBackwardFromMatrixMultiplyOutput<<<1,thisManyThreads>>>
			(d_delta, d_deltaSgemm,  d_cRules, middle.count, L.nIn, L.fs2);

		dDeltaRescaling<<<1,thisManyThreads>>>(d_delta,input.count,L.nIn,L.learningRateScale);
		cudaFree(d_deltaSgemm);
		//Dropout
		if (L.dropoutProbability>0) {
			dDropoutFeatures<<<1,thisManyThreads>>>
				(d_delta, d_featureSampleNumbers, input.count, L.nIn, d_featureWeight);
		}
	} else {
		safeCudaFree(d_delta);
		d_delta = NULL;
	}
	cudaThreadSynchronize();
}


void ConvolutionalComputationalLayer::applyDerivatives(float learningRate, float momentumDecayRate, float weightDecayRate, bool bupdate)
{
	if(bupdate) {
		L.applyDerivatives(d_deltaW, d_deltaB, learningRate, momentumDecayRate, weightDecayRate);
	}
	else {
		L.checkapplyDerivatives(d_deltaW, d_deltaB, learningRate, momentumDecayRate, weightDecayRate);
	}

	d2hMemcopy<float>(L.d_W, &(L.W[0]), L.W.size());
	d2hMemcopy<float>(L.d_B, &(L.B[0]), L.B.size());
}


void ConvolutionalComputationalLayer::backup(double cost)
{
	//		vector<float> B;
//
//		W.resize( L.W.size() );
//		B.resize( L.B.size() );

		//d2hMemcopy<float>(L.d_W, &W[0], W.size());
		//d2hMemcopy<float>(L.d_B, &B[0], B.size());

	backup_W.push_back(L.W);
	backup_B.push_back(L.B);
	backup_Cost.push_back(cost);

	double T = temperature(epoch,1600000);
	if( prob(this->s_Cost, cost, T) > this->rng.uniform(0.0f,1.0f) ) {
		this->s_W = L.W;
		this->s_B = L.B;
		this->s_Cost = cost;
	}
}

void ConvolutionalComputationalLayer::RandomWeight(float range)
{
	/*for(int s=0; s < 10; ++s) {
		if( rng.bernoulli(0.5) ) {
			int randix = rng.randint(L.W.size());
			L.W[randix] += rng.uniform(-range,range);
		}
		else {
			int randix = rng.randint(L.B.size());
			L.B[randix] += rng.uniform(-range,range);
		}
	}*/

	float u = range;
	//float u = 0.05;
	for(int i=0; i < L.W.size(); ++i) {
		if( rng.bernoulli(0.05) ) {
			L.W[i] += rng.uniform(-u,u);
		}
	}

	for(int i=0; i < L.B.size(); ++i) {
		if( rng.bernoulli(0.05) ) {
			L.B[i] += rng.uniform(-u,u);
		}
	}

	//cout<<endl;

	h2dMemcopy<float>(&L.W[0],L.d_W,L.W.size());
	h2dMemcopy<float>(&L.B[0],L.d_B,L.B.size());
}

void ConvolutionalComputationalLayer::applyBestCost()
{
	++epoch;

	//cout<<"IN:applyBestCost() len:"<<backup_Cost.size()<<endl;
	int best_idx = 0;
	double best_cost = backup_Cost[0];
	
	for(int i=1; i < backup_Cost.size(); ++i) {
		if( best_cost > backup_Cost[i] ) {
			best_cost = backup_Cost[i];
			best_idx = i;
		}
	}

	if( this->best_Cost > best_cost ) {
		// Best solution은 유지하고 있는다.
		this->best_Cost = best_cost;
		this->best_W = backup_W[best_idx];
		this->best_B = backup_B[best_idx];
	}


	int wsize = L.W.size();
	int bsize = L.B.size();

	L.W.clear();
	L.B.clear();

	//L.W.resize(wsize,0.0f);
	//L.B.resize(bsize,0.0f);

	L.W = backup_W[0];
	L.B = backup_B[0];

	h2dMemcopy<float>(&(L.W[0]), L.d_W, L.W.size());
	h2dMemcopy<float>(&(L.B[0]), L.d_B, L.B.size());

	L.constraintWeight();

	d2hMemcopy<float>(L.d_W, &L.W[0], L.W.size());
	d2hMemcopy<float>(L.d_B, &L.B[0], L.B.size());

	backup_W.clear();
	backup_B.clear();
	backup_Cost.clear();
	//cout<<"OUT:applyBestCost()"<<endl;
}

void ConvolutionalComputationalLayer::restoreWeight(int idx)
{
	L.W.clear();
	L.B.clear();

	if( idx == -1 ) {
		L.W = s_W;
		L.B = s_B;
	}
	else {
		L.W = best_W;
		L.B = best_B;
	}

	//		char sztmp[2048];
	//
	//		sprintf(sztmp,"restoreWeight: (W:[%d],B:[%d])->(W:[%d],B:[%d])",
	//				backup_W[idx].size(), backup_B[idx].size(),
	//				L.W.size(), L.B.size());
	//		cout<<sztmp<<endl;

	h2dMemcopy<float>(&(L.W[0]), L.d_W, L.W.size());
	h2dMemcopy<float>(&(L.B[0]), L.d_B, L.B.size());
}