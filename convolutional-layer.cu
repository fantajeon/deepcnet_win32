#include "convolutional-layer.h"


const int thisManyThreads = 1024;

//enum sigmoidType             {NOSIGMOID,   LOGISTIC,   RECTIFIEDLINEAR,   TANH,   SOFTMAX,   BLOCKY };
//const char *sigmoidNames[] ={"NOSIGMOID", "LOGISTIC", "RECTIFIEDLINEAR", "TANH", "SOFTMAX", "BLOCKY"};
//   _____ _    _ _____            ______ ______    _____
//  / ____| |  | |  __ \   /\     |  ____/ /  _ \  |  __ \
// | |    | |  | | |  | | /  \    | |__ / /| |_) | | |__) | __ ___  _ __
// | |    | |  | | |  | |/ /\ \   |  __/ / |  _ <  |  ___/ '__/ _ \| '_ \
// | |____| |__| | |__| / ____ \  | | / /  | |_) | | |   | | | (_) | |_) |
//  \_____|\____/|_____/_/    \_\ |_|/_/   |____/  |_|   |_|  \___/| .__/
//                                                                 | |
//                                                                 |_|

//kMaxout==1 when applying a nonlinear function (applied after maxout anyway)
__global__ void dSigmoidLogistic(float* g, int count, int nOut) {
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int k=0;k<nOut;k++) {
			g[i*nOut+k]=1.0f/(1.0f+expf(-g[i*nOut+k]));
		}
	}
}
__global__ void dSigmoidRectifiedLinear(float* g, int count, int nOut) {
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int k=0;k<nOut;k++) {
			g[i*nOut+k] = (g[i*nOut+k]>0.0) ? g[i*nOut+k] : 0.0f;
		}
	}
}
__global__ void dSigmoidBlocky(float* g, int count, int nOut) {
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int k=0;k<nOut;k++) {
			g[i*nOut+k]=
				(g[i*nOut+k]>1 )?
				1 :
			(( g[i*nOut+k]< -1 )?
				-1 :
			g[i*nOut+k] );
		}
	}
}
__global__ void dSigmoidTanh(float* g, int count, int nOut) {
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int k=0;k<nOut;k++) {
			g[i*nOut+k]=tanhf(g[i*nOut+k]);
		}
	}
}
__global__ void dSigmoidSoftmax(float* g, int count, int nOut) 
{
	float acc=0.0f;
	float mx=0.0f;
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		mx = g[i*nOut];
		for (int k=1;k<nOut;k++) {
			if (g[i*nOut+k]>mx) {
				mx=g[i*nOut+k];
			}
		}

		acc=0.0f;
		for (int k=0;k<nOut;k++) {
			g[i*nOut+k] = expf(g[i*nOut+k]-mx);
			acc += g[i*nOut+k];
		}

		for (int k=0;k<nOut;k++) {
			g[i*nOut+k] = g[i*nOut+k]/acc;
		}
	}
}


//SOFTMAX only occurs at the top layer;
//derivative contained in calculation of initial d_delta.
__global__ void dSigmoidBackpropLogistic(float* d, float* g, int count, int N) 
{
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int j=0; j<N; j++) {
			d[i*N+j]*=g[i*N+j]*(1.0f-g[i*N+j]);
		}
	}
}
__global__ void dSigmoidBackpropRectifiedLinear(float* d, float* g, int count, int N) 
{
	//for(int i=0; i<count; i++) {
	//	for (int j=threadIdx.x; j<N; j+=thisManyThreads)
	//		d[i*N+j]*=((g[i*N+j]>0)?1:0);
	//}
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int j=0; j<N; j++) {
			d[i*N+j]*=((g[i*N+j]>0)?1.0f:0.0f);
		}
	}
}
__global__ void dSigmoidBackpropBlocky(float* d, float* g, int count, int N) 
{
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int j=0; j<N; j++) {
			d[i*N+j]*=((g[i*N+j]>-1 && g[i*N+j] <1)?1:0);
		}
	}
}
__global__ void dSigmoidBackpropTanh(float* d, float* g, int count, int N) 
{
	for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
		for (int j=0; j<N; j++) {
			d[i*N+j]*=(1.0f+g[i*N+j])*(1.0f-g[i*N+j]);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dColumnSum(float* matrix, float* target, int nRows, int nColumns) 
{
	for (int col=threadIdx.x;col<nColumns;col+=thisManyThreads) {
		//target[col] = 0.0f;
		for (int row=0; row<nRows; row++)
			target[col]+=matrix[row*nColumns+col];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate softmaxProbability(i) - indicator(i=label)
// for i=0,1,...N-1 with N the number of character classes.
__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
	(int batchSize, float* topDelta, float* topGrid,
	int* labels, int N) 
{
	//for (int k=0;k<batchSize;k++) {
	//	for(int i=threadIdx.x;i<N;i+=thisManyThreads)
	//		topDelta[k*N+i]+=topGrid[k*N+i];


	//	if (threadIdx.x==0)
	//	topDelta[k*N+labels[k]]-=1;
	//}

	for (int k=threadIdx.x;k<batchSize;k+=thisManyThreads) {
		int label_id = labels[k];
		for(int i=0;i<N;i++)
			topDelta[k*N+i]+=topGrid[k*N+i];
		topDelta[k*N+label_id]-=1.0f;
	}
}

__device__ float sign(float v)
{
	return (v==0.0f) ? 0.0f :
		(v>0.0f) ? 1.0f : -1.0f;
}

__global__ void dConstraintWeight(float *d_kernel, float *d_bias, int fs2, int kMax, int nIn, int nOut)
{
	int stepsize = nOut*kMax;
	for(int i=0; i < nOut*kMax; i+=thisManyThreads) {
		float normfactor;
		float sum = 0.0f;
		for(int j=0; j < fs2*nIn; ++j) {
			/*if( fabs(d_kernel[i*stepsize + j]) < 1e-15f ) {
				d_kernel[i*stepsize + j] = 0.0f;
			}*/
			sum += d_kernel[i + j*stepsize]*d_kernel[i + j*stepsize];
		}
		/*if( fabs(d_bias[i]) < 1e-15f ) {
			d_bias[i] = 0.0f;
		}*/


		sum = sqrt(sum + d_bias[i]*d_bias[i]);

		//normfactor = sqrtf(2.0f)/(1e-7f+sum);
		//normfactor = ((sum>sqrtf(2.0f)) ? sqrtf(2.0f) : sum)/(1e-7f+sum);
		normfactor = sum/(1e-6f+sum);
		for(int j=0; j < fs2*nIn; ++j) {
			d_kernel[i + j*stepsize] *= normfactor;			
		}

		d_bias[i] *= normfactor;
	}
}

__global__ void dConstraintLearningRate(float *d_kernel, float *d_bias, int fs2, int kMax, int nIn, int nOut)
{
	int stepsize = nOut*kMax;
	for(int i=0; i < nOut*kMax; i+=thisManyThreads) {
		float normfactor;
		float sum = 0.0f;
		for(int j=0; j < fs2*nIn; ++j) {
			sum += d_kernel[i + j*stepsize]*d_kernel[i + j*stepsize];
		}
		/*if( fabs(d_bias[i]) < 1e-15f ) {
			d_bias[i] = 0.0f;
		}*/


		sum = sqrt(sum + d_bias[i]*d_bias[i]);

		//normfactor = sqrtf(2.0f)/(1e-7f+sum);
		//normfactor = ((sum>sqrtf(2.0f)) ? sqrtf(2.0f) : sum)/(1e-7f+sum);
		normfactor = sum/(1e-3f+sum);
		for(int j=0; j < fs2*nIn; ++j) {
			d_kernel[i + j*stepsize] *= normfactor;			
		}

		d_bias[i] *= normfactor;
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dGradientDescent2 //momentum
	(float* d_delta, float *d_delta1, float* d_momentum, 
	float* d_weights, 
	float* d_dw,
	float *d_learningRate,
	float *d_weightDecayRate, 
	int N,
	float learningRate, float momentumDecayRate, float weightDecayRate, int niter) 
{
	float np = 1.2f;
	float nm = 0.5f;
	float mt = 0.9f;
	//int subiter = niter%50;
	int subiter = niter;
	for(int i = threadIdx.x; i<N;i+=thisManyThreads) {
		float wt;
		float lr;
		if(niter==0) {
			wt = d_delta[i];
			d_dw[i] = wt;
			d_learningRate[i] = fabsf(d_delta[i])*learningRate;
			//d_weightDecayRate[i] = 0.5f;
		}
		else {
			//if( d_delta[i]*d_delta1[i] > 0 ) {
			//	// 같은 방향
			//	if( fabsf(d_delta[i]) < fabsf(d_delta1[i]) ) {
			//		d_learningRate[i] = max(d_learningRate[i]*nm,1e-14f);
			//		//d_weightDecayRate[i] = min(d_weightDecayRate[i]*np,1.0f);
			//	}
			//	d_dw[i] = d_delta[i];
			//	wt = d_dw[i];
			//}
			//else if( d_delta[i]*d_delta1[i] < 0 ) {
			//	// backtracking
			//	//d_learningRate[i] = min(d_learningRate[i]*np,1.0f);
			//	//d_weightDecayRate[i] = max(d_weightDecayRate[i]*nm,0.5f);
			//	wt = -d_dw[i];
			//	d_delta[i] = 0;
			//}
			//else {
			//	// 다시 초심으로 돌아간다.
			//	d_learningRate[i] = fabsf(d_delta[i])*1e-3;
			//	d_dw[i] = d_delta[i];
			//	wt = d_dw[i];
			//}

			wt = d_delta[i];
			d_learningRate[i] = min(1.0f,mt*d_learningRate[i]+(1.0f-mt)*fabsf(d_delta[i])*learningRate);
		}


		d_delta1[i] = d_delta[i];
		lr = d_learningRate[i];

		float k=0.01;
		momentumDecayRate = 0.99f;

		if(subiter == 0) {
			d_momentum[i]=wt;
		}
		else if(subiter < 5 ) {
			d_momentum[i]+=wt;
			if(subiter==4) {
				d_momentum[i] /= 6.0f;
			}
		}
		else {
			d_momentum[i]=d_momentum[i]*(1.0f-momentumDecayRate)+wt*momentumDecayRate;
		}

		if( subiter > 5 ) {
			d_weights[i] -= lr*(d_momentum[i]);
		}


		d_dw[i] = d_weights[i];

		//d_weights[i]-= wt;
		//d_delta[i]=0;
	}
}

__global__ void dBacktrackingGradientDescent2 //momentum
	(float* d_delta, float *d_delta1, float* d_momentum,
	float* d_weights,
	float* d_dw,
	float *d_learningRate,
	int N,
	float learningRate, float momentumDecayRate, float weightDecayRate, int niter)
{
	float np = 1.2f;
	float nm = 0.5f;
	for(int i = threadIdx.x; i<N;i+=thisManyThreads) {
		float wt;
		float lr;
		if(niter==0) {
			wt = d_delta[i];
			d_dw[i] = wt;
			d_learningRate[i] = fabsf(d_delta[i])*1e-5;
			d_delta1[i] = 0;
		}
		else {
			if( d_delta1[i]*d_delta[i] < 0 ) {
				// 복구를 한다.
				d_weights[i] = d_dw[i];
				//d_momentum[i] = (d_momentum[i]-momentumDecayRate*d_delta1[i]) + momentumDecayRate*d_delta[i];
				d_momentum[i] = d_delta[i];
				d_weights[i] -= d_learningRate[i]*d_momentum[i];
				//d_learningRate[i] = fabsf(d_delta[i])*1e-5;
				d_delta1[i] = 0.0f;
			}
		}

		//d_delta1[i] = d_delta[i];
	}
}




//__global__ void dGradientDescent2 //momentum
//	(float* d_delta, float *d_delta1, float* d_momentum, 
//	float* d_weights, 
//	float* d_dw,
//	float *d_learningRate, int N,
//	float learningRate, float momentumDecayRate, float weightDecayRate, int niter) 
//{
//	float np = 1.2f;
//	float nm = 0.5f;
//	for(int i = threadIdx.x; i<N;i+=thisManyThreads) {
//		float wt;
//		float lr;
//		float mt;
//		if(niter==0) {
//			wt = sign(d_delta[i])*learningRate;
//			mt = sign(d_delta[i]);
//			d_dw[i] = wt;
//			d_learningRate[i] = learningRate;
//		}
//		else {
//			if( d_delta[i]*d_delta1[i] > 0 ) {
//				if( fabsf(d_delta[i]) < fabsf(d_delta1[i]) ) {
//					d_learningRate[i] = max(d_learningRate[i]*nm,1e-13f);
//				}
//				
//				d_dw[i] = sign(d_delta[i])*d_learningRate[i];
//				wt = d_dw[i];
//				mt = sign(d_delta[i]);
//			}
//			else if( d_delta[i]*d_delta1[i] < 0 ) {
//				// backtracking
//				d_learningRate[i] = min(d_learningRate[i]*np,1.0f);
//				wt = -d_dw[i];
//				mt = -sign(d_delta1[i]);
//				d_delta[i] = 0;
//			}
//			else {
//				d_dw[i] = sign(d_delta[i])*d_learningRate[i];
//				wt = d_dw[i];
//				mt = sign(d_delta[i]);
//			}
//		}
//
//		d_delta1[i] = d_delta[i];
//		lr = d_learningRate[i];
//
//		float k=0.01;
//		momentumDecayRate = 0.95;
//		
//		d_weights[i] -= k*d_momentum[i] + (1.0f-k)*wt;
//
//		d_momentum[i]=d_momentum[i]*(1.0f-momentumDecayRate)+wt*momentumDecayRate;
//		//d_weights[i]-= wt;
//		//d_delta[i]=0;
//	}
//}

__global__ void dGradientDescent //momentum
	(float* d_delta, float* d_momentum, float* d_weights,  int N,
	float learningRate, float momentumDecayRate, float weightDecayRate) 
{
	for(int i = threadIdx.x; i<N;i+=thisManyThreads) {
		d_momentum[i]=d_momentum[i]*(1.0f-momentumDecayRate)+d_delta[i]*momentumDecayRate;
		d_weights[i]-=learningRate*(d_momentum[i]+d_weights[i]*weightDecayRate);
		//d_delta[i]=0;
	}
}



//////////////////////////////////////////////////////////////////////
// implementation
__host__ void ConvolutionalLayer::constraintWeight()
{
	dConstraintWeight<<<1,thisManyThreads>>>(d_W, d_B, fs2, kMaxout, nIn, nOut);
}

__host__ void ConvolutionalLayer::applyDerivatives(float* d_deltaW, float* d_deltaB, float learningRate, float momentumDecayRate, float weightDecayRate) 
{
	//dGradientDescent<<<1,thisManyThreads>>>(d_deltaW, d_momentumW, d_W, W.size(), learningRate*learningRateScale, momentumDecayRate, weightDecayRate);
	//dGradientDescent<<<1,thisManyThreads>>>(d_deltaB, d_momentumB, d_B, B.size(), learningRate*learningRateScale, momentumDecayRate, weightDecayRate);

	/*dGradientDescent<<<1,thisManyThreads>>>(d_deltaW, d_momentumW, d_W, W.size(), learningRate, momentumDecayRate, weightDecayRate);
	dGradientDescent<<<1,thisManyThreads>>>(d_deltaB, d_momentumB, d_B, B.size(), learningRate, momentumDecayRate, weightDecayRate);*/

	dGradientDescent2<<<1,thisManyThreads>>>(d_deltaW, d_W1, d_momentumW, d_W, d_dW1, d_lrW, d_weightDecayRateW, W.size(), max_scale*learningRate, momentumDecayRate, weightDecayRate, nloop);
	dGradientDescent2<<<1,thisManyThreads>>>(d_deltaB, d_B1, d_momentumB, d_B, d_dB1, d_lrB, d_weightDecayRateB, B.size(), max_scale*learningRate, momentumDecayRate, weightDecayRate, nloop);

	dConstraintLearningRate<<<1,thisManyThreads>>>(d_lrW,d_lrB, fs2, kMaxout, nIn, nOut);
	//dConstraintWeight<<<1,thisManyThreads>>>(d_deltaW, d_deltaB, fs2, kMaxout, nIn, nOut);
	d_zeroArray<float>(d_deltaB, B.size());
	d_zeroArray<float>(d_deltaW, W.size());

	++nloop;
}

__host__ void ConvolutionalLayer::checkapplyDerivatives(float* d_deltaW, float* d_deltaB, float learningRate, float momentumDecayRate, float weightDecayRate) 
{
	//dGradientDescent<<<1,thisManyThreads>>>(d_deltaW, d_momentumW, d_W, W.size(), learningRate*learningRateScale, momentumDecayRate, weightDecayRate);
	//dGradientDescent<<<1,thisManyThreads>>>(d_deltaB, d_momentumB, d_B, B.size(), learningRate*learningRateScale, momentumDecayRate, weightDecayRate);

	/*dGradientDescent<<<1,thisManyThreads>>>(d_deltaW, d_momentumW, d_W, W.size(), learningRate, momentumDecayRate, weightDecayRate);
	dGradientDescent<<<1,thisManyThreads>>>(d_deltaB, d_momentumB, d_B, B.size(), learningRate, momentumDecayRate, weightDecayRate);*/

	dBacktrackingGradientDescent2<<<1,thisManyThreads>>>(d_deltaW, d_W1, d_momentumW, d_W, d_dW1, d_lrW, W.size(), learningRate, momentumDecayRate, weightDecayRate, nloop);
	dBacktrackingGradientDescent2<<<1,thisManyThreads>>>(d_deltaB, d_B1, d_momentumB, d_B, d_dB1, d_lrB, B.size(), learningRate, momentumDecayRate, weightDecayRate, nloop);

	dConstraintLearningRate<<<1,thisManyThreads>>>(d_lrW,d_lrB, fs2, kMaxout, nIn, nOut);
	dConstraintWeight<<<1,thisManyThreads>>>(d_deltaW, d_deltaB, fs2, kMaxout, nIn, nOut);
	//d_zeroArray<float>(d_deltaB, B.size());
	//d_zeroArray<float>(d_deltaW, W.size());

	++nloop;
}

__host__ void ConvolutionalLayer::loadWeightsFromStream(std::ifstream &f) 
{
	W.resize(filterSize*filterSize*nIn*nOut*kMaxout);
	B.resize(nOut*kMaxout);
	void *pA = &W[0];
	std::streamsize szA = sizeof(float)*W.size();
	f.read( static_cast<char*>(pA), szA) ;
	f.read( (char*)&B[0], sizeof(float)*B.size() );
	h2dMemcopy<float>(&W[0],d_W,W.size());
	h2dMemcopy<float>(&B[0],d_B,B.size());
}

__host__ void ConvolutionalLayer::putWeightsToStream(std::ofstream &f)  
{
	d2hMemcopy<float>(d_W,&W[0],W.size());
	d2hMemcopy<float>(d_B,&B[0],B.size());
	f.write((char*)&W[0],sizeof(float)*W.size());
	f.write((char*)&B[0],sizeof(float)*B.size()); 
}

__host__ ConvolutionalLayer::ConvolutionalLayer(int width, int height, int fs, int ps, int s0, int s1, int s2, int in, int out, float learningRateScale, sigmoidType sig, float dropoutProbability, int kMaxout)
	: width(width),
	height(height),
	filterSize(fs), 
	poolSize(ps), 
	s0(s0), s1(s1), s2(s2), 
	nIn(in), nOut(out), sigmoid(sig), 
	dropoutProbability(dropoutProbability), 
	learningRateScale(learningRateScale),
	kMaxout(kMaxout),
	nloop(0)
{
	cout<<"ConvolutionalLayer()"<<endl;
	RNG rng;
	fs2=filterSize*filterSize;
	ps2=poolSize*poolSize;
	float fanIn=nIn*fs2;
	float fanOut=nOut*fs2*1.0f/ps2;
	float scale=pow(6.0f/(fanIn+fanOut),0.5f);
	//float scale=pow(1.0f/(fanIn+fanOut),0.5f);
	//float scale = 0.005;

	max_scale = scale;

	B.resize(nOut*kMaxout,0);
	d_B=d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_momentumB=d_allocateArrayZeroed<float>(B.size(),__FILE__,__LINE__);

	W.resize(filterSize*filterSize*nIn*nOut*kMaxout);
	double acc = 0.0;
	for (int i=0; i<W.size(); i++) {
		W[i]=rng.uniform(-scale,scale);
		/*if( W[i] > acc ) {
		acc = W[i];
		}*/
	}

	/*for (int i=0; i<W.size(); i++) {
	W[i] /= acc;
	W[i] *= 0.5f;
	}*/
	//W[i]=rng.uniform(-1,1);

	/*char sztmp[2048];
	sprintf(sztmp, "L%d-weight.txt");
	FILE *fp = fopen(sztmp, "wt");
	for (int i=0; i<W.size(); i++)
	fprintf(fp, "%f,", W[i]);
	fclose(fp);*/
	d_W=d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_momentumW=d_allocateArrayZeroed<float>(W.size(),__FILE__,__LINE__);

	d_lrW = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_lrB = d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_W1 = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_B1 =d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_dW1 = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_dB1 = d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_weightDecayRateW = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_weightDecayRateB = d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);


	cout<<"ConvolutionalLayer(), scale:"<<max_scale<<endl;
}

__host__ ConvolutionalLayer::ConvolutionalLayer
	(int fs, int ps, int s0, int s1, int s2, int in, int out, float learningRateScale, sigmoidType sig, float dropoutProbability, int kMaxout)
	: width(0),
	height(0),
	filterSize(fs), 
	poolSize(ps), 
	s0(s0), s1(s1), s2(s2), 
	nIn(in), nOut(out), sigmoid(sig), 
	dropoutProbability(dropoutProbability), 
	learningRateScale(learningRateScale),
	kMaxout(kMaxout) ,
	nloop(0)
{

	cout<<"ConvolutionalLayer()"<<endl;
	RNG rng;
	fs2=filterSize*filterSize;
	ps2=poolSize*poolSize;
	float fanIn=nIn*fs2;
	float fanOut=nOut*fs2*1.0f/ps2;
	float scale=pow(6.0f/(fanIn+fanOut),0.5f);
	//float scale=pow(1.0f/(fanIn+fanOut),0.5f);
	//float scale = 0.005;

	max_scale = scale;

	B.resize(nOut*kMaxout,0);
	d_B=d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_momentumB=d_allocateArrayZeroed<float>(B.size(),__FILE__,__LINE__);

	W.resize(filterSize*filterSize*nIn*nOut*kMaxout);
	double acc = 0.0;
	for (int i=0; i<W.size(); i++) {
		W[i]=rng.uniform(-scale,scale);
		/*if( W[i] > acc ) {
		acc = W[i];
		}*/
	}

	/*for (int i=0; i<W.size(); i++) {
	W[i] /= acc;
	W[i] *= 0.5f;
	}*/
	//W[i]=rng.uniform(-1,1);

	/*char sztmp[2048];
	sprintf(sztmp, "L%d-weight.txt");
	FILE *fp = fopen(sztmp, "wt");
	for (int i=0; i<W.size(); i++)
	fprintf(fp, "%f,", W[i]);
	fclose(fp);*/
	d_W=d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_momentumW=d_allocateArrayZeroed<float>(W.size(),__FILE__,__LINE__);

	d_lrW = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_lrB = d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_W1 = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_B1 =d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_dW1 = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_dB1 = d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
	d_weightDecayRateW = d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
	d_weightDecayRateB = d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);


	cout<<"ConvolutionalLayer(), scale:"<<max_scale<<endl;
}

