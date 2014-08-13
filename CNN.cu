#include "CNN.h"

const int thisManyThreads = 1024;

__global__ void dClassify(float* d_features, int* d_predictions, int batchSize, int nOut);

ComputationalCNN::ComputationalCNN(CNN &nn) : nn(nn) 
{
	CL.push_back(new ConvolutionalComputationalLayer(nn.L[0],0, input));
	for (int l=1;l<nn.L.size();l++)
		CL.push_back(new ConvolutionalComputationalLayer(nn.L[l],l,CL[l-1]->output));
}


void ComputationalCNN::buildSparseProcessingRulesOnCPU() 
{
	input.type=data->type;
	input.batchSize=data->batchSize;
	input.featureSampleNumbers=data->featureSampleNumbers;
	input.backgroundNullVectorNumbers=data->backgroundNullVectorNumbers;
	input.grids=data->grids;//Big copy. Turn input.grids into a pointer.
	for (int l=0; l<nn.L.size(); l++)
		CL[l]->initialize();
}

void ComputationalCNN::copySparseDataToGPU() 
{
	input.d_features=d_allocateArrayFromVector<float>(data->features,__FILE__,__LINE__);
	for (int l=0;l<nn.L.size();l++)
		CL[l]->copyDataToGPU();

	/*FILE *fp = fopen("in_feature.txt", "wt");
	for(int i=0; i < data->features.size(); i++) {
		fprintf(fp, "%f,", data->features[i]);
	}
	fclose(fp);*/
}

void ComputationalCNN::cleanUPFoward() 
{
	for (int l=0;l<nn.L.size();l++)
		CL[l]->cleanUPforward();
}

void ComputationalCNN::cleanUp() 
{
	cudaFree(input.d_features);
	for (int l=0;l<nn.L.size();l++)
		CL[l]->cleanUp();
}

void ComputationalCNN::forwardPropagate()
{
	for (int l=0;l<nn.L.size();l++) {
		//cout<<"Layer: "<<l<<endl;
		__cudaCheckError(__FILE__, __LINE__);
		CL[l]->forwards();
		__cudaCheckError(__FILE__, __LINE__);
	}

}


void ComputationalCNN::test() 
{
	int* d_predictions=d_allocateArray<int>(data->batchSize,__FILE__,__LINE__);
	__cudaCheckError(__FILE__, __LINE__);
	dClassify<<<1,thisManyThreads>>>
		(CL[CL.size()-1]->output.d_features, d_predictions,
		data->batchSize, nn.L[nn.L.size()-1].nOut);
	__cudaCheckError(__FILE__, __LINE__);

	vector<int>predictions(data->batchSize);
	d2hMemcopy<int>(d_predictions,&predictions[0],data->batchSize);
	safeCudaFree(d_predictions);

	//cout<<"(";
	for (int i=0;i<data->batchSize;i++) {
		data->nMistakes+=(predictions[i]!=data->labels[i]);
		//cout<<predictions[i]<<",";
	}
	//cout<<")"<<endl;
}

//   data->topGuesses.resize(data->batchSize);
//   for (int i=0;i<data->batchSize;i++) {
//     int prediction;
//     cublasIsamax(cublasHandle, nn.L[nn.L.size()-1].nOut,
//                  CL[CL.size()-1]->output.d_features+i*nn.L[nn.L.size()-1].nOut,
//                  1,&prediction);
//     prediction--; //Fortran indexing!
//     data->topGuesses[i].push_back(prediction);
//     data->nMistakes+=(prediction!=data->labels[i]);
//   }
// }


void ComputationalCNN::findTopTenGuesses() 
{
	// //   for (int batchItem=0; batchItem < data->batchSize; batchItem++) {
	// //     data->topGuesses[batchItem].resize(10,0);
	// //     for (int i=0;i<10;i++) {
	// //       float* x=&CL[CL.size()-1]->output.d_features[0]+data->batchItem*nn.L[nn.L.size()-1].nOut;
	// //       float mx=x[0];
	// //       for (int j=1;j<nn.L[nn.L.size()-1].nOut;j++) {
	// //         if (x[j]>mx) {data->topGuesses[batchItem][i]=j;mx=x[j];}}
	// //       x[data->topGuesses[batchItem][i]]-=1;
	// //     }
	// //   }
}

void ComputationalCNN::backwardPropagate() 
{
	//top layer: d Cost / d SoftmaxInput
	int* d_labels;
	float* d_delta;  //freed by the last call to backwards
	d_delta=d_allocateArrayZeroed<float>(CL[nn.L.size()-1]->output.count*nn.L[nn.L.size()-1].nOut,__FILE__,__LINE__);
	d_labels=d_allocateArrayFromVector<int>(data->labels,__FILE__,__LINE__);

	dDerivativeOfCostWRTpreSoftmaxTopLevelWeights<<<1,thisManyThreads>>>
		(data->batchSize, d_delta, CL[CL.size()-1]->output.d_features, d_labels, nn.L[nn.L.size()-1].nOut);

	//int sz = nn.L[nn.L.size()-1].nOut;
	//vector<float> W2;
	//W2.resize(sz);
	//d2hMemcopy<float>( d_delta, &(W2[0]), W2.size() );

	safeCudaFree(d_labels);
	for (int l=CL.size()-1;l>=0;l--)
		CL[l]->backwards(d_delta);
}

void ComputationalCNN::applyDerivatives() 
{
	//for (int l=0;l<CL.size();l++)
	//	CL[l]->applyDerivatives(nn.learningRate*exp(-learningRateDecayRate*nn.epoch)
	//	, nn.momentumDecayRate, nn.weightDecayRate);

	//nn.learningRate = (std::max<double>)(1e-10,(std::min<double>)(1.0, 0.01*data->nMistakes/(double)data->batchSize));
	//nn.learningRate = (std::max<double>)(1e-10,data->nMistakes/(double)data->batchSize*0.05);
	//nn.momentumDecayRate = 0.5;

	static uint32_t seq = 0;
	for (int l=0;l<CL.size();l++) {
		CL[l]->applyDerivatives(nn.learningRate, nn.momentumDecayRate, nn.weightDecayRate, (seq == l) ? true : false);
		//CL[l]->applyDerivatives(nn.learningRate, nn.momentumDecayRate, nn.weightDecayRate, true);
	}

	++seq;
	if( seq >= CL.size() ) {
		seq = 0;
	}

	

	// CL[l]->applyDerivatives(nn.learningRate/(1+nn.epoch/learningRateDecayRate)
	//                         , nn.momentumDecayRate, nn.weightDecayRate);
	//cout<<"applyDerivatives with learning Rate["<<nn.learningRate<<"]"<<endl;
}



void ComputationalCNN::backup(double cost)
{
	for (int l=0;l<nn.L.size();l++)
		CL[l]->backup(cost);
}

void ComputationalCNN::applyBestCost()
{
	for (int l=0;l<nn.L.size();l++)
		CL[l]->applyBestCost();
}

void ComputationalCNN::randomWeight(double range)
{
	int randidx = rng.randint(nn.L.size());
	CL[randidx]->RandomWeight(range);
	/*for(int i=0; i < nn.L.size(); ++i) {
		CL[i]->RandomWeight(range);
	}*/
	//cout<<"randomWeight layer"<<randidx<<endl;
}

void ComputationalCNN::restoreWeight(int idx)
{
	for (int l=0;l<nn.L.size();l++)
		CL[l]->restoreWeight(idx);
}