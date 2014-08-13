#include "CNN.h"

#include <boost/thread.hpp>



std::vector<float> *g_pregularizingConstants;
std::vector<Picture*> (*g_ptrainCharacters);
std::vector<Picture*> (*g_ptestCharacters);

void init_cnn()
{
	g_pregularizingConstants = new std::vector<float>();
	g_ptrainCharacters = new std::vector<Picture*>();
	g_ptestCharacters = new std::vector<Picture*>();
}

void deinit_cnn()
{
	if(g_pregularizingConstants) {
		delete g_pregularizingConstants;
		g_pregularizingConstants = NULL;
	}

	if(g_ptrainCharacters) {
		for(int i=0; i < g_ptrainCharacters->size(); ++i) {
			delete (*g_ptrainCharacters)[i];
		}
		delete g_ptrainCharacters;
		g_ptrainCharacters = NULL;
	}

	if(g_ptestCharacters) {
		for(int i=0; i < g_ptestCharacters->size(); ++i) {
			delete (*g_ptestCharacters)[i];
		}

		delete g_ptestCharacters;
		g_ptestCharacters = NULL;
	}
}


const char *sigmoidNames[] ={"NOSIGMOID", "LOGISTIC", "RECTIFIEDLINEAR", "TANH", "SOFTMAX", "BLOCKY"};

//void loadData(); //Application specific loading mechanism

void replaceTestSetWithValidationSet(float p) 
{
	RNG rng;
	rng.seed(0);
	while ( (*g_ptestCharacters).size() > 0) { //Delete the test set
		delete (*g_ptestCharacters).back();
		(*g_ptestCharacters).pop_back();
	}
	vector<Picture*> c((*g_ptrainCharacters)); //Split the training set into
	(*g_ptrainCharacters).clear();             //training and validation sets.
	while (c.size()>0) {
		if (rng.uniform()<p)
			(*g_ptrainCharacters).push_back(c.back());
		else
			(*g_ptestCharacters).push_back(c.back());
		c.pop_back();
	}
	cout << "Replacing test set with validation set.\n";
}

void smallerTestSet(float p) 
{
	RNG rng;
	rng.seed(0);
	vector<Picture*> c((*g_ptestCharacters));
	(*g_ptestCharacters).clear();
	while (c.size()>0) {
		if (rng.uniform()<p)
			(*g_ptestCharacters).push_back(c.back());
		else
			delete c.back();
		c.pop_back();
	}
	cout << "Reducing test set size.\n";
}

void calculateRegularizingConstants(int nInputFeatures) 
{
	cout << "Using " << (*g_ptrainCharacters).size() << " training samples to calculate regularizing constants." << endl;
	RNG rng;
	SparseCnnInterface interface(TRAINBATCH);
	//regularizingConstants.resize(nInputFeatures,1.0f);//Assume initially empty.

	g_pregularizingConstants->clear();
	for(int i=0; i < nInputFeatures; ++i) {
		g_pregularizingConstants->push_back(1.0f);
	}

	for (int i=0;i<10000;i++)
		(*g_ptrainCharacters)[rng.index((*g_ptrainCharacters))]->codifyInputData(interface);
	for (int i=0; i<nInputFeatures; i++) {
		(*g_pregularizingConstants)[i]=0;
		for (int j=0; j<interface.count; j++)
			(*g_pregularizingConstants)[i]=
				max(abs(interface.features[i+j*nInputFeatures]),
					(*g_pregularizingConstants)[i]);
	}
	cout << "Regularizing constants: ";
	for (int i=0; i<nInputFeatures; i++)
		cout << (*g_pregularizingConstants)[i] << " ";
	cout << endl;
}


/////////////////////////////////////////////////////////////////////////


static boost::mutex CNNmutex;


void ComputationalCNN::processBatch(SparseCnnInterface *d) 
{
	static int bisfirst = 1;

	data=d;
	input.count=data->count;
	__cudaCheckError(__FILE__, __LINE__);
	buildSparseProcessingRulesOnCPU();
	__cudaCheckError(__FILE__, __LINE__);
	copySparseDataToGPU();
	__cudaCheckError(__FILE__, __LINE__);

	
	if (data->type == UNLABELLEDBATCH)
		findTopTenGuesses();
	else if (data->type ==  TRAINBATCH) {
		
		data->nMistakes = 0;

		forwardPropagate();
		test();
		backwardPropagate();
		cleanUPFoward();

		boost::mutex::scoped_lock lock(CNNmutex);
		if(bisfirst==0) {
			//if( nn.epoch%1 == 0 ) {
			//	applyDerivatives();
			//	//cout<<"applyDerivatives"<<endl;
			//}
			if (nn.epoch%500==0)
				nn.saveWeights();
		}
		else
			bisfirst = 0;
		++nn.epoch;
	}
	else {
		forwardPropagate();
		test();
		cleanUPFoward();
	}
	cleanUp();
}


void ComputationalCNN::processRandomBatch(SparseCnnInterface *d) 
{
	static int bisfirst = 1;

	data=d;
	input.count=data->count;
	__cudaCheckError(__FILE__, __LINE__);
	buildSparseProcessingRulesOnCPU();
	__cudaCheckError(__FILE__, __LINE__);
	copySparseDataToGPU();
	__cudaCheckError(__FILE__, __LINE__);

	
	if (data->type == UNLABELLEDBATCH)
		findTopTenGuesses();
	else if (data->type ==  TRAINBATCH) {
		double best_cost = 1.0;
		int best_nMistakes = 0;
		int init_nMistakes = 0;

		double cost = 0;
		double init_cost = 0;
		data->nMistakes = 0;

		// 1차 현 상황 테스트
		restoreWeight(1);		// BEST 모드로 한다.
		forwardPropagate();
		__cudaCheckError(__FILE__, __LINE__);
		
		test();
		cost = data->nMistakes/(double)(data->batchSize);
		backup(cost);
		cleanUPFoward();


		init_cost = cost;
		init_nMistakes = data->nMistakes;
		best_cost = cost;
		best_nMistakes = data->nMistakes;

		//cout<<"cost:"<<cost;

		for(int trial=0; trial < 50; ++trial) {
			data->nMistakes = 0;
			restoreWeight(-1);		// SA 처음 상태로 복원
			randomWeight(nn.learningRate);
			forwardPropagate();
			test();
			cleanUPFoward();

			cost = data->nMistakes/(double)(data->batchSize);

			cout.precision(8);
			//cout<<":"<<cost<<flush;
			backup(cost);

			if(best_cost > cost) {
				best_cost = cost;
				best_nMistakes = data->nMistakes;
			}
		}
		//cout<<"Best:"<<best_cost<<endl;
		applyBestCost();	// backwardPropgate()을 대체

		data->nMistakes = 0;
		restoreWeight(1);		// best solution으로 회귀
		forwardPropagate();
		test();

		boost::mutex::scoped_lock lock(CNNmutex);
		if(bisfirst==0) {
			//if( nn.epoch%1 == 0 ) {
			//	applyDerivatives();
			//	//cout<<"applyDerivatives"<<endl;
			//}
			if (nn.epoch%500==0)
				nn.saveWeights();
		}
		else
			bisfirst = 0;
		++nn.epoch;
	}
	else {
		restoreWeight(1);	// best solution으로 테스트를 한다.
		forwardPropagate();
		test();
		cleanUPFoward();
	}
	cleanUp();
}




/////////////////////////////////////////////////////////////////////////
// CNN classs definition

void CNN::saveWeights() 
{
	char filename[100];
	sprintf(filename,weightFileNameFormat,epoch);
	ofstream f;
	f.open(filename,ios::out | ios::binary);
	for (int i=0; i<L.size(); i++)
		L[i].putWeightsToStream(f);
	f.write( (char*)&(*g_pregularizingConstants)[0],sizeof(float)*nInputFeatures);
	f.close();
}


void CNN::loadWeights() 
{
	char filename[100];
	sprintf(filename,weightFileNameFormat,epoch);
	ifstream f;
	f.open(filename,ios::in | ios::binary);
	if (!f) {
		cout <<"Cannot find " << filename << endl;
		exit(EXIT_FAILURE);
	}
	cout << "Loading network parameters from " << filename << endl;
	for (int i=0; i<L.size(); i++)
		L[i].loadWeightsFromStream(f);
	(*g_pregularizingConstants).reserve(nInputFeatures);
	f.read((char*)&(*g_pregularizingConstants)[0],sizeof(float)*nInputFeatures);
	f.close();
}


CNN::CNN(int nInputFeatures, int nInputSpatialSize, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch) :
	nInputFeatures(nInputFeatures),
	nInputSpatialSize(nInputSpatialSize),
	learningRate(learningRate),
	momentumDecayRate(momentumDecayRate),
	weightDecayRate(weightDecayRate),
	epoch(epoch) 
{
		cudaError_t cudaErr = cudaStreamCreate(&cublasstream);

        if (cudaErr != cudaSuccess) {
			cout << "!!!! cannot create stream: " << cudaErr << endl;
			exit(EXIT_FAILURE);
		}

		cublasStatus_t ret = cublasCreate(&cublasHandle);
		if (ret != CUBLAS_STATUS_SUCCESS) {
			cout << "cublasCreate returned error code" << ret << endl;
			exit(EXIT_FAILURE);
		}
		//cublasSetStream(cublasHandle, cublasstream);

		loadData();
#ifdef VALIDATION
		replaceTestSetWithValidationSet();
#endif
#ifdef SMALLERTESTSET
		smallerTestSet();
#endif
		cout << "size0\t#in\tdropout\tFilter\tsize1\tkMaxout\tPool\tsize2\t#out\tFunction\n";
}


CNN::~CNN() 
{
	cublasDestroy(cublasHandle);
	cudaStreamDestroy(cublasstream);

	for (int i=0; i<(*g_ptrainCharacters).size(); i++)
		delete (*g_ptrainCharacters)[i];
	(*g_ptrainCharacters).resize(0);
	for (int i=0; i<(*g_ptestCharacters).size(); i++)
		delete (*g_ptestCharacters)[i];
	(*g_ptestCharacters).resize(0);
}



void CNN::addLayer(int filterSize, int poolSize, int nFilters, float learningRateScale, sigmoidType sigmoidFunction, float dropoutProbability, int kMaxout) 
{
	if (kMaxout>1)
		sigmoidFunction=NOSIGMOID;
	int s0, nIn;
	if (L.size()==0) {
		s0=nInputSpatialSize;
		nIn=nInputFeatures;
	} else {
		s0=L.back().s2;
		nIn=L.back().nOut;
	}
	if (filterSize>s0) {
		cout << "filterSize is too big for this layer!"<<endl;
		exit(EXIT_FAILURE);
	}
	if ((s0-filterSize+1)%poolSize!=0) {
		cout << "poolSize does not divide the size of the output of the filters for this layer!"<<endl;
		exit(EXIT_FAILURE);
	}
	L.push_back(ConvolutionalLayer(filterSize,
		poolSize,
		s0, s0-filterSize+1, (s0-filterSize+1)/poolSize,
		nIn, nFilters,
		learningRateScale,
		sigmoidFunction,
		dropoutProbability,
		kMaxout));
	cout << L.back().s0 << "\t"
		<< L.back().nIn << "\t"
		<< L.back().dropoutProbability << "\t"
		<< L.back().filterSize << "\t"
		<< L.back().s1 << "\t"
		<< L.back().kMaxout << "\t"
		<< L.back().poolSize << "\t"
		<< L.back().s2 << "\t"
		<< L.back().nOut << "\t"
		<< sigmoidNames[sigmoidFunction] << "\n";
}


void CNN::initialize() 
{
	if (epoch>0)
		loadWeights();
	else {
		cout << "Initialized network parameters using the uniform distribution." << endl;
		calculateRegularizingConstants(nInputFeatures);
	}
}



/////////////////////////////////////////////////////////////////////////
// DeepCNet
const sigmoidType baseSigmoidType=RECTIFIEDLINEAR;

DeepCNet::DeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
		float momentumDecayRate, float weightDecayRate, int epoch,
		vector<float>	learningRateScales,
		vector<float> dropoutProbabilities,
		vector<int> kMaxouts) :
		CNN(nInputFeatures, 3*(1<<l), learningRate, momentumDecayRate, weightDecayRate, epoch)
{
	if (scale_N!=nInputSpatialSize) {
		cout << "scale_N should be " << 3*(1<<l) << endl;
		exit(EXIT_FAILURE);
	}
	if (dropoutProbabilities.empty()) dropoutProbabilities.resize(l+2,0);
	if (dropoutProbabilities.size()!=l+2) {
		cout << "Need " << l+2<< " dropout probabilities." << endl;
		exit(EXIT_FAILURE);
	}
	if (kMaxouts.empty()) kMaxouts.resize(l+2,1);
	if (kMaxouts.size()!=l+2) {
		cout << "Need " << l+2<< " kMaxout values." << endl;
		exit(EXIT_FAILURE);
	}
	addLayer(3, 2, k, learningRateScales[0], baseSigmoidType, dropoutProbabilities[0], kMaxouts[0]);
	for (int i=2; i<=l; i++)
		addLayer(2, 2, i*k, learningRateScales[i-1], baseSigmoidType, dropoutProbabilities[i-1], kMaxouts[i-1]);
	addLayer(2,1,(l+1)*k,
		learningRateScales[l],
		baseSigmoidType
		//TANH
		,dropoutProbabilities[l],kMaxouts[l]);
	addLayer(1,1,nOutputClasses, learningRateScales[l+1], SOFTMAX, dropoutProbabilities[l+1], kMaxouts[l+1]);
	initialize();
	cout << "DeepCNet(" << l << "," << k << ")" << endl;
}




/////////////////////////////////////////////////////////////////////////
// DeepDeepCNet
DeepDeepCNet::DeepDeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
		float momentumDecayRate, float weightDecayRate, int epoch,
		vector<float> dropoutProbabilities,
		vector<int> kMaxouts) :
		CNN(nInputFeatures, 3*(1<<l), learningRate, momentumDecayRate, weightDecayRate, epoch) 
{
	if (scale_N!=nInputSpatialSize) {
		cout << "scale_N shoule be " << 3*(1<<l) << endl;
		exit(EXIT_FAILURE);
	}
	if (dropoutProbabilities.empty()) dropoutProbabilities.resize(l+3,0);
	if (dropoutProbabilities.size()!=l+3) {
		cout << "Need " << l+3<< " dropout probabilities." << endl;
		exit(EXIT_FAILURE);
	}
	if (kMaxouts.empty()) kMaxouts.resize(l+3,1);
	if (kMaxouts.size()!=l+3) {
		cout << "Need " << l+3<< " kMaxout values." << endl;
		exit(EXIT_FAILURE);
	}
	addLayer(3, 2, k, 1.0f,baseSigmoidType,dropoutProbabilities[0],kMaxouts[0]);
	for (int i=2; i<=l; i++)
		addLayer(2, 2, i*k, 1.0f,baseSigmoidType,dropoutProbabilities[i-1],kMaxouts[i-1]);
	addLayer(2,1,(l+1)*k,1.0f,TANH,dropoutProbabilities[l],kMaxouts[l]);
	addLayer(1,1,(l+2)*k,1.0f,TANH,dropoutProbabilities[l+1],kMaxouts[l+1]);
	addLayer(1,1,nOutputClasses, 1.0f,SOFTMAX,dropoutProbabilities[l+2],kMaxouts[l+2]);
	initialize();
	cout << "DeepDeepCNet(" << l << "," << k << ")" << endl;
}


//////////////////////////////////////////////////////////////////////////////////
// FlatDeepCNet

FlatDeepCNet::FlatDeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
	float momentumDecayRate, float weightDecayRate, int epoch,
	vector<float> dropoutProbabilities,
	vector<int> kMaxouts) :
	CNN(nInputFeatures, 3*(1<<l), learningRate, momentumDecayRate, weightDecayRate, epoch) 
{
	if (scale_N!=nInputSpatialSize) {
		cout << "scale_N shoule be " << 3*(1<<l) << endl;
		exit(EXIT_FAILURE);
	}
	if (dropoutProbabilities.empty()) dropoutProbabilities.resize(l+2,0);
	if (dropoutProbabilities.size()!=l+2) {
		cout << "Need " << l+2<< " dropout probabilities." << endl;
		exit(EXIT_FAILURE);
	}
	if (kMaxouts.empty()) kMaxouts.resize(l+2,1);
	if (kMaxouts.size()!=l+2) {
		cout << "Need " << l+2<< " kMaxout values." << endl;
		exit(EXIT_FAILURE);
	}
	addLayer(3, 2, k, 1.0f,baseSigmoidType,dropoutProbabilities[0],kMaxouts[0]);
	for (int i=2; i<=l; i++)
		addLayer(2, 2, k, 1.0f,baseSigmoidType,dropoutProbabilities[i-1],kMaxouts[i-1]);
	addLayer(2,1,k,1.0f,baseSigmoidType,dropoutProbabilities[l],kMaxouts[l]);
	addLayer(1,1,nOutputClasses, 1.0f,SOFTMAX,dropoutProbabilities[l+1],kMaxouts[l+1]);
	initialize();
	cout << "FlatDeepCNet(" << l << "," << k << ")" << endl;
}



//////////////////////////////////////////////////////////////////////////////////
// LeNet5
LeNet5::LeNet5(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch) :
	CNN(nInputFeatures, scale_N, learningRate, momentumDecayRate, weightDecayRate, epoch) {
		addLayer(5,2, 2*sizeMultiplier,1.0f,baseSigmoidType);
		addLayer(5,2, 5*sizeMultiplier,1.0f,baseSigmoidType);
		addLayer(L.back().s2,1,50*sizeMultiplier,1.0f,TANH);
		addLayer(1,1,nOutputClasses,1.0f,SOFTMAX);
		initialize();
		cout << "LeNet5: sizeMultiplier = " << sizeMultiplier << endl;
}

/////////////////////////////////////////////////////////////////////////////////
// LeNet7
LeNet7::LeNet7(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch, vector<float>	learningRateScales) :
	CNN(nInputFeatures, scale_N, learningRate, momentumDecayRate, weightDecayRate, epoch) {
		addLayer(5,4, 1*sizeMultiplier,learningRateScales[0],baseSigmoidType);
		addLayer(6,3, 3*sizeMultiplier,learningRateScales[1],baseSigmoidType);
		addLayer(6,1,12*sizeMultiplier,learningRateScales[2],baseSigmoidType);
		addLayer(1,1,60*sizeMultiplier,learningRateScales[3],baseSigmoidType);
		addLayer(1,1,nOutputClasses,learningRateScales[4],SOFTMAX,0.0f);
		initialize();
		cout << "LeNet7: sizeMultiplier = " << sizeMultiplier << endl;
}

/////////////////////////////////////////////////////////////////////////////////
// LeNetJ
LeNetJ::LeNetJ(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch, vector<float>	learningRateScales) :
	CNN(nInputFeatures, scale_N, learningRate, momentumDecayRate, weightDecayRate, epoch) {
		addLayer(5,4, 10*sizeMultiplier,learningRateScales[0],baseSigmoidType);
		addLayer(6,3, 15*sizeMultiplier,learningRateScales[1],baseSigmoidType);
		addLayer(6,1, 15*sizeMultiplier,learningRateScales[2],baseSigmoidType);
		addLayer(1,1, 10*sizeMultiplier,learningRateScales[3],baseSigmoidType);
		addLayer(1,1, nOutputClasses,learningRateScales[4],SOFTMAX, 0.5f);
		initialize();
		cout << "LeNetJ: sizeMultiplier = " << sizeMultiplier << endl;
}