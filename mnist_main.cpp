//#ifdef MNIST
#if 1


#include <vector>
#include <boost/assign/list_of.hpp>
#include "CNN.h"
#include "run.h"
#include "OfflineGrid.h"

//const int nCharacters=10;
//const int nInputFeatures=1;
//const int scale_N=96;
//const int startEpoch=0*1000;
//const char weightFileNameFormat[]="mnist_epoch-%d.cnn";
//const int trainingBatchSize=100;
//const float learningRate=1.0f/(float)trainingBatchSize;
//const float learningRateDecayRate=learningRate/(1e+6);


const int nCharacters=10;
const int nInputFeatures=1;
const int scale_N=96;
const int startEpoch=0;
const char weightFileNameFormat[]="mnist_epoch-%d.cnn";
const int trainingBatchSize=50;
//const float learningRate=0.5f/(float)trainingBatchSize;
const float learningRate=1.0f;
const float learningRateDecayRate=1e-12;
const int decreasing_check_epoch = 2000;


#define ACTION train_test(6000,6000)


CNN *g_pcnn = NULL;
ComputationalCNN *g_pccnn = NULL;

Picture* OfflineGridPicture::distort() 
{
	OfflineGridPicture* pic=new OfflineGridPicture(*this);
	if(pic==NULL) 
	{
		cout<<"distor() null"<<endl;
	}
	RNG rng;
	pic->jiggle(rng,2);
	return pic;
}

static int intToggleEndianness(int a) 
{
	int b=0;
	b+=a%256*(1<<24);a>>=8;
	b+=a%256*(1<<16);a>>=8;
	b+=a%256*(1<< 8);a>>=8;
	b+=a%256*(1<< 0);
	return b;
}

static void loadMnistC(string filename, vector<Picture*> &characters) 
{
	ifstream f(filename.c_str(), ios::in|ios::binary);
	if (!f) {
		cout <<"Cannot find " << filename << endl;
		exit(EXIT_FAILURE);
	}

	int a,n1,n2,n3;
	f.read((char*)&a,4);
	f.read((char*)&a,4);
	n1=intToggleEndianness(a);
	f.read((char*)&a,4);
	n2=intToggleEndianness(a);
	f.read((char*)&a,4);
	n3=intToggleEndianness(a);
	for (int i1=0; i1 < n1; i1++) {
		OfflineGridPicture* character = new OfflineGridPicture(n2,n3);
		unsigned char *bitmap=new unsigned char[n2*n3];
		memset(bitmap, 0xCC, sizeof(unsigned char)*n2*n3);
		f.read((char *)bitmap,n2*n3);
		for (int j=0;j<n2*n3;j++)
			character->bitmap[j]=(float)bitmap[j]-128.0f;
			//character->bitmap[j]=(float)bitmap[j];
		characters.push_back(character);
		delete [] bitmap;
	}
}

static void loadMnistL(string filename, vector<Picture*> &characters) 
{
	ifstream f(filename.c_str(), ios::in|ios::binary);
	if (!f) {
		cout <<"Cannot find " << filename << endl;
		exit(EXIT_FAILURE);
	}

	int a,n;
	char l;

	f.read((char*)&a,4);
	f.read((char*)&a,4);
	n=intToggleEndianness(a);
	for (int i=0;i<n;i++) {
		f.read(&l, 1);
		characters[i]->label=l;
	}
}

void loadData() 
{
	string trainC("Data/MNIST/train-images-idx3-ubyte");
	string trainL("Data/MNIST/train-labels-idx1-ubyte");
	string testC("Data/MNIST/t10k-images-idx3-ubyte");
	string testL("Data/MNIST/t10k-labels-idx1-ubyte");
	loadMnistC(trainC, (*g_ptrainCharacters));
	loadMnistL(trainL, (*g_ptrainCharacters));
	loadMnistC(testC, (*g_ptestCharacters));
	loadMnistL(testL, (*g_ptestCharacters));
}

int main() 
{
	init_cnn();

	__cudaCheckError(__FILE__, __LINE__);
	//g_pcnn = new DeepCNet(5,60,
	//				nInputFeatures, nCharacters,
	//				learningRate, 0.5, 0, startEpoch
	//				, boost::assign::list_of(0.0)(0.0)(0.1)(0.2)(0.3)(0.4)(0.5)
	//				//, boost::assign::list_of(3)(3)(3)(3)(3)(3)(5)
	//				);
	
	__cudaCheckError(__FILE__, __LINE__);
	g_pcnn = new LeNet7(5,nInputFeatures,nCharacters,learningRate,0.1,0,startEpoch,boost::assign::list_of(3)(3)(3)(3)(3));
	g_pccnn = new ComputationalCNN(*g_pcnn);
	__cudaCheckError(__FILE__, __LINE__);

	ACTION;


	deinit_cnn();

	delete g_pcnn;
	delete g_pccnn;
	return 0;
}

#endif // #define MNIST