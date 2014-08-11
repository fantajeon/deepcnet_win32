//#ifdef CIFAR100
#if 0

#include <vector>
#include <boost/assign/list_of.hpp>
#include "CNN.h"
#include "run.h"

const int scale_N=96;
const int trainingBatchSize=100;
//const float learningRate=0.0005/(double)trainingBatchSize;
const float learningRate=0.05/(double)trainingBatchSize;
const float learningRateDecayRate=1.0/(double)trainingBatchSize;
const int nInputFeatures=3;
//const int startEpoch=758000;
const int startEpoch=0;
const int nCharacters=100;
const char weightFileNameFormat[]="cifar100_epoch-%d.cnn";




#define ACTION train_test(2000,2000)
#include "CNN.h"
#include "OfflineGrid.h"

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

//Picture* OfflineGridPicture::distort() 
//{
//	OfflineGridPicture* pic=new OfflineGridPicture(xSize+40,ySize+40,label);
//	RNG rng;
//	float xStretch=rng.uniform(-0.2,0.2);
//	float yStretch=rng.uniform(-0.2,0.2);
//	int flip_h=rng.randint(2);
//	int r=rng.randint(3);
//	float alpha=rng.uniform(-0.2,0.2);
//
//	for (int y=0; y<pic->ySize; y++) {
//		for (int x=0; x<pic->xSize;x++) {
//			FloatPoint p(x+pic->xOffset+0.5,y+pic->yOffset+0.5);
//			p.stretch_x(xStretch);
//			p.stretch_y(yStretch);
//			if (flip_h==1) p.flip_horizontal();
//			if (r==0) p.rotate(alpha);
//			if (r==1) p.slant_x(alpha);
//			if (r==2) p.slant_y(alpha);
//			for (int i=0; i<nInputFeatures; i++)
//				pic->bitmap[x+y*pic->xSize+i*pic->xSize*pic->ySize]=interpolate(p, i);
//		}
//	}
//
//	pic->jiggle(rng,2);
//	return pic;
//}


void readBINFile(vector<Picture*> &characters, const char* filename, bool mirror) 
{
	ifstream file(filename,ios::in|ios::binary);
	if (!file) {
		cout <<"Cannot find " << filename << endl;
		exit(EXIT_FAILURE);
	}
	cout << "\r" << filename;
	unsigned char label[2];
	while (file.read((char*)label,2)) {
		OfflineGridPicture* character = new OfflineGridPicture(32,32,label[1]);
		unsigned char bitmap[3072];
		file.read((char*)bitmap,3072);
		for (int i=0;i<3072;i++)
			character->bitmap[i]=bitmap[i]-128.0; //Grey == (0,0,0)
		characters.push_back(character);
	}
	file.close();
}


void loadData()
{
	char filenameTrain[]="./Data/CIFAR100/train.bin";

	readBINFile((*g_ptrainCharacters),filenameTrain);
	char filenameTest[]="./Data/CIFAR100/test.bin";
	readBINFile((*g_ptestCharacters),filenameTest);
	cout <<" " << (*g_ptrainCharacters).size()<< " " << (*g_ptestCharacters).size() << endl;
}

int main() {
	init_cnn();

	//g_pcnn = new FlatDeepCNet(5,500,nInputFeatures,nCharacters,learningRate,0.5,0,startEpoch
	//			 , boost::assign::list_of(0.0)(0.0)(0.1)(0.2)(0.3)(0.4)(0.5)
	//			 //, boost::assign::list_of(1)(1)(1)(1)(1)(1)(5)
	//			 );
	g_pcnn = new LeNet7(5,nInputFeatures,nCharacters,learningRate,0.5,0,startEpoch,boost::assign::list_of(1.0f)(1.0f)(1.0f)(1.0f)(1.0f));
	//g_pcnn = new LeNetJ(15,nInputFeatures,nCharacters,learningRate,0.5,0,startEpoch);
	g_pccnn = new ComputationalCNN(*g_pcnn);

	ACTION;


	deinit_cnn();

	delete g_pcnn;
	delete g_pccnn;
	return 0;
}



#endif