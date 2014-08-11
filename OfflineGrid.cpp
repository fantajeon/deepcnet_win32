#include "OfflineGrid.h"



//	// 해당 포멧에 맞게 구현을 해야한다.
//Picture* OfflineGridPicture::distort ()
//{
//}

OfflineGridPicture::OfflineGridPicture(int xSize, int ySize, int label_) 
	: xSize(xSize), ySize(ySize) 
{
	label=label_;
	xOffset=-xSize/2;
	yOffset=-ySize/2;
	bitmap.resize(nInputFeatures*ySize*xSize);
}


OfflineGridPicture::~OfflineGridPicture() {}


void OfflineGridPicture::jiggle(RNG &rng, int offlineJiggle) 
{
	xOffset+=rng.randint(offlineJiggle*2+1)-offlineJiggle;
	yOffset+=rng.randint(offlineJiggle*2+1)-offlineJiggle;
}

void OfflineGridPicture::drawPGM(char* filename) 
{
	ofstream f(filename);
	f << "P2\n"<< scale_N*nInputFeatures << " " << scale_N<< endl<< 255<< endl;
	for (int y=-scale_N/2; y<scale_N/2; y++) {
		for (int i=0; i<nInputFeatures;i++) {
			for (int x=-scale_N/2; x<scale_N/2;x++) {
				FloatPoint p(x+0.5,y+0.5);
				f << (int)(127.9*interpolate(p,i)/(*g_pregularizingConstants)[i]+128) << " ";
			}
			f << endl;
		}
		f << endl;
	}
	f.close();
}


void OfflineGridPicture::drawPPM(char* filename) 
{ //for when nInputFeatures==3
	ofstream f(filename);
	f << "P3\n"<< xSize<< " " << ySize<< endl<< 255<< endl;
	for (int y=0; y<ySize; y++) {
		for (int x=0; x<xSize;x++) {
			for (int col=0; col<3; col++) {
				f << 128+(int)bitmap[x+y*xSize+col*xSize*ySize] << " ";
			}
		}
		f << endl;
	}
	f << endl;
	f.close();
}


float OfflineGridPicture::interpolate(FloatPoint& p, int i) 
{
	return bilinearInterpolationScaled<float>
		(p, &bitmap[i*xSize*ySize],
		xSize, ySize,
		xOffset,       yOffset,
		xOffset+xSize, yOffset+ySize);
}

void OfflineGridPicture::codifyInputData (SparseCnnInterface &interface) 
{
	for  (int i=0; i<nInputFeatures; i++)
		interface.features.push_back(0); //Background feature
	int backgroundNullVectorNumber=interface.count++;
	interface.backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
	vector<int> grid(scale_N*scale_N,backgroundNullVectorNumber);
	for (int x=0; x<xSize; x++) {
		for (int y=0; y<ySize; y++) {
			if (x+xOffset+scale_N/2>=0 && x+xOffset+scale_N/2<scale_N &&
				y+yOffset+scale_N/2>=0 && y+yOffset+scale_N/2<scale_N) {
					bool flag=false;
					for (int i=0; i<nInputFeatures; i++)
						if (abs(bitmap[x+y*xSize+i*xSize*ySize])>0.005*(*g_pregularizingConstants)[i])
							flag=true;
					if (flag) {
						int n=(x+xOffset+scale_N/2)*scale_N+(y+yOffset+scale_N/2);
						grid[n]=interface.count++;
						for (int i=0; i<nInputFeatures; i++)
							interface.features.push_back
							(bitmap[x+y*xSize+i*xSize*ySize]/(*g_pregularizingConstants)[i]);
					}
			}
		}
	}
	interface.grids.push_back(grid);
	while (interface.featureSampleNumbers.size() < interface.count)
		interface.featureSampleNumbers.push_back(interface.batchSize);
	interface.batchSize++;
	interface.labels.push_back(label);
}
