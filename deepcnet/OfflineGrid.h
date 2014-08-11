#pragma once

#include <vector>
#include "FloatPoint.h"
#include "CNN.h"

class Picture;
class SparseCnnInterface;
class RNG;

class OfflineGridPicture : public Picture 
{
public:
  short int xOffset;
  short int yOffset;
  short int xSize;
  short int ySize;
  std::vector<float> bitmap; //nInputFeatures*ySize*xSize (row major order)
  void codifyInputData (SparseCnnInterface &interface);
  Picture* distort ();

  OfflineGridPicture(int xSize, int ySize, int label_ = -1);
  ~OfflineGridPicture();
  void jiggle(RNG &rng, int offlineJiggle);
  // void drawPGM(char* filename) {
  //   ofstream f(filename);
  //   f << "P2\n"<< xSize*nInputFeatures << " " << ySize<< endl<< 255<< endl;
  //   for (int y=0; y<ySize; y++) {
  //     for (int i=0; i<nInputFeatures;i++) {
  //       for (int x=0; x<xSize;x++) {
  //         f << (int)(127.9*bitmap[x+y*xSize+i*ySize*xSize]/regularizingConstants[i]+128) << " ";
  //       }
  //       f << endl;
  //     }
  //     f << endl;
  //   }
  //   f.close();
  // }
  void drawPGM(char* filename);
  void drawPPM(char* filename);
  float interpolate(FloatPoint& p, int i);
};


void readBINFile(vector<Picture*> &characters, const char* filename, bool mirror=false) ;