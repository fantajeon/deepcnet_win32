#include "FeatureCloud.h"


void FeatureCloudPicture::codifyInputData (SparseCnnInterface &interface) 
{
  interface.batchSize++;
  interface.labels.push_back(label);
  vector<int> grid;
  if (features.empty())
    interface.labelLocationInOutput.push_back(0);
  else {
    grid.resize(scale_N*scale_N,0);
    interface.labelLocationInOutput.push_back(interface.outputsCount++);
    for (int i=0; i<features.size(); i++) {
      int x=max(0,min(scale_N-1,(int)features[i].x));
      int y=max(0,min(scale_N-1,(int)features[i].y));
      int n=x*scale_N+y;
      if (grid[n]==0) {
        grid[n]=interface.count++;
        for (int k=0; k<nInputFeatures; k++)
          interface.features.push_back(features[i].val[k]/regularizingConstants[k]);
      }
    }
  }
  interface.grids.push_back(grid);
}

//Fit inside a scale_n x scale_n box inside [scale_N,scale_N]^2
void normalize(Features &fs)
{
  float x=fs[0].x;
  float X=fs[0].x;
  float y=fs[0].y;
  float Y=fs[0].y;
  for (int i=0;i<fs.size();i++) {
    x=min(fs[i].x,x);
    X=max(fs[i].x,X);
    y=min(fs[i].y,y);
    Y=max(fs[i].y,Y);
  }
  float scaleF=scale_n/max(max(X-x,Y-y),0.0001f);
  for (int i=0;i<fs.size();i++) {
    fs[i].x=(fs[i].x-0.5*(X+x))*scaleF+0.5*scale_N;
    fs[i].y=(fs[i].y-0.5*(Y+y))*scaleF+0.5*scale_N;
  }
}

void featureCloudJiggle(Features &features, RNG &rng, float max_delta)
{
  float dx=rng.uniform(-max_delta,max_delta);
  float dy=rng.uniform(-max_delta,max_delta);
  for (int i=0;i<features.size();i++) {
    features[i].x+=dx;
    features[i].y+=dy;
  }
}