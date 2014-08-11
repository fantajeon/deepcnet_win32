

struct Feature {
  float x;
  float y;
  float val[nInputFeatures];
};

typedef vector<Feature> Features;

class FeatureCloudPicture : public Picture 
{
public:
  Features features;
  void codifyInputData (SparseCnnInterface &interface);
  Picture* distort ();
  ~FeatureCloudPicture() {}
};


//Fit inside a scale_n x scale_n box inside [scale_N,scale_N]^2
void normalize(Features &fs);
void featureCloudJiggle(Features &features, RNG &rng, float max_delta);

#include "readOfflineCasiaSIFT.h"
