#pragma once

#include "CNN.h"
#include "FloatPoint.h"

class SparseCnnInterface;
class Picture;

typedef std::vector<FloatPoint> OnlinePenStroke;
typedef std::vector<OnlinePenStroke> OnlinePenStrokes;


class OnlinePicture : public Picture {
public:
  OnlinePenStrokes ops;
  void codifyInputData (SparseCnnInterface &interface);
  Picture* distort ();
  ~OnlinePicture() {}
};



void stretchXY(OnlinePenStrokes &character, RNG &rng, float max_stretch);
void rotate(OnlinePenStrokes &character, RNG &rng, float max_angle);
void slant_x(OnlinePenStrokes &character, RNG &rng, float max_alpha);
void slant_y(OnlinePenStrokes &character, RNG &rng, float max_alpha);
void stretch4(OnlinePenStrokes &character, RNG &rng, float max_stretch);
void characterED(OnlinePenStrokes &character, EDfield &f);
void jiggleCharacter(OnlinePenStrokes &character, RNG &rng, float max_delta);
void jiggleStrokes(OnlinePenStrokes &character, RNG &rng, float max_delta);


void printStroke(OnlinePenStroke &stroke);
void printOnlinePenStrokes(OnlinePenStrokes &character);
OnlinePenStrokes breakSharpAngles(OnlinePenStrokes &in);
//Fit characters inside a scale_n x scale_n box, center the origin
void normalize(OnlinePenStrokes &ops);



//Signature functions
int mm(int m);
void additiveKron(float* a,float* b,float* c,int m,int n);
void signatureAdditiveKron(float* a, float* b, float* c, int i, int j);
void signatureProduct(float* sig1, float* sig2, int m);
void multiplicativeKron(float* a,float* b,float* c,float alpha, int m,int n);
void oneStepSignature(vector<float> &sig, float dx, float dy, int m);
vector<float> signature(OnlinePenStroke &path,int m, int start, int finish);
vector<vector<float> > signatureWindows(OnlinePenStroke &path, int m, int windowSize, vector<float> &regularizingConstants);
int mapToGrid(float coord,int resolution);
OnlinePenStroke constantSpeed(OnlinePenStroke &path, float density, int multiplier = 1);
void OnlinePicture::codifyInputData (SparseCnnInterface &interface);
void drawGraphs(OnlinePenStrokes &paths, OnlinePenStrokes &paths2);
void show_characters();

//#include "readAssamese.h"
//#include "readUJIpenchars.h"
//#include "readPendigits.h"
//#include "readOnlineCasia.h"

