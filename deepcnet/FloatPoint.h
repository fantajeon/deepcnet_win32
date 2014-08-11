#pragma once

#include <vector>
#include <boost/thread.hpp>

class FloatPoint;
class EDfield {
  int resolution;
  int scale; //aim to operate on the square [-scale/2,scale/2]^2
  std::vector<std::vector<float>> rf;
public:
  void convolve_gaussian(std::vector<float> &a, float sigma, int n);
  EDfield (int resolution, int scale, float sigma, float amp);
  EDfield() {}
  void stretch(FloatPoint &p);
};

class FloatPoint {
public:
  float x;
  float y;
  FloatPoint();
  FloatPoint(float x, float y);
  void flip_horizontal();
  void stretch_x(float alpha);
  void stretch_y(float alpha);
  void rotate(float angle);
  void slant_x(float alpha);
  void slant_y(float alpha);
  void stretch4(float cxx, float cxy, float cyx, float cyy);
  void stretch(EDfield& f);
};

template<class T> T bilinearInterpolation(FloatPoint& p, T* array, int xSize, int ySize) {
  //Treat array as a rectangle [0,xSize]*[0,ySize]. Associate each value of the array with the centre of the corresponding unit square.
  int ix=floor(p.x-0.5);
  int iy=floor(p.y-0.5);
  float rx=p.x-0.5-ix;
  float ry=p.y-0.5-iy;
  T c00=0, c01=0, c10=0, c11=0;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c00=array[ix+iy*xSize];
  ix++;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c10=array[ix+iy*xSize];
  iy++;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c11=array[ix+iy*xSize];
  ix--;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c01=array[ix+iy*xSize];
  return (1-rx)*(1-ry)*c00+(1-rx)*ry*c01+rx*(1-ry)*c10+rx*ry*c11;
}

template<class T> T bilinearInterpolationScaled(FloatPoint p, T* array, int xSize, int ySize, float xMin, float yMin, float xMax,  float yMax) {
  p.x=(p.x-xMin)*xSize/(xMax-xMin);
  p.y=(p.y-yMin)*ySize/(yMax-yMin);
  return bilinearInterpolation<T>(p,array,xSize,ySize);
}


//Use to precalculate a large number of EDfield objects
class EDfields {
  int resolution, scale;
  float sigma, amp;
  boost::thread_group tg;
  void t(int j);

public:
  std::vector<EDfield> edf;
  EDfields(int n, int resolution, int scale, float sigma, float amp);
  void convolve_gaussian(std::vector<float> &a, float sigma, int n);
};
