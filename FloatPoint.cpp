#include "RNG.h"
#include "FloatPoint.h"

///////////////////////////////////////////////////////////////////////////////////
// FloatPoint
FloatPoint::FloatPoint() 
{
}

FloatPoint::FloatPoint(float x, float y) : x(x), y(y) 
{

}
void FloatPoint::flip_horizontal() 
{
	x=-x;
}


void FloatPoint::stretch_x(float alpha) 
{
	x*=(1+alpha);
}


void FloatPoint::stretch_y(float alpha) 
{
	y*=(1+alpha);
}

void FloatPoint::rotate(float angle) 
{
	float c=cos(angle);
	float s=sin(angle);
	float xx=+x*c+y*s;
	float yy=-x*s+y*c;
	x=xx;
	y=yy;
}


void FloatPoint::slant_x(float alpha)
{
	y+=alpha*x;
}


void FloatPoint::slant_y(float alpha) 
{
	x+=alpha*y;
}


void FloatPoint::stretch4(float cxx, float cxy, float cyx, float cyy) 
{
	float tx=x;
	float ty=y;
	x=(1+cxx)*tx+cxy*ty;
	y=(1+cyy)*ty+cyx*tx;
}


void FloatPoint::stretch(EDfield& f) 
{
	f.stretch(*this);
}


////////////////////////////////////////////////////////////////////////////////////////////
// EDfield
void EDfields::t(int j) 
{
	for (int i=j;i<edf.size();i+=6)
		edf[i]=EDfield(resolution,scale,sigma,amp);
}


EDfields::EDfields(int n, int resolution, int scale, float sigma, float amp)
: resolution(resolution), scale(scale), sigma(sigma), amp(amp) 
{
	edf.resize(n);
	for (int i=0; i<6; i++)
		tg.add_thread(new boost::thread(boost::bind(&EDfields::t,this,i)));
	tg.join_all();
}


void EDfield::convolve_gaussian(std::vector<float> &a, float sigma, int n) 
{ //inplace
  std::vector<float> b(n*n,0);
  for (int i=0;i<n;i++) {
    for (int j=0; j<n; j++) {
      for (int k= (std::max<int>)(0,j-3*sigma);k<=(std::min<int>)(n-1,j+3*sigma);k++)
        b[i*n+j]+=a[i*n+k]*exp(-(k-j)*(k-j)/2.0f/sigma/sigma)/sigma/0.82; //Gives EDfield components mean absolute magnitude amp.
    }
  }
  for (int i=0;i<n;i++) {
    for (int j=0; j<n; j++) {
      a[i*n+j]=0;
      for (int k=(std::max<int>)(0,i-3*sigma);k<=(std::min<int>)(n-1,i+3*sigma);k++)
        a[i*n+j]+=b[k*n+j]*exp(-(k-i)*(k-i)/2.0f/sigma/sigma);
    }
  }
}

EDfield::EDfield (int resolution, int scale, float sigma, float amp) :
  resolution(resolution), scale(scale) {
  RNG rng;
  rf.resize(2);
  for (int k=0; k<2; k++) {
    rf[k].resize(resolution*resolution);
    for (int i=0;i<resolution;i++) {
      for (int j=0; j<resolution; j++) {
        rf[k][i*resolution+j]=rng.uniform(-amp,amp);
      }
    }
    convolve_gaussian(rf[k], sigma, resolution);
  }
}
void EDfield::stretch(FloatPoint &p) {
  float dx=bilinearInterpolationScaled<float>(p, &rf[0][0], resolution, resolution, -0.6*scale, -0.6*scale, 0.6*scale, 0.6*scale);
  float dy=bilinearInterpolationScaled<float>(p, &rf[1][1], resolution, resolution, -0.6*scale, -0.6*scale, 0.6*scale, 0.6*scale);
  p.x+=dx;
  p.y+=dy;
}




