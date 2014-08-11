#include "RNG.h"
#include "Online.h"


void stretchXY(OnlinePenStrokes &character, RNG &rng, float max_stretch)
{
	float dx=rng.uniform(-max_stretch,max_stretch);
	float dy=rng.uniform(-max_stretch,max_stretch);
	for (int i=0;i<character.size();i++) {
		for (int j=0;j<character[i].size();j++) {
			character[i][j].stretch_x(dx);
			character[i][j].stretch_y(dy);
		}}}

void rotate(OnlinePenStrokes &character, RNG &rng, float max_angle) {
	float angle=rng.uniform(-max_angle,max_angle);
	for (int i=0;i<character.size();i++) {
		for (int j=0;j<character[i].size();j++) {
			character[i][j].rotate(angle);
		}
	}
}
void slant_x(OnlinePenStrokes &character, RNG &rng, float max_alpha) {
	float alpha=rng.uniform(-max_alpha,max_alpha);
	for (int i=0;i<character.size();i++)
		for (int j=0;j<character[i].size();j++)
			character[i][j].slant_x(alpha);
}
void slant_y(OnlinePenStrokes &character, RNG &rng, float max_alpha) {
	float alpha=rng.uniform(-max_alpha,max_alpha);
	for (int i=0;i<character.size();i++)
		for (int j=0;j<character[i].size();j++)
			character[i][j].slant_y(alpha);
}
void stretch4(OnlinePenStrokes &character, RNG &rng, float max_stretch) {
	float cxx=rng.uniform(-max_stretch,+max_stretch);
	float cxy=rng.uniform(-max_stretch,+max_stretch);
	float cyx=rng.uniform(-max_stretch,+max_stretch);
	float cyy=rng.uniform(-max_stretch,+max_stretch);
	for (int i=0;i<character.size();i++)
		for (int j=0;j<character[i].size();j++)
			character[i][j].stretch4(cxx,cxy,cyx,cyy);
}
void characterED(OnlinePenStrokes &character, EDfield &f) {
	for (int i=0;i<character.size();i++)
		for (int j=0;j<character[i].size();j++)
			character[i][j].stretch(f);
}

void jiggleCharacter(OnlinePenStrokes &character, RNG &rng, float max_delta) {
	float dx=rng.uniform(-max_delta,max_delta);
	float dy=rng.uniform(-max_delta,max_delta);
	for (int i=0;i<character.size();i++)
		for (int j=0;j<character[i].size();j++) {
			character[i][j].x+=dx;
			character[i][j].y+=dy;
		}
}
void jiggleStrokes(OnlinePenStrokes &character, RNG &rng, float max_delta) {
	for (int i=0;i<character.size();i++) {
		float dx=rng.uniform(-max_delta,max_delta);
		float dy=rng.uniform(-max_delta,max_delta);
		for (int j=0;j<character[i].size();j++) {
			character[i][j].x+=dx;
			character[i][j].y+=dy;
		}
	}
}

class OnlinePicture : public Picture {
public:
	OnlinePenStrokes ops;
	void codifyInputData (SparseCnnInterface &interface);
	Picture* distort ();
	~OnlinePicture() {}
};

void printStroke(OnlinePenStroke &stroke) {
	cout <<" ";
	for (int i=0; i<stroke.size(); i++)
		cout <<"(" << stroke[i].x << "," << stroke[i].y <<") ";
	cout <<endl;
}

void printOnlinePenStrokes(OnlinePenStrokes &character) {
	cout << character.size() << " strokes"<<endl;
	for (int i=0; i<character.size(); i++)
		printStroke(character[i]);
}

OnlinePenStrokes breakSharpAngles(OnlinePenStrokes &in) {
	OnlinePenStrokes out;
	for (int i=0; i<in.size(); i++) {
		OnlinePenStroke ps;
		for (int j=0;j<in[i].size();j++) {
			ps.push_back(in[i][j]);
			if (j>0 && j<in[i].size()-1) {
				float dx1=in[i][j  ].x-in[i][j-1].x;
				float dy1=in[i][j  ].y-in[i][j-1].y;
				float dx2=in[i][j+1].x-in[i][j  ].x;
				float dy2=in[i][j+1].y-in[i][j  ].y;
				float dot=dx1*dx2+dy1*dy2;
				float m1=pow(pow(dx1,2)+pow(dy1,2),0.5);
				float m2=pow(pow(dx2,2)+pow(dy2,2),0.5);
				if (dot < -0.1*m1*m2) {
					out.push_back(ps);
					ps.clear();
					ps.push_back(in[i][j]);
				}
			}
		}
		out.push_back(ps);
	}
	return out;
}


//Fit characters inside a scale_n x scale_n box, center the origin
void normalize(OnlinePenStrokes &ops)
{
	float x=ops[0][0].x;
	float X=ops[0][0].x;
	float y=ops[0][0].y;
	float Y=ops[0][0].y;
	for (int i=0;i<ops.size();i++) {
		for (int j=0;j<ops[i].size();j++) {
			x=min(ops[i][j].x,x);
			X=max(ops[i][j].x,X);
			y=min(ops[i][j].y,y);
			Y=max(ops[i][j].y,Y);
		}
	}
	float scaleF=scale_n/max(max(X-x,Y-y),0.0001f);
	for (int i=0;i<ops.size();i++) {
		for (int j=0;j<ops[i].size();j++) {
			ops[i][j].x=(ops[i][j].x-0.5*(X+x))*scaleF;
			ops[i][j].y=(ops[i][j].y-0.5*(Y+y))*scaleF;
		}
	}
	//ops=breakSharpAngles(ops);
}



//Signature functions

int mm(int m) {
	return (1<<m)-1;
}

void additiveKron(float* a,float* b,float* c,int m,int n) {
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++) {
			c[i*n+j]+=a[i]*b[j];
		}
	}
}

void signatureAdditiveKron(float* a, float* b, float* c, int i, int j) {
	additiveKron(a+mm(i),b+mm(j),c+mm(i+j),1<<i,1<<j);
}

void signatureProduct(float* sig1, float* sig2, int m) {
	//sig1 inplace, times sig2
	for(int i=m;i>0;i--)
		for(int j=1;j<=i;j++)
			signatureAdditiveKron(sig1,sig2,sig1,i-j,j);
}

void multiplicativeKron(float* a,float* b,float* c,float alpha, int m,int n) {
	for (int i=0;i<m;i++)
		for (int j=0;j<n;j++)
			c[i*n+j]=alpha*a[i]*b[j];
}

void oneStepSignature(vector<float> &sig, float dx, float dy, int m) {
	sig[0]=1;
	if (m>0) {
		sig[1]=dx;
		sig[2]=dy;}
	for (int i=2;i<=m;i++) multiplicativeKron(&sig[mm(1)],&sig[mm(i-1)],&sig[mm(i)],1.0/i,1<<1,1<<(i-1));
}

vector<float> signature(OnlinePenStroke &path,int m, int start, int finish) {
	vector<float> sig1(mm(m+1),0);
	vector<float> sig2(mm(m+1),0);
	sig1[0]=sig2[0]=1;
	if (finish>start)
		oneStepSignature(sig1,path[start+1].x-path[start].x,path[start+1].y-path[start].y,m);
	for (int i=start+1;i<finish;i++) {
		oneStepSignature(sig2,path[i+1].x-path[i].x,path[i+1].y-path[i].y,m);
		signatureProduct(&sig1[0],&sig2[0],m);
	}
	return sig1;
}

vector<vector<float> > signatureWindows
	(OnlinePenStroke &path, int m, int windowSize, vector<float> &regularizingConstants) {
		vector<vector<float> > sW(path.size());
		for (int i=0;i<path.size();i++) {
			int first=max(0,i-windowSize);
			int last=min((int)path.size()-1,i+windowSize);
			sW[i]=signature(path,m,first,last);
			for (int j=0;j<nInputFeatures;j++) sW[i][j]/=regularizingConstants[j];
		}
		return sW;
}


int mapToGrid(float coord,int resolution) {
	return max(0,min(resolution-1,(int)(coord+0.5*scale_N)));
}

OnlinePenStroke constantSpeed(OnlinePenStroke &path, float density, int multiplier = 1)
{
	vector<float> lengths(path.size());
	lengths[0]=0;
	for (int i=1;i<path.size();i++) {
		lengths[i]=lengths[i-1]+pow(pow(path[i].x-path[i-1].x,2)+pow(path[i].y-path[i-1].y,2),0.5);
	}
	float lTotal=lengths[path.size()-1];
	int n=(int)(0.5+lTotal/density);
	n*=multiplier;
	OnlinePenStroke r(n+1);
	int j=0;
	float alpha;
	r[0].x=path[0].x;
	r[0].y=path[0].y;
	for (int i=1;i<=n;i++) {
		while(n*lengths[j+1]<i*lTotal) j++;
		alpha=(lengths[j+1]-i*lTotal/n)/(lengths[j+1]-lengths[j]);
		r[i].x=path[j].x*alpha+path[j+1].x*(1-alpha);
		r[i].y=path[j].y*alpha+path[j+1].y*(1-alpha);
	}
	return r;
}


// //alternative to ssg
// float octogramXDirections[]={1,0.707,0,-0.707,-1,-0.707, 0, 0.707};
// float octogramYDirections[]={0,0.707,1, 0.707, 0,-0.707,-1,-0.707,};
// void octogram(OnlinePenStrokes &paths, int resolution,
//               int& count, vector<int> &g, vector<float> &grid) {
//   vector<float> w(9*resolution*resolution,0);
//   for (int i=0; i<paths.size(); i++) {
//     OnlinePenStroke csp=constantSpeed(paths[i],0.5);
//     for (int j=0;j<csp.size()-1;j++) {
//       int n=mapToGrid(csp[j].x,resolution)*resolution+mapToGrid(csp[j].y,resolution);
//       float dx=(csp[j+1].x-csp[j].x)/0.5;
//       float dy=(csp[j+1].y-csp[j].y)/0.5;
//       w[9*n]=1;
//       for (int k=0;k<8;k++)
//         if (octogramXDirections[k]*dx+octogramYDirections[k]*dy>0.85) //cos(pi/8)=0.924
//           w[9*n+k+1]=1;
//     }
//   }
//   for (int n=0;n<resolution*resolution;n++)
//     if (w[9*n]==1) {
//       g[n]=count++;
//       for (int k=0;k<9;k++)
//         grid.push_back(w[9*n+k]);
//     }
// }

void OnlinePicture::codifyInputData (SparseCnnInterface &interface) {
	//Assume we need a null vector for the background
	for(int i=0;i<nInputFeatures;i++)
		interface.features.push_back(0);
	int backgroundNullVectorNumber=interface.count++;
	interface.backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
	vector<int> grid(scale_N*scale_N,backgroundNullVectorNumber);

	int windowSize=4;
	float scale=delta/windowSize;
	int multiplier=max((int)(2*scale+0.5),1);
	for (int i=0; i<ops.size(); i++) {
		OnlinePenStroke CSP=constantSpeed(ops[i],scale,1);
		OnlinePenStroke csp=constantSpeed(ops[i],scale,multiplier);
		vector<vector<float> > sW=
			signatureWindows(CSP,nIteratedIntegrals,windowSize,regularizingConstants);
		for (int j=0; j<csp.size(); j++) {
			int J=j/multiplier;
			int x=mapToGrid(csp[j].x,scale_N);
			int y=mapToGrid(csp[j].y,scale_N);
			int n=x*scale_N+y;
			if (grid[n]==backgroundNullVectorNumber) {
				grid[n]=interface.count++;
				for (int k=0; k<nInputFeatures; k++)
					interface.features.push_back(sW[J][k]);
			}
		}
	}
	interface.grids.push_back(grid);
	while (interface.featureSampleNumbers.size() < interface.count)
		interface.featureSampleNumbers.push_back(interface.batchSize);
	interface.batchSize++;
	interface.labels.push_back(label);
}


void drawGraphs(OnlinePenStrokes &paths, OnlinePenStrokes &paths2) {
	vector<int> g(2*scale_N*scale_N,0);
	for (int i=0; i<paths.size(); i++) {
		OnlinePenStroke csp=constantSpeed(paths[i],3.0,6);
		for (int j=0; j<csp.size(); j++) {
			int n=mapToGrid(csp[j].x,scale_N)*2*scale_N+mapToGrid(csp[j].y,scale_N);
			g[n]=1;
		}
	}
	for (int i=0; i<paths2.size(); i++) {
		OnlinePenStroke csp=constantSpeed(paths2[i],3.0,6);
		for (int j=0; j<csp.size(); j++) {
			int n=mapToGrid(csp[j].x,scale_N)*2*scale_N+mapToGrid(csp[j].y,scale_N)+scale_N;
			g[n]=1;
		}
	}
	for(int i=0; i< scale_N+2;i++) cout <<"--";cout<<endl;
	for(int i=0; i<scale_N; i++) {
		cout <<".";
		for(int j=0; j<2*scale_N; j++) {
			if (g[i*2*scale_N+j]==0)
				cout << " ";
			else
				cout << "X";
		}
		cout <<"."<< endl;
	}
	for(int i=0; i< scale_N+2;i++) cout <<"--";cout<<endl;
}

void show_characters() {
	RNG rng;
	OnlinePenStroke l;
	{
		FloatPoint p;
		float sn=0.5*(scale_n+1);
		p.x=-sn;
		for (p.y=-sn;p.y<sn;p.y++)
			l.push_back(p);
		p.y=sn;
		for (p.x=-sn;p.x<sn;p.x++)
			l.push_back(p);
		p.x=sn;
		for (p.y=sn;p.y>-sn;p.y--)
			l.push_back(p);
		p.y=-sn;
		for (p.x=sn;p.x>-sn;p.x--)
			l.push_back(p);
	}
	while(true) {
		int i=rng.index(trainCharacters);
		OnlinePicture* a=new OnlinePicture(*dynamic_cast<OnlinePicture*>(trainCharacters[i]));
		a->ops.push_back(l);
		OnlinePicture* b=dynamic_cast<OnlinePicture*>(a->distort());
		cout << i << " " << a->label<<endl;
		printOnlinePenStrokes(a->ops);
		drawGraphs(a->ops, b->ops);
		delete a, b;
		sleep(4);
	}
}


//Example distortion functions

// Picture* OnlinePicture::distort() {
//   OnlinePicture* pic=new OnlinePicture(*this);
//   RNG rng;
//   jiggleStrokes(pic->ops,rng,1);
//   stretchXY(pic->ops,rng,0.3);
//   int r=rng.randint(3);
//   if (r==0) rotate(pic->ops,rng,0.3);
//   if (r==1) slant_x(pic->ops,rng,0.3);
//   if (r==2) slant_y(pic->ops,rng,0.3);
//   jiggleCharacter(pic->ops,rng,12);
//   return pic;
// }


// EDfields edf(pow(10,5),24,28,6,3);
// Picture* OnlinePicture::distort() {
//   OnlinePicture* pic=new OnlinePicture(*this);
//   RNG rng;
//   int ind=rng.index(edf.edf);
//   characterED(pic->ops,edf.edf[ind]);
//   return pic;
// }
