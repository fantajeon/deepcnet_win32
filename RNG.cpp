#define _CRT_RAND_S

#include "RNG.h"

#include <string>
#include <iostream>
#include <boost\non_type.hpp>
#include <boost\utility.hpp>
#include <boost\thread.hpp>
//#include <boost\random.hpp>
#include <boost/generator_iterator.hpp>
#include <random>


using namespace std;

//#include<sys/time.h>
static boost::mutex RNGseedGeneratorMutex;
static std::mt19937_64 RNGseedGenerator;
static  unsigned int    g_number;


RNG::RNG() 
{
	extern const int startEpoch;

	//m_pgen = static_cast<void*>(new std::mt19937());
	//std::mt19937 *pgen = static_cast<std::mt19937*>(m_pgen);

	RNGseedGeneratorMutex.lock();
	m_gen.seed(RNGseedGenerator()+startEpoch);
	RNGseedGeneratorMutex.unlock();
	// clock_gettime(CLOCK_REALTIME, &ts);
	// gen.seed(ts.tv_nsec);

	//RNGseedGeneratorMutex.lock();
	//cout<<"Lock(g_number:"<<g_number<<endl;
	//m_gen.seed(rand_s( &g_number )+rand_s( &g_number )+rand_s( &g_number ));
	//cout<<"UnLock(g_number:"<<g_number<<endl;
	//RNGseedGeneratorMutex.unlock();
}

RNG::~RNG()
{
	//delete m_pgen;
}

void RNG::seed(uint64_t s) 
{
	//std::mt19937 *pgen = static_cast<std::mt19937*>(m_pgen);
	//pgen->seed(s);
	m_gen.seed(s);
}

uint64_t RNG::gen()
{
	/*std::mt19937 *pgen = static_cast<std::mt19937*>(m_pgen);

	return (*pgen)();*/
	return m_gen();
}

uint64_t RNG::randint(uint64_t n) 
{
	RNGseedGeneratorMutex.lock();
	std::uniform_int_distribution<uint64_t> dis(0, n-1);
	uint64_t ret = dis(m_gen);
	//if( n == 60000 ) {
	//	FILE *fp = fopen("rnd.txt", "at");
	//	fprintf(fp, "%d,",ret);
	//	fclose(fp);
	//}
	RNGseedGeneratorMutex.unlock();
	return ret;
}

float RNG::uniform(float a, float b) 
{
	uint64_t k = randint(4294967297);
	return a+(b-a)*(float)k/4294967296.0f;
}

int RNG::bernoulli(float p) {
	if (uniform()<p)
		return 1;
	else
		return 0;
}


vector<int> RNG::NchooseM(int n, int m) 
{
	vector<int> ret(m);
	int ctr=m;
	for(int i=0;i<n;i++)
		if (rand()<ctr*1.0/(n-i)) ret[--ctr]=i;
	return ret;
}

