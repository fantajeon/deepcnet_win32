#ifndef __DEEPCNET_RNG_H__
#define __DEEPCNET_RNG_H__

#pragma once

#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>
#include <random>


using namespace std;

class RNG {
	//  timespec ts;
public:
	std::mt19937_64		m_gen;
	RNG();
	virtual ~RNG();

	void seed(uint64_t s);
	uint64_t gen();

	uint64_t randint(uint64_t n);

	float uniform(float a=0, float b=1) ;

	int bernoulli(float p);

	template <typename T>
	int index(vector<T> &v) 
	{
		if (v.size()==0) cout << "RNG::index called for empty vector!";
		return gen()%v.size();
	}

	vector<int> NchooseM(int n, int m) ;
};


#endif // #define __DEEPCNET_RNG_H__