#pragma once

#include "CNN.h"
#include <boost/thread.hpp>



class BatchProducer {
	boost::mutex batchDequeueMutex;
	boost::mutex batchCounterMutex;
public:
	boost::thread_group workers;
	vector<Picture*>* dataset;
	deque<SparseCnnInterface*> dq;
	int batchCounter; //# batches started to be created.
	int batchCounter2;//# batches "popped" from the deque

	SparseCnnInterface* pop() {
		while (dq.size()==0)
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		batchDequeueMutex.lock();
		SparseCnnInterface* batch=dq.front();
		dq.pop_front();
		batchCounter2++;
		batchDequeueMutex.unlock();
		return batch;
	}
	void push(SparseCnnInterface* batch) {
		while (dq.size()>2000)
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		batchDequeueMutex.lock();
		dq.push_back(batch);
		batchDequeueMutex.unlock();
	}
	virtual bool workToDo () = 0;
	virtual void batchProducerThread() = 0;
	BatchProducer (vector<Picture*>* dataset) : dataset(dataset), batchCounter(0), batchCounter2(0) {}
	void start(int nThreads = 8) {
		for (int i=0; i<nThreads; i++){
			workers.add_thread(new boost::thread(boost::bind(&BatchProducer::batchProducerThread,this)));
		}
	}
	int getBatchCounter () {
		batchCounterMutex.lock();
		int c = batchCounter++;
		batchCounterMutex.unlock();
		return c;
	}
	void join() {
		workers.join_all();
	}
};

class RandomizedTrainingBatchProducer : public BatchProducer {
public:
	boost::mutex batchIndexMutex;
	RNG rng;
	int next;
	vector<int>		indeces;

	void shuffle()
	{
		for(int s=0; s < 10; ++s) {
			//cout<<"shuffle batches...."<<endl;
			for(int i=0; i < indeces.size(); ++i) {
				int j = (int)rng.randint(indeces.size());
				int tmp = indeces[i];
				indeces[i] = indeces[j];
				indeces[j] = tmp;
			}
		}
		next = 0;
	}

	void batchProducerThread() 
	{
		while (true) {
			int c = getBatchCounter();
			SparseCnnInterface* batch = new SparseCnnInterface(TRAINBATCH);
			for (int i=0;i<trainingBatchSize;i++) {
				batchIndexMutex.lock();
				int idx = indeces[next++];
				if( next >= indeces.size() ) {
					batch->batchIsEof = true;
					shuffle();
				}
				batchIndexMutex.unlock();
				//cout<<"idx:"<<idx<<"ptr:"<<(unsigned int*)(void*)&rng<<endl;
				Picture* pic=dataset->at(idx)->distort();
				pic->codifyInputData(*batch);
				delete pic;
			}
			push(batch);
		}
	}
	bool workToDo () {
		return true;
	}
	RandomizedTrainingBatchProducer (vector<Picture*>* dataset) : BatchProducer(dataset) {
		indeces.resize( dataset->size() );
		for(int i=0; i < dataset->size(); ++i) {
			indeces[i] = i;
		}
		shuffle();
		start(2);
	}
};


class TestsetBatchProducer : public BatchProducer {
public:
	void batchProducerThread() {
		while (true) {
			int c = getBatchCounter();
			if (c*trainingBatchSize>=dataset->size())
				break;
			SparseCnnInterface* batch = new SparseCnnInterface(TESTBATCH);
			for (int i=c*trainingBatchSize;
				i<min((c+1)*trainingBatchSize,(int)(dataset->size()));
				i++)
				dataset->at(i)->codifyInputData(*batch);
			push(batch);
		}
	}
	bool workToDo () {
		return batchCounter2*trainingBatchSize < dataset->size() || dq.size()>0;
	}
	TestsetBatchProducer (vector<Picture*>* dataset) : BatchProducer(dataset) {
		start();
	}
};


void train_test(int nTrain, int nTest) ;
