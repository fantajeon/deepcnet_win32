#include "run.h"

extern ComputationalCNN *g_pccnn;

vector<double> linspace(double min, double max, int n)
{
	vector<double> result;
	// vector iterator
	int iterator = 0;

	for (int i = 0; i <= n-2; i++)
	{
		double temp = min + i*(max-min)/(floor((double)n) - 1);
		result.insert(result.begin() + iterator, temp);
		iterator += 1;
	}

	//iterator += 1;

	result.insert(result.begin() + iterator, max);
	return result;
}



void test(bool verbose) 
{
	TestsetBatchProducer bp(&(*g_ptestCharacters));
	int total=0;
	int wrong=0;
	while(bp.workToDo()) {
		SparseCnnInterface* batch=bp.pop();
		g_pccnn->processBatch(batch);
		wrong+=batch->nMistakes;
		total+=batch->batchSize;
		delete batch;
		if (verbose)
			cout << "Epoch:"<<g_pccnn->nn.epoch<<"Test set size: " << total << " Test error: " << wrong*100.0/total << "%" <<endl;
	}
	bp.join();
	if (!verbose)
		cout << "Epoch:"<<g_pccnn->nn.epoch<<" Test set size: " << total << " Test error: " << wrong*100.0/total << "%" <<endl;

	g_pccnn->nn.saveWeights();

	FILE *fplog = fopen("test_log.txt", "at");
	fprintf(fplog, "Epoch: %d Test set size: %d Test error: %f%%\n", g_pccnn->nn.epoch, total, wrong*100.0/total);
	fclose(fplog);
}


void train(int nTrain) 
{
	RandomizedTrainingBatchProducer bp(&(*g_ptrainCharacters));
	int mistakes=0;
	int total=0;
	for (int epoch=startEpoch+1; bp.workToDo(); epoch++) {
		SparseCnnInterface* batch=bp.pop();
		__cudaCheckError(__FILE__, __LINE__);
		g_pccnn->processBatch(batch);
		__cudaCheckError(__FILE__, __LINE__);
		mistakes+=batch->nMistakes;
		total+=trainingBatchSize;
		delete batch;
		if (epoch%nTrain==0) {
			cout << "Training batch: " << epoch << " " << "Mistakes: " << mistakes*100.0/total << "%" << endl;
			mistakes=0;
			total=0;
		}
	}
}

float zero_epsilon()
{
	float machEps = 1.0f;

	printf( "current Epsilon, 0 + current Epsilon\n" );
	do {
		machEps /= 2.0f;
		// If next epsilon yields 1, then break, because current
		// epsilon is the machine epsilon.
	}
	while ((float)(0.0 + (machEps/2.0)) != 0.0);

	printf( "%G\t%.20f\n", machEps, (0.0f + machEps) );
	return machEps;
}

void train_test(int nTrain, int nTest) 
{
	float loweset_lr = zero_epsilon();

	RNG rng;
	//__cudaCheckError(__FILE__, __LINE__);
	RandomizedTrainingBatchProducer bp(&(*g_ptrainCharacters));
	uint64_t mistakes=0;
	uint64_t total=0;
	double per_trend = 1.0;
	double per_one = 0.0;
	double per_batch = 0.0;
	const int num_subbatch = 1;
	const double trend_scale = 1.05;
	char sztmp[4096];
	bool bIsEof = false;
	float init_descre_rate = 0.99;
	float descre_rate = init_descre_rate;

	FILE *fplog = fopen("test_log.txt", "at");
	fprintf(fplog, "===================================\n");
	fclose(fplog);

	int countinue_phase = 0;
	double prev_percent = 100.0;
	vector<double> learning_rates = linspace(1e-15,1e-1,100);

	vector<SparseCnnInterface*>		worstbatch;

	//g_pccnn->nn.learningRate = learning_rates[99];
	for (int epoch=startEpoch+1; bp.workToDo(); epoch++) {
		vector<SparseCnnInterface*> subbatch;
		__cudaCheckError(__FILE__, __LINE__);

		bIsEof = false;
		for(int s=0; s < num_subbatch; ++s) {
			SparseCnnInterface* batch=bp.pop();
			subbatch.push_back(batch);

			if( batch->batchIsEof ) {
				bIsEof = true;
			}
		}
		__cudaCheckError(__FILE__, __LINE__);

		uint64_t mistakes_old, total_old;
		int witer_counter = 0;

		mistakes_old = mistakes;
		total_old = total;
		double lr = g_pccnn->nn.learningRate;
		do {
			int subnMistakes = 0;
			int subtotal = 0;

			// backtracking global percent
			mistakes = mistakes_old;
			total = total_old;


			for(int s=0; s < subbatch.size(); ++s) {
				SparseCnnInterface* batch = subbatch[s];
				batch->nMistakes = 0;
				g_pccnn->processBatch(batch);

				mistakes += batch->nMistakes;
				total += trainingBatchSize;

				subnMistakes += batch->nMistakes;
				subtotal += trainingBatchSize;

				per_batch = batch->nMistakes/(double)trainingBatchSize;
				per_one = subnMistakes/(double)subtotal;
				per_trend = mistakes/(double)total;


				_snprintf(sztmp,sizeof(sztmp), "Training batch[S]:%d(sub:%d/w:%d) M:%.2f%%(%.2f%%)/%.6f%% lr:%e",
					epoch, s, witer_counter, per_one*100.0, per_batch*100.0, per_trend*100.0, 
					g_pccnn->nn.learningRate);

				cout << sztmp <<endl;
			}

			g_pccnn->applyDerivatives();
			__cudaCheckError(__FILE__, __LINE__);

			//g_pccnn->nn.learningRate *= 0.1;
			++witer_counter;
		} while(per_one*0.99 > per_trend && witer_counter < 0 );
		g_pccnn->nn.learningRate = lr;

		for(int o=0; o < subbatch.size(); ++o) {
			SparseCnnInterface* batch = subbatch[o];
			double per_batch = batch->nMistakes/(double)trainingBatchSize;
#if 0
			if( per_batch > per_trend*trend_scale ) {
				if( rng.uniform(0.0f,1.0f) < 0.1f ) {
					worstbatch.push_back(batch);
				}
				if( worstbatch.size() == num_subbatch ) {
					int witer_counter = 0;

					//double back_learning_rate = g_pccnn->nn.learningRate;
					//g_pccnn->nn.learningRate = back_learning_rate*10.0;
					do {
						int subnMistakes = 0;
						int subtotal = 0;

						for(int s=0; s < worstbatch.size(); ++s) {
							SparseCnnInterface* batch = worstbatch[s];
							batch->nMistakes = 0;
							g_pccnn->processBatch(batch);

							subnMistakes += batch->nMistakes;
							subtotal += trainingBatchSize;

							per_one = subnMistakes/(double)subtotal;
							per_batch = batch->nMistakes/(double)trainingBatchSize;

							_snprintf(sztmp,sizeof(sztmp), "=Training batch[H]: %d(sub:%d/w:%d) M: %.2f%%(%.2f%%)/%.2f%% lr: %0.2e",
								epoch, s, witer_counter, per_one*100.0, per_batch*100.0, per_trend*100.0, g_pccnn->nn.learningRate);

							cout << sztmp <<endl;
						}

						g_pccnn->applyDerivatives();
						__cudaCheckError(__FILE__, __LINE__);
						++witer_counter;
					} while(per_one > per_trend && witer_counter < 5);

					//g_pccnn->nn.learningRate = back_learning_rate;
					vector<SparseCnnInterface*>	worstbatch2;
					for(int w=0; w < worstbatch.size(); ++w) {
						SparseCnnInterface* batch = worstbatch[w];
						per_batch = batch->nMistakes/(double)trainingBatchSize;
						if( per_batch > per_trend*trend_scale && rng.uniform(0.0f,1.0f) < 0.1f ) {
							worstbatch2.push_back(batch);
						}
						else 
							delete batch;
					}
					worstbatch.clear();
					worstbatch = worstbatch2;
					worstbatch2.clear();
				}
			}
			else
				delete batch;
#else
			delete batch;
#endif
		}

		//cout<<"Hard traning batch size: "<<worstbatch.size()<<endl;
		subbatch.clear();

		/*float decay = (std::max<float>)( numeric_limits<float>::epsilon(), learningRateDecayRate );
		if( epoch%100 == 0 ) {
		g_pccnn->nn.learningRate = (std::max<float>)(1e-13f,g_pccnn->nn.learningRate-decay);
		}*/

		//if (epoch%10==0)
		//cout << "Training batch: " << epoch << " " << "Batch Mistakes: " << per_one*100.0 << "%("<<per_trend*100.0<< "%) learning rate: "<<g_pccnn->nn.learningRate<<endl;
		//cout << "\rTraining batch: " << epoch << " " << "Mistakes: " << mistakes*100.0/total << "%       " << flush;
		/*if (epoch%nTrain==0) {
		mistakes=0;
		total=0;
		}*/

		if(epoch%decreasing_check_epoch==0) {
			/*int low_idx = (int)floor(per_trend);
			float d = per_trend - (double)low_idx;
			if( low_idx >= 99 ) {
				g_pccnn->nn.learningRate = (std::min<float>)(g_pccnn->nn.learningRate,learning_rates[99]);
			}
			else {
				g_pccnn->nn.learningRate = (std::min<float>)(g_pccnn->nn.learningRate,(1.0-d)*learning_rates[low_idx] + d*learning_rates[low_idx+1]);
			}*/

			double r = (prev_percent<per_trend) ? prev_percent/per_trend : per_trend/prev_percent;
			if( per_trend > prev_percent ) {
				float prev_learningRate = g_pccnn->nn.learningRate;
				g_pccnn->nn.learningRate = (std::max<float>)(g_pccnn->nn.learningRate*descre_rate,loweset_lr);
				_snprintf(sztmp, sizeof(sztmp), "derease learning rate: %e->%e(decresing rate:%e)", prev_learningRate, g_pccnn->nn.learningRate,descre_rate);
				cout<<sztmp<<endl;

				descre_rate *= 0.5;
			}
			else {
				descre_rate = init_descre_rate;
			}
			prev_percent = (std::min<double>)(prev_percent,per_trend);
		}
		if(bIsEof) {
			cout<<"MiniBatch-EOF"<<endl;
			mistakes=0;
			total=0;
		}

		//if (epoch%500==0) {
		if(bIsEof) {
			test(false);
		}
	}
}
