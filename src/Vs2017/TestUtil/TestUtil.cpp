// TestUtil.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "../fasttext_api.h"


int main()
{
	auto hPtr = CreateFastText();
	
	/*TrainingArgs args;
	args.Epochs = 25;
	args.LearningRate = 1.0;
	args.WordNGrams = 3;
	args.Verbose = 2;
	
	TrainSupervised(hPtr, "C:\\_Models\\cooking.train.txt", "C:\\_Models\\fasttext", args, nullptr);
	*/
	
	
	LoadModel(hPtr, "C:\\_Models\\fasttext.bin");

	char** labels;
	int nLabels = GetLabels(hPtr, &labels);
	DestroyStrings(labels, nLabels);


	char* buff;
	
	float prob = PredictSingle(hPtr, "what is the difference between a new york strip and a bone-in new york cut sirloin ?", &buff);
		
	char** buffers;
	float* probs = new float[5];

	int cnt = PredictMultiple(hPtr,"what is the difference between a new york strip and a bone-in new york cut sirloin ?", &buffers, probs, 5);

	DestroyString(buff);
	DestroyStrings(buffers, 5);

	DestroyFastText(hPtr);
    return 0;
}

