// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <string>
#include <sstream>
#include "../fasttext.h"
#include "fasttext_api.h"

using namespace fasttext;

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

FT_API(void*) CreateFastText()
{
	auto result = new FastText();
	return result;
}

FT_API(void) LoadModel(void* hPtr, const char* path)
{
	auto fastText = static_cast<FastText*>(hPtr);
	fastText->loadModel(path);
}

FT_API(void) DestroyFastText(void* hPtr)
{
	delete static_cast<FastText*>(hPtr);
}

int GetMaxLabelLenght(void* hPtr)
{
	auto fastText = static_cast<FastText*>(hPtr);
	auto dict = fastText->getDictionary();
	int numLabels = dict->nlabels();
	int maxLen = 0;

	for (int i = 0; i < numLabels; ++i)
	{
		auto label = dict->getLabel(i);
		if (label.length() > maxLen)
		{
			maxLen = label.length();
		}
	}

	return maxLen;
}

FT_API(void) TrainSupervised(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, const char* labelPrefix)
{
	auto fastText = static_cast<FastText*>(hPtr);
	auto args = Args();
	args.verbose = trainArgs.Verbose;
	args.input = std::string(input);
	args.output = std::string(output);
	args.model = model_name::sup;
    args.loss = loss_name::softmax;
    args.minCount = 1;
    args.minn = trainArgs.MinCharNGrams;
    args.maxn = trainArgs.MaxCharNGrams;
    args.lr = trainArgs.LearningRate;
	args.wordNgrams = trainArgs.WordNGrams;
	args.epoch = trainArgs.Epochs;

	if (labelPrefix != nullptr)
	{
		args.label = std::string(labelPrefix);
	}

	fastText->train(args);
	fastText->saveModel();
	fastText->saveVectors();
}

FT_API(float) PredictSingle(void* hPtr, const char* input, char* predicted)
{
	auto fastText = static_cast<FastText*>(hPtr);
	std::vector<std::pair<real,std::string>> predictions;
	std::istringstream inStream(input);

	fastText->predict(inStream, 1, predictions, 0);

	if (predictions.size() == 0)
	{
		return 0;
	}

	auto len = predictions[0].second.length();
	predictions[0].second.copy(predicted, len);
	predicted[len] = '\0';
	
	return std::exp(predictions[0].first);
}