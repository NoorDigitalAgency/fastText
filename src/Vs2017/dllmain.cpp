// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <string>
#include "../fasttext.h"

using namespace fasttext;

#define FT_API(RetType) __declspec(dllexport) RetType __stdcall

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

FT_API(void) DestroyFastText(void* hPtr)
{
	delete static_cast<FastText*>(hPtr);
}

FT_API(void) TrainSupervised(void* hPtr, const char* input, const char* output)
{
	auto fastText = static_cast<FastText*>(hPtr);
	auto args = Args();
	args.input = std::string(input);
	args.output = std::string(output);
	args.model = model_name::sup;
    args.loss = loss_name::softmax;
    args.minCount = 1;
    args.minn = 0;
    args.maxn = 0;
    args.lr = 0.1;

	fastText->train(args);
	fastText->saveModel();
	fastText->saveVectors();
}