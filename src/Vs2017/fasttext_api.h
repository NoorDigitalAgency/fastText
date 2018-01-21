#pragma once

#ifdef FASTTEXT_EXPORTS
	#define FT_API(RetType) EXTERN_C __declspec(dllexport) RetType __cdecl
#else 
	#define FT_API(RetType) EXTERN_C __declspec(dllimport) RetType __cdecl
#endif

#pragma pack(push, 1)
typedef struct TrainingArgs
{
	int Epochs = 5;
	double LearningRate = 0.1;
	int WordNGrams = 1;
	int MinCharNGrams = 0;
	int MaxCharNGrams = 0;
	int Verbose = 0;
} TrainingArgs;
#pragma pack(pop)

FT_API(void*) CreateFastText();
FT_API(void) LoadModel(void* hPtr, const char* path);
FT_API(void) DestroyFastText(void* hPtr);
FT_API(int) GetMaxLabelLenght(void* hPtr);
FT_API(void) TrainSupervised(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs);
FT_API(float) PredictSingle(void* hPtr, const char* input, char* predicted);