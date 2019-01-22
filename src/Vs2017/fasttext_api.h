#pragma once

#ifdef FASTTEXT_EXPORTS
	#define FT_API(RetType) EXTERN_C __declspec(dllexport) RetType __stdcall
#else 
	#define FT_API(RetType) EXTERN_C __declspec(dllimport) RetType __stdcall
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
FT_API(int) GetLabels(void* hPtr, char*** labels);
FT_API(void) TrainSupervised(void* hPtr, const char* input, const char* output, TrainingArgs trainArgs, const char* labelPrefix);
FT_API(void) DestroyString(char* string);
FT_API(void) DestroyStrings(char** strings, int cnt);
FT_API(float) PredictSingle(void* hPtr, const char* input, char** predicted);
FT_API(int) PredictMultiple(void* hPtr, const char* input, char*** predictedLabels, float* predictedProbabilities, int n);

FT_API(int) GetSentenceVector(void* hPtr, const char* input, float** vector);
FT_API(void) DestroyVector(float* vector);