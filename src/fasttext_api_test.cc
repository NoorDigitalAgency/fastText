#include <iostream>
#include "fasttext_api.h"

int main(int argc, char** argv) {
    auto fastText = CreateFastText();

    DestroyFastText(fastText);
    return 0;
}
