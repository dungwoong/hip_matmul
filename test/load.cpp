#include "hiplass/hiplass.h"
#include <cstdlib>
#include <iostream>

// g++ -std=c++20 -I include test/load.cpp -o test/load

int main() {
    using tup1 = hiplass::IntTuple<1, 2>;
    using tup2 = hiplass::IntTuple<3, 4>;
    using tup3 = hiplass::elementwise_multiply_t<tup1, tup2>;

    std::cout << "(" << tup3::get<0>() << ", " << tup3::get<1>() << ")" << std::endl;

    constexpr int nthreads = 2 * 8;
    float* gArr = static_cast<float*>(malloc(32 * 16 * sizeof(float)));
    float* sArr = static_cast<float*>(malloc(16 * 8 * sizeof(float)));

    hiplass::gt global_tensor(gArr, 1, 1, 32, 16);
    hiplass::st<16, 8> shared_tensor(sArr);

    for (int i = 0; i < 32 * 16; i++) {
        gArr[i] = (float)i;
    }

    for (int threadIdx = 0; threadIdx < nthreads; threadIdx++) {
        hiplass::load<nthreads>(global_tensor, shared_tensor, threadIdx, 0, 0, 16, 8);
    }

    // should be 264, 265... 280 281 ...
    for (int i = 0; i < 16; i++) {
        float* base = sArr + (i * 8);
        for (int j = 0; j < 8; j++) {
            std::cout << base[j] << " ";
        }
        std::cout << std::endl;
    }

    free(gArr);
    free(sArr);

}