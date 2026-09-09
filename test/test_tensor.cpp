
#include <memory>
#include <iostream>
#include <cstdlib>
#include "hiten/Tensor.h"

int main() {

    hiten::Tensor t(4, hiten::Device::Cpu);
    float* f = (float*) t.data();
    f[0] = 5.f;
    {
        hiten::Tensor t2(4, hiten::Device::Hip);
        t2.cpu();
        std::cout << "t2 is " << t2.device() << std::endl;
        auto t3 = t;
        ((float*)t3.data())[0] = 10.f;
        t3.hip();
        t3.cpu();
    }
    std::cout << "t is " << t.device() << std::endl;
    std::cout << "t[0] is " << ((float*) t.data())[0] << std::endl;
    std::cout << "End" << std::endl;
}