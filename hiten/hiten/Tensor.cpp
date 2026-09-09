#include "hiten/Tensor.h"
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>


namespace hiten {
Storage::Storage(int nbytes, hiten::Device device) : nbytes_(nbytes), device_(device) {
    std::cout << "Adding Tensor" << std::endl;
    switch (device_) {
        case Device::Cpu: {data_ = malloc(nbytes_); break;}
        case Device::Hip: {hipMalloc(&data_, nbytes_); break;}
    }
}

Storage::~Storage() {
    free_resources();
    std::cout << "Destroying Tensor" << std::endl;
}

void Storage::cpu() {
    if (device_ == Device::Hip) {
        void* new_ptr = malloc(nbytes_);
        hipMemcpy(new_ptr, static_cast<const void *>(data_), nbytes_, hipMemcpyDeviceToHost);
        hipFree(data_);
        data_ = new_ptr;
        device_ = Device::Cpu;
    }
}

void Storage::hip() {
    if (device_ == Device::Cpu) {
        void* new_ptr;
        hipMalloc(&new_ptr, nbytes_);
        hipMemcpy(new_ptr, static_cast<const void *>(data_), nbytes_, hipMemcpyHostToDevice);
        free(data_);
        data_ = new_ptr;
        device_ = Device::Hip;
    }
}

void Storage::free_resources() {
    switch (device_) {
        case Device::Hip: {hipFree(data_); break;}
        case Device::Cpu: {free(data_); break;}
    }
}

Tensor::Tensor(int nbytes, Device device) : storage_(std::make_shared<hiten::Storage>(nbytes, device)) {}
}
