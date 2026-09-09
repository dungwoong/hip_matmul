#pragma once
#include <memory>
#include <vector>
#include <string>

namespace hiten {

enum class DType {
    Float32,
};

enum class Device {
    Hip,
    Cpu
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
    }
    return 0;
}

inline std::string device_str(Device d) {
    switch (d) {
        case Device::Hip: return "hip";
        case Device::Cpu: return "cpu";
    }
    return "unknown";
}

class Storage {
    friend class Tensor;
public:
    Storage(int nbytes, Device device);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& other) noexcept 
        : data_(other.data_), nbytes_(other.nbytes_), device_(other.device_) {
        other.data_ = nullptr;
    }
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            free_resources();
            data_ = other.data_;
            nbytes_ = other.nbytes_;
            device_ = other.device_;
            other.data_ = nullptr;
        }
        return *this;
    }

    void* data() { return data_; }
    void cpu();
    void hip();
    void copy_values(void* ptr, int bytes);
    void free_resources();
private:
    void* data_;
    int nbytes_;
    Device device_;
};

class Tensor {
public:
    Tensor(int nbytes, Device device);

    ~Tensor() = default;

    void cpu() {storage_.get()->cpu();}

    void hip() {storage_.get()->hip();}

    void from_numpy(/*???*/) {}

    std::string device() {return device_str(storage_.get()->device_);}

    Tensor(const Tensor& t) : storage_(t.storage_) {}

    Tensor& operator=(const Tensor& t) {
        storage_ = t.storage_;
        return *this;
    }

    Tensor(const Tensor&& t) : storage_(t.storage_) {}

    Tensor& operator=(const Tensor&& t) {
        storage_ = t.storage_;
        return *this;
    }

    void* data() {return storage_.get()->data_;} // debug for now


private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    DType dtype_;
};
}