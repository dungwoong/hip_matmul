#pragma once
#include "hiplass/utils.h"
#include <concepts>

namespace hiplass {
struct st_tag {};
struct gt_tag {};

struct gt {
    using tag = gt_tag;
    int dim0, dim1, dim2, dim3;

    float* data;

    HOSTDEVICE gt(float* p, int dim0, int dim1, int dim2, int dim3) : data(p), dim0(dim0), dim1(dim1), dim2(dim2), dim3(dim3) {}

    HOSTDEVICE int idx(int i0, int i1, int i2, int i3) const {
        return i3 + dim3 * (i2 + dim2 * (i1 + dim1 * i0));
    }

    HOSTDEVICE float& operator()(int i0, int i1, int i2, int i3) const {
        return data[idx(i0, i1, i2, i3)];
    }
};

template <int rows_, int cols_>
struct Layout2D {
    static constexpr int rows = rows_;
    static constexpr int cols = cols_;
    
};

template <int rows_, int cols_, int padding_>
struct st {
    using tag = st_tag;
    static constexpr int rows = rows_;
    static constexpr int cols = cols_;
    static constexpr int padding = padding_;

    float (*data)[cols + padding];

    HOSTDEVICE st(float p[rows][cols + padding]) : data(p) {}

    HOSTDEVICE constexpr int size() const {
        return rows * cols;
    }

    // HOSTDEVICE int idx(int r, int c) const {
    //     return r * cols + c;
    // }

    HOSTDEVICE float& operator()(int r, int c) {
        // return data[idx(r, c)];
        return data[r][c];
    }
};

template <typename T>
concept IsST = requires {typename T::tag; } && std::same_as<typename T::tag, st_tag>;

template <typename T>
concept IsGT = requires { typename T::tag; } && std::same_as<typename T::tag, gt_tag>;

template <int nthreads>
HOSTDEVICE void load(const IsGT auto& G, IsST auto& S, int laneId, int i0, int i1, int i2, int i3) {
    // Global -> shared, 4 elements at a time: each thread reads a float4 from
    // G (contiguous in the fastest-varying / column dimension) and unpacks it
    // into shared memory a scalar at a time (S rows are padded, so a vector
    // store there isn't reliably 16B-aligned).
    static_assert(S.cols % 4 == 0, "float4 load requires S.cols divisible by 4");
    constexpr int VCOLS = S.cols / 4;
    static_assert(nthreads % VCOLS == 0); // assume that we don't have to recalculate colOffset
    int rowOffset = laneId / VCOLS;
    int vecCol = laneId % VCOLS;
    int colOffset = vecCol * 4;
    int rowIncr = nthreads / VCOLS;

    #pragma unroll
    for (int rowIdx = rowOffset; rowIdx < S.rows; rowIdx += rowIncr) {
        float4 val = *reinterpret_cast<const float4*>(&G(i0, i1, i2 + rowIdx, i3 + colOffset));
        S(rowIdx, colOffset + 0) = val.x;
        S(rowIdx, colOffset + 1) = val.y;
        S(rowIdx, colOffset + 2) = val.z;
        S(rowIdx, colOffset + 3) = val.w;
    }
}


/*
Loads go GMEM --> Regs --> LDS. This breaks load into 2 parts so we can do them simultaneously,
if that becomes an issue
*/
template <int nthreads>
HOSTDEVICE void loadRegs(const IsGT auto& G, IsST auto& S, float* R, int laneId, int i0, int i1, int i2, int i3) {
    // Global -> registers, 4 elements at a time: each thread reads a float4
    // from G and unpacks it into R a scalar at a time (R is a caller-owned
    // stack array with no guaranteed 16B alignment).
    static_assert(S.cols % 4 == 0, "float4 load requires S.cols divisible by 4");
    constexpr int VCOLS = S.cols / 4;
    static_assert(nthreads % VCOLS == 0); // assume that we don't have to recalculate colOffset
    int rowOffset = laneId / VCOLS;
    int vecCol = laneId % VCOLS;
    int colOffset = vecCol * 4;
    int rowIncr = nthreads / VCOLS;

    for (int rowIdx = rowOffset, i = 0; rowIdx < S.rows; rowIdx += rowIncr, i += 4) {
        float4 val = *reinterpret_cast<const float4*>(&G(i0, i1, i2 + rowIdx, i3 + colOffset));
        R[i + 0] = val.x;
        R[i + 1] = val.y;
        R[i + 2] = val.z;
        R[i + 3] = val.w;
    }
}

template <int nthreads>
HOSTDEVICE void loadShared(const IsGT auto& G, IsST auto& S, float *R, int laneId, int i0, int i1, int i2, int i3) {
    // static_assert(S.size() % nthreads == 0);
    static_assert(nthreads % S.cols == 0); // assume that we don't have to recalculate colOffset
    int nTrips = S.size() / nthreads;
    int rowOffset = laneId / S.cols;
    int colOffset = laneId % S.cols;
    int rowIncr = nthreads / S.cols;

    for (int rowIdx = rowOffset, i=0; rowIdx < S.rows; rowIdx += rowIncr, i++) {
        S(rowIdx, colOffset) = R[i];
    }
}

template <int nthreads>
HOSTDEVICE void store(IsST auto& S, const IsGT auto& G, int laneId, int i0, int i1, int i2, int i3) {
    // static_assert(S.size() % nthreads == 0);
    static_assert(nthreads % S.cols == 0); // assume that we don't have to recalculate colOffset
    int nTrips = S.size() / nthreads;
    int rowOffset = laneId / S.cols;
    int colOffset = laneId % S.cols;
    int rowIncr = nthreads / S.cols;

    for (int rowIdx = rowOffset; rowIdx < S.rows; rowIdx += rowIncr) {
        G(i0, i1, i2 + rowIdx, i3 + colOffset) = S(rowIdx, colOffset);
    }
}
}