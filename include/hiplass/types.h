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

template <int rows_, int cols_>
struct st {
    using tag = st_tag;
    static constexpr int rows = rows_;
    static constexpr int cols = cols_;

    float* data;

    HOSTDEVICE st(float* p) : data(p) {}

    HOSTDEVICE constexpr int size() const {
        return rows * cols;
    }

    HOSTDEVICE int idx(int r, int c) const {
        return r * cols + c;
    }

    HOSTDEVICE float& operator()(int r, int c) const {
        return data[idx(r, c)];
    }
};

template <typename T>
concept IsST = requires {typename T::tag; } && std::same_as<typename T::tag, st_tag>;

template <typename T>
concept IsGT = requires { typename T::tag; } && std::same_as<typename T::tag, gt_tag>;

template <int nthreads>
HOSTDEVICE void load(const IsGT auto& G, IsST auto& S, int laneId, int i0, int i1, int i2, int i3) {
    // static_assert(S.size() % nthreads == 0);
    static_assert(nthreads % S.cols == 0); // assume that we don't have to recalculate colOffset
    int nTrips = S.size() / nthreads;
    int rowOffset = laneId / S.cols;
    int colOffset = laneId % S.cols;
    int rowIncr = nthreads / S.cols;

    for (int rowIdx = rowOffset; rowIdx < S.rows; rowIdx += rowIncr) {
        S(rowIdx, colOffset) = G(i0, i1, i2 + rowIdx, i3 + colOffset);
    }
}
}