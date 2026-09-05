#pragma once
#include <array>
#include <cstddef>
#include "hiplass/utils.h"

/*
CUTLASS has nested IntTuple. 

Vals * ... becomes nested V2 * (V1 * (V0))
Vals... just becomes V0, V1, V2
(As * Bs)... becomes (A0 * B0), (A1 * B1), ...
*/
namespace hiplass {
template <int... Vals>
struct IntTuple {
    static constexpr std::size_t rank = sizeof...(Vals);

    template <std::size_t I>
    static constexpr int get() {
        static_assert(I < rank, "index out of bounds");
        constexpr int arr[] = {Vals...};
        return arr[I];
    }

    static constexpr int size() {
        return (Vals * ... * 1);
    }

    HOSTDEVICE static std::array<int, rank> coord(int idx) {
        constexpr int arr[] = {Vals...};
        std::array<int, rank> c{};
        #pragma unroll
        for (std::size_t i = 0; i < rank; ++i) {
            c[i] = idx % arr[i];
            idx /= arr[i];
        }
        return c;
    }
};

template <typename A, typename B>
struct ElementwiseMultiply;

template <int... As, int... Bs>
struct ElementwiseMultiply<IntTuple<As...>, IntTuple<Bs...>> {
    static_assert(sizeof...(As) == sizeof...(Bs), "rank mismatch");
    using type = IntTuple<(As * Bs)...>;
};

template <typename A, typename B>
using elementwise_multiply_t = typename ElementwiseMultiply<A, B>::type;


/*
Threads handle a tile, and threads are laid out in waves
waves serially loop over a layout,
and then multiple waves in the WorkGroup are parallelized
*/
template<typename THREAD_TILE, typename WAVE_LAYOUT, typename WAVE_SERIAL_LAYOUT, typename WAVE_PARALLEL_LAYOUT>
struct HierarchicalLoad {
    using WAVE_TILE = elementwise_multiply_t<THREAD_TILE, WAVE_LAYOUT>;
    using WAVE_S_TILE = elementwise_multiply_t<WAVE_TILE, WAVE_SERIAL_LAYOUT>;
    using WAVE_P_TILE = elementwise_multiply_t<WAVE_S_TILE, WAVE_PARALLEL_LAYOUT>;

    HOSTDEVICE static int rowA(int waveid, int laneid, int m_serial_idx, int m_thread_idx) {
        int waveM = WAVE_PARALLEL_LAYOUT::coord(waveid)[0];   // which Wave Tile row-block
        int laneM = WAVE_LAYOUT::coord(laneid)[0];            // which thread-tile row within the wave

        return waveM      * WAVE_S_TILE::template get<0>()
            + m_serial_idx * WAVE_TILE::template get<0>()
            + laneM      * THREAD_TILE::template get<0>()
            + m_thread_idx;
    }

    HOSTDEVICE static int rowB(int waveid, int laneid, int n_serial_idx, int n_thread_idx) {
        int waveN = WAVE_PARALLEL_LAYOUT::coord(waveid)[1];
        int laneN = WAVE_LAYOUT::coord(laneid)[1];

        return waveN      * WAVE_S_TILE::template get<1>()
            + n_serial_idx * WAVE_TILE::template get<1>()
            + laneN      * THREAD_TILE::template get<1>()
            + n_thread_idx;
    }
};

/*
Same hierarchy as HierarchicalLoad, but for writing C back out: since C is
[M x N], both dimensions matter together, so this gives you the (row, col)
pair for one register slot in one call instead of rowA/rowB separately.
*/
template<typename THREAD_TILE, typename WAVE_LAYOUT, typename WAVE_SERIAL_LAYOUT, typename WAVE_PARALLEL_LAYOUT>
struct HierarchicalStore {
    using WAVE_TILE = elementwise_multiply_t<THREAD_TILE, WAVE_LAYOUT>;
    using WAVE_S_TILE = elementwise_multiply_t<WAVE_TILE, WAVE_SERIAL_LAYOUT>;
    using WAVE_P_TILE = elementwise_multiply_t<WAVE_S_TILE, WAVE_PARALLEL_LAYOUT>;

    HOSTDEVICE static std::array<int, 2> crds(int waveid, int laneid, int wave_s_idx, int thread_s_idx) {
        std::array<int, 2> WavePIdx = WAVE_PARALLEL_LAYOUT::coord(waveid);
        std::array<int, 2> WaveSIdx = WAVE_SERIAL_LAYOUT::coord(wave_s_idx);
        std::array<int, 2> ThreadPIdx = WAVE_LAYOUT::coord(laneid);
        std::array<int, 2> ThreadSIdx = THREAD_TILE::coord(thread_s_idx);

        int row = (
            (WavePIdx[0] * WAVE_S_TILE::template get<0>()) +
            (WaveSIdx[0] * WAVE_TILE::template get<0>()) +
            (ThreadPIdx[0] * THREAD_TILE::template get<0>()) +
            (ThreadSIdx[0])
        );
        int col = (
            (WavePIdx[1] * WAVE_S_TILE::template get<1>()) +
            (WaveSIdx[1] * WAVE_TILE::template get<1>()) +
            (ThreadPIdx[1] * THREAD_TILE::template get<1>()) +
            (ThreadSIdx[1])
        );
        return {row, col};
    }
};
} // namespace hiplass