#pragma once
#include <vector>
#include <random>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h> 
#include <nanobind/stl/vector.h> 

namespace nb = nanobind;

template <typename T>
constexpr T kEps = static_cast<T>(1e-7);

// 1. Corrected core conversion helper.
template <typename T>
inline nb::ndarray<nb::numpy, T> vec2npy_generic(std::vector<T>& vec, size_t ndim, const size_t* shape_ptr) {
    // Keep ownership semantics uniform for both empty and non-empty arrays.
    // Some runtime paths (debugger hooks / array converters) are sensitive to
    // default-constructed array objects; an explicit shape + owning capsule is
    // more robust.
    auto* v_ptr = new std::vector<T>(std::move(vec));
    nb::capsule deleter(v_ptr, [](void* p) noexcept {
        delete reinterpret_cast<std::vector<T>*>(p);
    });

    // nanobind copies the values pointed to by shape_ptr inside the constructor.
    return nb::ndarray<nb::numpy, T>(v_ptr->data(), ndim, shape_ptr, deleter);
}
template <typename T>
inline nb::ndarray<nb::numpy, T> vec2npy(std::vector<T>& vec) {
    size_t s[1] = { vec.size() };
    return vec2npy_generic(vec, 1, s);
}

template <typename T>
inline nb::ndarray<nb::numpy, T> vec2npy_2d(std::vector<T>& vec, size_t d1, size_t d2) {
    size_t s[2] = { d1, d2 }; // Stack memory is fine here because it is only needed during construction.
    return vec2npy_generic(vec, 2, s);
}

template <typename T>
inline nb::ndarray<nb::numpy, T> vec2npy_3d(std::vector<T>& vec, size_t d1, size_t d2, size_t d3) {
    size_t s[3] = { d1, d2, d3 };
    return vec2npy_generic(vec, 3, s);
}

// 3. TemporalGraph remains unchanged.
template <typename ID_T, typename TIME_T>
struct TemporalGraph {
    using IntArray = nb::ndarray<ID_T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using FloatArray = nb::ndarray<TIME_T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

    IntArray indptr;
    IntArray indices;
    IntArray eid;
    FloatArray ts;

    TemporalGraph() = default;
    TemporalGraph(IntArray _indptr, IntArray _indices, IntArray _eid, FloatArray _ts)
        : indptr(_indptr), indices(_indices), eid(_eid), ts(_ts) {}

    const ID_T* indptr_ptr() const { return indptr.data(); }
    const ID_T* indices_ptr() const { return indices.data(); }
    const ID_T* eid_ptr() const { return eid.data(); }
    const TIME_T* ts_ptr() const { return ts.data(); }
};

// 4. Xorshift32
struct Xorshift32 {
    uint32_t state;
    explicit Xorshift32(uint32_t seed) : state(seed) {}
    inline uint32_t operator()() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return state;
    }
};
