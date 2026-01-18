// Minimal c10 complex compatibility header for vendored kernels.
#pragma once

#include <cuda_runtime.h>

namespace c10 {

template<typename T>
struct complex {
    T real_;
    T imag_;

    __host__ __device__ constexpr complex(T r = T{}, T i = T{}) : real_(r), imag_(i) {}

    __host__ __device__ constexpr complex(const complex&) = default;
    __host__ __device__ constexpr complex& operator=(const complex&) = default;
};

template<typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real_ + b.real_, a.imag_ + b.imag_);
}

template<typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real_ - b.real_, a.imag_ - b.imag_);
}

template<typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real_ * b.real_ - a.imag_ * b.imag_,
                      a.real_ * b.imag_ + a.imag_ * b.real_);
}

template<typename T>
__host__ __device__ inline complex<T> operator*(T scalar, const complex<T>& a) {
    return complex<T>(scalar * a.real_, scalar * a.imag_);
}

template<typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& a, T scalar) {
    return complex<T>(scalar * a.real_, scalar * a.imag_);
}

template<typename T>
__host__ __device__ inline complex<T> operator/(const complex<T>& a, T scalar) {
    return complex<T>(a.real_ / scalar, a.imag_ / scalar);
}

} // namespace c10

namespace std {
// Provide std::conj overload for c10::complex used in kernels.
template<typename T>
__host__ __device__ inline c10::complex<T> conj(const c10::complex<T>& z) {
    return c10::complex<T>(z.real_, -z.imag_);
}
} // namespace std
