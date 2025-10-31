#ifndef UTILS
#define UTILS

#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#define WARP_SIZE 32
// 向上取整
#define CEIL(a, b) (a + b - 1) / b

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// 定义 AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16 宏
#ifndef AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16
#define AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(TYPE, NAME, ...) \
  [&] { \
    const auto& the_type = TYPE; \
    at::ScalarType _st = ::detail::scalar_type(the_type); \
    switch (_st) { \
      case at::ScalarType::Float: { \
        using scalar_t = float; \
        return __VA_ARGS__(); \
      } \
      case at::ScalarType::Double: { \
        using scalar_t = double; \
        return __VA_ARGS__(); \
      } \
      case at::ScalarType::Half: { \
        using scalar_t = at::Half; \
        return __VA_ARGS__(); \
      } \
      case at::ScalarType::BFloat16: { \
        using scalar_t = at::BFloat16; \
        return __VA_ARGS__(); \
      } \
      default: \
        AT_ERROR(#NAME, " not implemented for ", toString(_st)); \
    } \
  }()
#endif

#endif // UTILS