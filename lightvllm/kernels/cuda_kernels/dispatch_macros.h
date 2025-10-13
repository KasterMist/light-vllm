#ifndef DISPATCH_MACROS_H
#define DISPATCH_MACROS_H

#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

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

#endif // DISPATCH_MACROS_H