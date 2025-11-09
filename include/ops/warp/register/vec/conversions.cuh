/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Copies data from one register vector to another.
 *
 * @tparam RV1 The type of the destination register vector.
 * @tparam RV2 The type of the source register vector.
 * @param dst[out] The destination register vector.
 * @param src[in] The source register vector to copy from.
 */
#ifdef KITTENS_CDNA4
template<ducks::rv::all RV1, ducks::rv::all RV2>
__device__ static inline void copy(RV1 &dst, const RV2 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;

    using D1_1 = base_types::packing<D1>::unpacked_type;
    using D1_2 = base_types::packing<D1_1>::packed_type;

    using D2_1 = base_types::packing<D2>::unpacked_type;
    using D2_2 = base_types::packing<D2_1>::packed_type;

    static_assert(!(std::is_same_v<D1_1, fp8e4m3> ^ std::is_same_v<D2_1, fp8e4m3>),
                  "If either D1_1 or D2_1 is fp8e4m3, both must be fp8e4m3.");

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) { // just a simple copy / typecast
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
            }
        }
    }
    else { // Inner dimensions are not the same, this is really a layout conversion.
        int laneid = kittens::laneid();
        if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && std::is_same_v<typename RV2::layout, align_l>) { // align -> ortho layout
            const int give = (laneid % 16) / 2;
            const int x_or_y = (laneid % 16) % 2;
            const int take = (((laneid % 16) / 8) * 32) + ((laneid / 16) * 8) + (laneid % 8);
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                D2_1 val = x_or_y ? src[i][give].y : src[i][give].x;
                dst[i][0] = __float2bfloat16(__shfl(__bfloat162float(val), take));
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, align_l> && std::is_same_v<typename RV2::layout, ortho_l>) { // ortho -> align layout
            const int lane_offset = 32 * ((laneid % 32) / 16) + 8 * (laneid / 32);
            const int inner_lane = laneid % 16;
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                
                #pragma unroll
                for(int j = 0; j < 16; j++) {
                    const int inner_dim = (inner_lane ^ j);
                    const int take = lane_offset + (inner_dim % 8) + 16 * (inner_dim / 8);
                    const int x_or_y = take % 2;

                    D2_1 val = __float2bfloat16(__shfl(__bfloat162float(src[i][0]), take));

                    if (x_or_y) {
                        dst[i][inner_dim / 2].y = val;
                    }
                    else {
                        dst[i][inner_dim / 2].x = val;
                    }
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && std::is_same_v<typename RV2::layout, naive_l>) { // naive -> ortho layout
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim / 2; i++) {

                if constexpr (std::is_same_v<D2_1, float>) {
                    uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(src[i][0]), __float_as_uint(src[i][0]), false, true);
                    dst[i * 2][0] = __uint_as_float(res.x);
                    dst[i * 2 + 1][0] = __uint_as_float(res.y);
                }
                else if constexpr (std::is_same_v<D2_1, bf16>) {
                    uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(src[i][0]), __bfloat16_as_ushort(src[i][0]), false, true);
                    dst[i * 2][0] = __ushort_as_bfloat16(res.x);
                    dst[i * 2 + 1][0] = __ushort_as_bfloat16(res.y);
                }
                else if constexpr (std::is_same_v<D2_1, half>) {
                    uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(src[i][0]), __half_as_ushort(src[i][0]), false, true);
                    dst[i * 2][0] = __ushort_as_half(res.x);
                    dst[i * 2 + 1][0] = __ushort_as_half(res.y);
                } else {
                    static_assert(false, "Unsupported type");
                }

            }

            if constexpr (RV1::outer_dim % 2 == 1) {
                if constexpr (std::is_same_v<D2_1, float>) {
                    uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(src[RV2::outer_dim - 1][0]), __float_as_uint(src[RV2::outer_dim - 1][0]), false, true);
                    dst[RV1::outer_dim - 1][0] = __uint_as_float(res.x);
                }
                else if constexpr (std::is_same_v<D2_1, bf16>) {
                    uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(src[RV2::outer_dim - 1][0]), __bfloat16_as_ushort(src[RV2::outer_dim - 1][0]), false, true);
                    dst[RV1::outer_dim - 1][0] = __ushort_as_bfloat16(res.x);
                }
                else if constexpr (std::is_same_v<D2_1, half>) {
                    uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(src[RV2::outer_dim - 1][0]), __half_as_ushort(src[RV2::outer_dim - 1][0]), false, true);
                    dst[RV1::outer_dim - 1][0] = __ushort_as_half(res.x);
                } else {
                    static_assert(false, "Unsupported type");
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && std::is_same_v<typename RV2::layout, ortho_l>) { // ortho -> naive layout
            const int hi_or_lo = laneid / 32;
            #pragma unroll
            for(int i = 0; i < RV2::outer_dim / 2; i++) {
                dst[i][0] = src[i * 2 + hi_or_lo][0];
            }

            if constexpr (RV2::outer_dim % 2 == 1) {
                dst[RV1::outer_dim - 1][0] = src[RV2::outer_dim - 1][0];
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, align_l> && std::is_same_v<typename RV2::layout, naive_l>) { // naive -> align layout
            const int lane_offset = 8 * (laneid / 32);
            const int inner_lane = laneid % 32;
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim / 2; i++) {
                    
                #pragma unroll
                for(int j = 0; j < 32; j++) {
                    const int inner_dim = (inner_lane ^ j);
                    const int take = lane_offset + (inner_dim % 8) + 16 * (inner_dim / 8);
                    const int outer_dim = i * 2 + (take / 32);
                    const int x_or_y = take % 2;

                    D2_1 val = __float2bfloat16(__shfl(__bfloat162float(src[i][0]), take));

                    if (x_or_y) {
                        dst[outer_dim][(inner_dim % 16) / 2].y = val;
                    }
                    else {
                        dst[outer_dim][(inner_dim % 16) / 2].x = val;
                    }
                }
            }

            if constexpr (RV1::outer_dim % 2 == 1) {
                #pragma unroll
                for(int j = 0; j < 32; j++) {
                    const int inner_dim = (inner_lane ^ j);
                    const int take = lane_offset + (inner_dim % 8) + 16 * (inner_dim / 8);
                    const int outer_dim = (RV1::outer_dim - 1) + (take / 32);
                    const int x_or_y = take % 2;

                    D2_1 val = __float2bfloat16(__shfl(__bfloat162float(src[RV2::outer_dim - 1][0]), take));

                    if (outer_dim >= RV1::outer_dim) {
                        continue;
                    }

                    if (x_or_y) {
                        dst[outer_dim][(inner_dim % 16) / 2].y = val;
                    }
                    else {
                        dst[outer_dim][(inner_dim % 16) / 2].x = val;
                    }
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && std::is_same_v<typename RV2::layout, align_l>) { // align -> naive layout
            const int give = (laneid % 32) / 2;
            const int x_or_y = (laneid % 32) % 2;

            const int give_inner_dim = give % 8;
            const int give_outer_dim = give / 8;

            const int take = (((laneid % 16) / 8) * 32) + ((laneid / 16) * 8) + (laneid % 8);
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                D2_1 val = x_or_y ? src[i * 2 + give_outer_dim][give_inner_dim].y : src[i * 2 + give_outer_dim][give_inner_dim].x;
                dst[i][0] = __float2bfloat16(__shfl(__bfloat162float(val), take));
            }
        }
        else {
            static_assert(false, "Unsupported layout conversion");
        }
    }
}
#else
template<ducks::rv::all RV1, ducks::rv::all RV2>
__device__ static inline void copy(RV1 &dst, const RV2 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;
    static_assert(false, "This function is not implemented for CDNA3");
}
#endif
} // namespace kittens