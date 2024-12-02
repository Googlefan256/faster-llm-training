#include <cuda_bf16.h>

extern "C" __global__ void sqrt_forward_kernel(__nv_bfloat16 *output_ptr, const __nv_bfloat16 *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(sqrtf(__bfloat162float(input_ptr[idx])));
    }
}