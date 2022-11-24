#include "seal/util/hello.cuh"
#include <sys/time.h>
// #include "cgbn/cgbn.h"
#define off 0
#define memSize 16384 * 2
#define N_SEGMENT 128
using namespace std;
double cpuSecond();
#define CHECK(call)                                                          \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }

__device__ volatile int g_mutex;
uint64_t *value_d;
struct seal::util::MultiplyUIntModOperand *roots_d;
uint64_t *polyAptr, *polyBptr, *polyCptr;
cudaStream_t stream[N_SEGMENT];
__global__ void hello()
{
    printf("GPU\n");
}
__device__ inline uint64_t top64(uint64_t x, uint64_t y)

{
    uint64_t a = x >> 32, b = x & 0xffffffff;
    uint64_t c = y >> 32, d = y & 0xffffffff;
    uint64_t ac = a * c;
    uint64_t bc = b * c;
    uint64_t ad = a * d;
    uint64_t bd = b * d;
    uint64_t mid34 = (bd >> 32) + (bc & 0xffffffff) + (ad & 0xffffffff);
    uint64_t upper64 = ac + (bc >> 32) + (ad >> 32) + (mid34 >> 32);
    return upper64;
}
__device__ inline uint64_t mul_mod(std::uint64_t x, struct seal::util::MultiplyUIntModOperand y, std::uint64_t modulue)
{
    std::uint64_t tmp1;
    const std::uint64_t p = modulue;
    tmp1 = top64(x, y.quotient);
    return y.operand * x - tmp1 * p;
}

// __global__ void cuNTT(
//     std::uint64_t *values, int log_n, struct seal::util::MultiplyUIntModOperand const *roots, std::uint64_t modulue)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int N = 1 << log_n;
//     for (int i = 0; i < log_n; i++)
//     {
//         size_t gap = N >> (i+1);
//         size_t m = 1 << (i);
//         uint64_t u = values[idx];
//         uint64_t v = mul_mod(values[address + gap], roots[address / gap + m], modulue);

//         values[address] = (u + v + modulue) % modulue;
//         values[address + t] = (u - v + modulue) % modulue;
//         if(threadIdx.x == 0){
//             atomicAdd((int*)&g_mutex, 1);
//         }
//         while(g_mutex % blockDim.x != 0){

//         }
//         // __syncthreads();
//     }
// }

// __global__ void cuNTT(
//     std::uint64_t *values, int log_n, struct seal::util::MultiplyUIntModOperand const *roots, std::uint64_t modulue)
// {
//     int idx = threadIdx.x;
//     int N = 1 << log_n;
//     int unrollNum = N / blockDim.x / 2;
//     int unrolloffset = blockDim.x ;
//     // printf("%d , %d , %d \n",N,unrollNum,unrolloffset);
//     for (int i = 0; i < log_n; i++)
//     {
//         size_t gap = N >> (i+1);
//         size_t m = 1 << (i);
//         // seal::util::MultiplyUIntModOperand r = roots[idx / gap + m];
//         uint64_t u = values[address];
//         uint64_t v = mul_mod(values[address + gap], roots[address / gap + m], modulue);
//         values[address] = (u + v + modulue) % modulue;
//         values[address + gap] = (u - v + modulue) % modulue;
//         __syncthreads();
//     }
// }

__global__ void cuNTT(
    std::uint64_t *values, int log_n, struct seal::util::MultiplyUIntModOperand const *roots, std::uint64_t modulue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d,%d,%d\n",idx,blockDim.x,blockIdx.x);
    int N = 1 << log_n;
    // printf("gpu call\n");
    for (int i = 0; i < log_n; i++)
    {
        int t = (N / 2) >> i;
        int address = idx / t * t + idx;
        int m = (1 << i);
        uint64_t u = values[address];
        uint64_t v = mul_mod(values[address + t], roots[idx / t + m], modulue);
        // if(idx / t + m == off)printf("%d,%d,%d,%d,value%lld,r%lld\n",i,address,address+t,idx / t +
        // m,values[address],roots[idx / t + m].operand); if(address ==
        // 2320)printf("%d,%d,%d,%d,value%lld,r%lld\n",i,address,address+t,idx / t + m,values[address],roots[idx / t +
        // m].operand);
        values[address] = (u + v + 2 * modulue) % modulue;
        values[address + t] = (u - v + 2 * modulue) % modulue;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            // printf("blockIdx%d\n",blockIdx.x);
            atomicAdd((int *)&g_mutex, 1);
            // printf("%d,%d\n",g_mutex,gridDim.x);
        }
        while (g_mutex % (gridDim.x) != 0)
        {
            // printf("%d\n",g_mutex);
        }
        // printf("%lu\n",values[6666]);
    }
}

// std::uint64_t mul_mod(std::uint64_t a,std::uint64_t b, std::uint64_t p){
//     a %= p; b %= p;
//     std::uint64_t c = (long double)a * b / p;
//     std::uint64_t ans = a * b - c * p;
//     if (ans < 0) ans += p;
//     else if (ans >= p) ans -= p;
//     return ans;
// }
__device__ inline unsigned char add_uint64(
    unsigned long long operand1, unsigned long long operand2, unsigned long long *result)
{
    *result = operand1 + operand2;
    return static_cast<unsigned char>(*result < operand1);
}
__global__ void cuProduct(
    uint64_t *polyAptr, uint64_t *polyBptr, uint64_t *polyCptr, int N, uint64_t modulus_value, uint64_t const_ratio_0,
    uint64_t const_ratio_1)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
    uint64_t a = polyAptr[idx];
    uint64_t b = polyBptr[idx];
    // multiply_uint64(get<0>(I), get<1>(I), z);
    z[0] = a * b;
    z[1] = top64(a, b);
    // Multiply input and const_ratio
    // Round 1
    // multiply_uint64_hw64(z[0], const_ratio_0, &carry);
    carry = top64(z[0], const_ratio_0);
    // multiply_uint64(z[0], const_ratio_1, tmp2);
    tmp2[0] = z[0] * const_ratio_1;
    tmp2[1] = top64(z[0], const_ratio_1);

    tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);
    // tmp3 = tmp2[1] + (unsigned char)()
    // Round 2
    // multiply_uint64(z[1], const_ratio_0, tmp2);
    tmp2[0] = z[1] * const_ratio_0;
    tmp2[1] = top64(z[1], const_ratio_0);
    carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

    // This is all we care about
    tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

    // Barrett subtraction
    tmp3 = z[0] - tmp1 * modulus_value;

    // Claim: One more subtraction is enough
    // get<2>(I) = SEAL_COND_SELECT(tmp3 >= modulus_value, tmp3 - modulus_value, tmp3);
    polyCptr[idx] = (tmp3 + 2 * modulus_value) % modulus_value;

    // printf("%lld*%lld%s%lld=%lld\n",a,b,"%",modulus_value,polyCptr[idx]);
}
int test1(
    std::uint64_t *values, int log_n, struct seal::util::MultiplyUIntModOperand const *roots, std::uint64_t modulue)
{
    return 0;
    int N = 1 << log_n;
    int nByte = N * sizeof(uint64_t);
    // uint64_t *value_d;struct seal::util::MultiplyUIntModOperand *roots_d;
    // printf("@value_d%p\n",value_d);
    if (!value_d)
    {
        printf("value_d\n");
        CHECK(cudaMalloc(&value_d, memSize * sizeof(uint64_t)));
        printf("value_d%p\n", value_d);
    }
    if (!roots_d)
    {
        printf("roots_d\n");
        cudaMalloc(&roots_d, memSize * sizeof(struct seal::util::MultiplyUIntModOperand));
        printf("roots_d%p\n", roots_d);
    }
    double start = cpuSecond();
    // printf("%d\n",N);
    // cudaMalloc(&value_d,N * sizeof(uint64_t));
    // printf("value_d%p\n",value_d);
    // cudaHostAlloc(&values,nByte,cudaHostAllocMapped);
    cudaMemcpy(value_d, values, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    // cudaMalloc(&roots_d,N * sizeof(struct seal::util::MultiplyUIntModOperand));
    // cudaHostAlloc((void**)&roots,N * sizeof(struct seal::util::MultiplyUIntModOperand),cudaHostAllocMapped);
    cudaMemcpy(roots_d, roots, N * sizeof(struct seal::util::MultiplyUIntModOperand), cudaMemcpyHostToDevice);
    double startk = cpuSecond();
    cuNTT<<<N / 2 / 1024, 1024>>>(value_d, log_n, roots_d, modulue);
    double endk = cpuSecond();
    cudaMemcpy(values, value_d, nByte, cudaMemcpyDeviceToHost);
    // cudaFree(value_d);
    // cudaFree(roots_d);
    double end = cpuSecond();
    // printf("%lf,%lf",endk - startk, end -start);
    return 1;
#ifdef de
    int N = 1 << log_n;
    int nByte = N * sizeof(uint64_t);
    // uint64_t *value_d;struct seal::util::MultiplyUIntModOperand *roots_d;
    double start = cpuSecond();
    // cudaMalloc(&value_d,N * sizeof(uint64_t));
    cudaHostAlloc(&values, nByte, cudaHostAllocMapped);
    // cudaMemcpy(value_d,values,N * sizeof(uint64_t),cudaMemcpyHostToDevice);
    // cudaMalloc(&roots_d,N * sizeof(struct seal::util::MultiplyUIntModOperand));
    cudaHostAlloc((void **)&roots, N * sizeof(struct seal::util::MultiplyUIntModOperand), cudaHostAllocMapped);
    // cudaMemcpy(roots_d,roots,N * sizeof(struct seal::util::MultiplyUIntModOperand),cudaMemcpyHostToDevice);
    double startk = cpuSecond();
    cuNTT<<<N / 2 / 1024, 1024>>>(values, log_n, roots, modulue);
    double endk = cpuSecond();
    // cudaMemcpy(values,value_d,nByte,cudaMemcpyDeviceToHost);
    cudaFreeHost(values);
    cudaFreeHost((void *)roots);
    double end = cpuSecond();
// printf("%lf,%lf",endk - startk, end -start);
#endif
    size_t n = size_t(1) << log_n;
    seal::util::MultiplyUIntModOperand r;
    std::uint64_t u;
    std::uint64_t v;
    // pointers for faster indexing
    std::uint64_t *x = nullptr;
    std::uint64_t *y = nullptr;
    std::size_t gap = n >> 1;
    std::size_t m = 1;
    printf("%d\n", n);
    int cnt = 0;
    // printf("!!!!!!!!!!!!!%p\n", values[6666]);
    int rootoff = 0;
    // exit(1);
    for (; m <= (n >> 1); m <<= 1)
    {
        // printf("%d\n",m);
        std::size_t offset = 0;
        if (gap < 4)
        {
            for (std::size_t i = 0; i < m; i++)
            {
                r = *++roots;
                rootoff++;
                x = values + offset;
                y = x + gap;
                for (std::size_t j = 0; j < gap; j++)
                {
                    u = *x;
                    // v = 1;
                    v = multiply_uint_mod_lazy(*y, r, modulue);
                    if (rootoff == off || x - values == 2320)
                        printf("#%d,%d,%d,%d,value%lld,r%lld\n", i, x - values, y - values, rootoff, u, r.operand);
                    *x++ = (u + v) % modulue;
                    *y++ = (u - v + 2 * modulue) % modulue;
                }
                offset += gap << 1;
            }
        }
        else
        {
            for (std::size_t i = 0; i < m; i++)
            {
                // printf("%d\n",offset);
                r = *++roots;
                rootoff++;
                x = values + offset;
                y = x + gap;
                for (std::size_t j = 0; j < gap; j += 1)
                {
                    // printf("%d\n",cnt++);
                    u = *x;
                    // v = 1;
                    v = multiply_uint_mod_lazy(*y, r, modulue);
                    if (rootoff == off || x - values == 2320)
                        printf("#%d,%d,%d,%d,value%lld,r%lld\n", i, x - values, y - values, rootoff, u, r.operand);
                    *x++ = (u + v) % modulue;
                    *y++ = (u - v + 2 * modulue) % modulue;
                    // u = *x;
                    // v = multiply_uint_mod_lazy(*y, r, modulue);
                    // *x++ = (u + v) % modulue;
                    // *y++ = (u - v + 2 * modulue) % modulue;
                    // u = *x;
                    // v = multiply_uint_mod_lazy(*y, r, modulue);

                    // *x++ = (u + v) % modulue;
                    // *y++ = (u - v + 2 * modulue) % modulue;
                    // u = *x;
                    // v = multiply_uint_mod_lazy(*y, r, modulue);
                    // *x++ = (u + v) % modulue;
                    // *y++ = (u - v + 2 * modulue) % modulue;
                }
                offset += gap << 1;
            }
        }
        gap >>= 1;
    }
    // exit(0);
    return 1;
}

int test1(std::complex<double> *value, int log_n, std::complex<double> const *roots, std::uint64_t modulue)
{
    // printf("fft\n");
    return 0;
}
int Product(
    seal::util::ConstCoeffIter operand1, seal::util::ConstCoeffIter operand2, size_t coeff_count,
    const seal::Modulus &modulus, seal::util::CoeffIter result)
{
    return 0;
    // printf("%d\n",coeff_count);
    if (!polyAptr)
    {
        CHECK(cudaMalloc(&polyAptr, memSize * sizeof(uint64_t)));
        CHECK(cudaMalloc(&polyBptr, memSize * sizeof(uint64_t)));
        CHECK(cudaMalloc(&polyCptr, memSize * sizeof(uint64_t)));
    }
    const uint64_t modulus_value = modulus.value();
    const uint64_t const_ratio_0 = modulus.const_ratio()[0];
    const uint64_t const_ratio_1 = modulus.const_ratio()[1];

    int nByte = coeff_count * sizeof(uint64_t);
    cudaMemcpy(polyAptr, operand1.ptr(), nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(polyBptr, operand2.ptr(), nByte, cudaMemcpyHostToDevice);

    cuProduct<<<coeff_count/256, 256>>>(
            polyAptr, polyBptr, polyCptr, coeff_count, modulus_value, const_ratio_0,
            const_ratio_1);

    cudaMemcpy(result.ptr(), polyCptr, nByte, cudaMemcpyDeviceToHost);
    return 1;

    // int N_SEGMENT = 128;
    int iElem = coeff_count / N_SEGMENT;

    // cudaStream_t stream[N_SEGMENT];
    if (!stream[0])
    {
        for (int i = 0; i < N_SEGMENT; i++)
        {
            CHECK(cudaStreamCreate(&stream[i]));
        }
    }
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    for (int i = 0; i < N_SEGMENT; i++)
    {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(
            &polyAptr[ioffset], operand1.ptr() + ioffset, nByte / N_SEGMENT, cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(
            &polyBptr[ioffset], operand2.ptr() + ioffset, nByte / N_SEGMENT, cudaMemcpyHostToDevice, stream[i]));
        cuProduct<<<1, iElem, 0, stream[i]>>>(
            &polyAptr[ioffset], &polyBptr[ioffset], &polyCptr[ioffset], iElem, modulus_value, const_ratio_0,
            const_ratio_1);
        CHECK(cudaMemcpyAsync(
            result.ptr() + ioffset, &polyCptr[ioffset], nByte / N_SEGMENT, cudaMemcpyDeviceToHost, stream[i]));
    }
    // CHECK(cudaEventRecord(stop, 0));
    // int counter = 0;
    // while (cudaEventQuery(stop) == cudaErrorNotReady)
    // {
    // counter++;
    // }
    // printf("cpu counter:%d\n", counter);
    cudaDeviceSynchronize();
    // for(int i=0;i<N_SEGMENT;i++)
    // {
    //     CHECK(cudaStreamDestroy(stream[i]));
    // }
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // for(int i = 8000; i < 8001; i++){
    //     __int128_t a = operand1[i];
    //     __int128_t b = operand2[i];
    //     // __int128_t p =
    //     __int128_t res = a * b % modulus_value;
    //     // res = (int64_t)res;
    //     // printf("%lld,%lld,%lld\n",a,b,res);
    //     // printf("#%lld*%lld%s%lld=%lld\n",operand1[i],operand2[i],"%",modulus_value,result[i]);
    //     // printf("%d\n",res == result[i]);
    // }
    // exit(1);

    return 1;
    // exit(1);
}