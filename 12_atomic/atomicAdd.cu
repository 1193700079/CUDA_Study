
#include <cstdio>

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

__device__ double myatomicAdd(double *address, double val)
{
    unsigned long long *address_as_ull = (unsigned long long *)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void sumArrayWithAtomic(real *array, real *sum, int N)
{
    // 获取线程的全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // 确保索引在数组范围内
    if (index < N)
    {
        // 使用atomicAdd原子地更新总和  目前是支持double类型 老一些的硬件设备可能不支持 需要利用CAS
        myatomicAdd(sum, array[index]);
    }
}

int main()
{
    const int N = 100; // 数组大小
    const size_t size = N * sizeof(real);

    // 分配和初始化设备内存
    real *d_array, *d_sum;
    cudaMalloc(&d_array, size);
    cudaMalloc(&d_sum, sizeof(real));
    real h_array[N], h_sum = 0;
    for (int i = 1; i <= N; i++)
    {
        h_array[i - 1] = i;
    }
    // 假设h_array已经初始化
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(real), cudaMemcpyHostToDevice);

    // 启动kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sumArrayWithAtomic<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_sum, N);

    // 将结果复制回主机
    cudaMemcpy(&h_sum, d_sum, sizeof(real), cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_array);
    cudaFree(d_sum);

    printf("Total sum: %f\n", h_sum);

    return 0;
}
