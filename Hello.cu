#include<stdio.h>
__global__ void hellomama(){
    printf("Hello, Mama,first time try kortesi!\n blockIdx.x=%d,threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
}
int main(){
    hellomama<<<2,5>>>();
    cudaDeviceSynchronize();
    return 0;
}