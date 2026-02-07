#// we did it in python notebook but we can also do it in c++ file and compile it with nvcc 
%%writefile Matrix_multiplication.cu 
#include<bits/stdc++.h>
#include<cuda_runtime.h>
using namespace std;
__global__ void MatrixMultiplication(float* A, float* B, float* C,
                                     int M, int N, int P,
                                     int offset, int KK)
{
    int batchIndex = threadIdx.x + offset;
    if(batchIndex >= KK) return;

    float* a = A + batchIndex * M * N;
    float* b = B + batchIndex * N * P;
    float* c = C + batchIndex * M * P;

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            for(int l = 0; l < P; l++)
                c[i*P + l] += a[i*N + j] * b[j*P + l];
}


void printOneMatrix(float* A,int batchIndex, int M, int N){
    cout<<"Batch index:"<<batchIndex<<endl;
    float* matrixA = A + batchIndex*(M*N);
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            cout<<matrixA[i*N+j]<<" ";
        }
        cout<<endl;
    }
}
    

// main function 
int main(int argumentCount, char* argumentVector[]){
    int threadCount = atoi(argumentVector[1]);
    int KK = atoi(argumentVector[2]); 
    
    // amount of matrix to the operation
    int matrixM =3, matrixN=3, matrixP=3;
    int NewMatrixSizeA=matrixM*matrixN* KK;
    int NewMatrixSizeB=matrixN*matrixP* KK;
    int NewMatrixSizeC= matrixM*matrixP* KK;

    
    // allocate memory for the matrices
    float *hardware_A = new float[NewMatrixSizeA];
    float *hardware_B = new float[NewMatrixSizeB];
    float *hardware_C = new float[NewMatrixSizeC];

    
    // malloc (GPU allocation)
    float *device_A, *device_B, *device_C;



    cudaMalloc(&device_A, NewMatrixSizeA * sizeof(float));
    cudaMalloc(&device_B, NewMatrixSizeB * sizeof(float));
    cudaMalloc(&device_C, NewMatrixSizeC * sizeof(float));
    cudaMemset(device_C, 0, NewMatrixSizeC * sizeof(float));

    // initialize the matrices with random values
    for(int i=0; i<NewMatrixSizeA; i++){
        hardware_A[i] = (rand())%10;
    }
    for(int i=0; i<NewMatrixSizeB; i++){
        hardware_B[i] = (rand())%10;
    }

    // copy the matrices from host to device
    cudaMemcpy(device_A, hardware_A, NewMatrixSizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, hardware_B, NewMatrixSizeB * sizeof(float), cudaMemcpyHostToDevice);

    // kernel launch parameters
    int gun_korte_hbe = KK; // KK is already defined at line 32
    int offset=0;
    while(gun_korte_hbe>0)
    {
        int currentBatchSize = min(gun_korte_hbe, threadCount);
        MatrixMultiplication<<<1, currentBatchSize>>>(device_A,device_B,device_C,matrixM,matrixN,matrixP,offset,KK);
        cudaDeviceSynchronize();
        offset+=currentBatchSize;
        gun_korte_hbe-=currentBatchSize;
    }

    // copy the result back to host
    cudaMemcpy(hardware_C,device_C,NewMatrixSizeC*sizeof(float),cudaMemcpyDeviceToHost);

    cout<<"All operations are done"<<endl;

    //output section
    int HowManyToPrint = min(KK, 5);
    for(int i=0;i<HowManyToPrint;i++){
        cout<<"Matrix set :"<<i<<"----\n";
        cout<<"Matrix A:\n";
        printOneMatrix(hardware_A,i, matrixM, matrixN);
        cout<<"Matrix B:\n";
        printOneMatrix(hardware_B,i, matrixN, matrixP);
        cout<<"Matrix C:\n";
        printOneMatrix(hardware_C,i, matrixM, matrixP);
    }
    // free the memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    delete [] hardware_A;
    delete [] hardware_B;
    delete [] hardware_C;
    return 0;

}

// !nvcc -arch=sm_75 Matrix_multiplication.cu -o mm
// !./mm 5 102
// !time ./mm 5 102