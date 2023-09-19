#include <stdio.h>
#include <stdlib.h>


// start with a small array to test
#define ROW 512
#define COL 512

__global__ void transpose(int *a, int *c, int nrow, int ncol){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    // printf("CUDA %d %d\n",i, j);
    if (i < ncol && j < nrow) {
        c[j*ncol + i] = a[i*ncol + j];
    }
}

int main( void ){
    int a[ROW][COL];      // host copies of a, c
    int c[ROW][COL];
    int *dev_a;      // device copies of a, c (just pointers)
    int *dev_c;

    // get the size of the arrays I will need
    int size_2d = ROW * COL * sizeof(int);

    // Allocate the memory
    cudaMalloc( (void**)&dev_a, size_2d);
    cudaMalloc( (void**)&dev_c, size_2d);

    // Populate the 2D array on host with something small and known as a test
    for (int i=0; i < ROW; i++){
        for (int j=0; j < COL; j++){
            a[i][j] = i * COL + j;
            // printf("%d ", a[i][j]);
        }
        // printf("\n");
    }

    // Copy the memory
    cudaMemcpy( dev_a, a, size_2d, cudaMemcpyHostToDevice );
    // cudaMemcpy( dev_c, c, size_c, cudaMemcpyHostToDevice );

    // Run the kernal function
    dim3 tblocks(32, 32, 1);
    dim3 grid((COL/tblocks.x)+1, (ROW/tblocks.y)+1, 1);
    printf("Kernel launched with %d %d\n",grid.x, grid.y);
    transpose<<< grid, tblocks >>>(dev_a, dev_c, ROW, COL);
    auto transposeErr = cudaGetLastError();
    if(transposeErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(transposeErr));
    
    auto asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
    // copy the output back to the host
    cudaMemcpy( c, dev_c, size_2d, cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize();

    // Print the output
    printf("\n");
    for (int i = 0; i < ROW; i++){
        for (int j=0; j < COL; j++){
            if (c[i][j] != a[j][i]) {
                printf("Error: (%d,%d) %d\n",i,j, c[i][j]);
            }
        }
    }

    // Releasae the memory
    cudaFree( dev_a );
    cudaFree( dev_c );
}

// Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)     GridXYZ         BlockXYZ                   Name               
//  --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------  --------------  ---------------------------------
//     100.0           15’456          1  15’456.0  15’456.0    15’456    15’456          0.0    17   17    1    32   32    1  transpose(int *, int *, int, int)