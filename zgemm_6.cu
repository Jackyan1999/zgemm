// optimize sgemm

#include <stdio.h>
#include <stdlib.h>
#include "hpl_complex.h"
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

 //#define PRINT_MAT
// #define PRINT_sharedmem
//  #define PRINT_REG

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_DOUBLE2(pointer) (reinterpret_cast<double2 *>(&(pointer))[0])

#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }
// const static complex zmone   =   { -1.0, 0.0 };
const static complex zone = {1.0, 0.0};
const static complex zero = {0.0, 0.0};


// __global__ void Zgemm(
//     complex *__restrict__ A,
//     complex *__restrict__ B,
//     complex *__restrict__ C,
//     const int M,
//     const int N,
//     const int K)
// {
//     // Block index
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     // Thread index
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     // the threads number in Block of X,Y
//     const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
//     const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
//     const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

//     // thread id in cur Block
//     const int tid = ty * THREAD_X_PER_BLOCK + tx;

//     // shared memory
//     //the first 2 is double buffer the second is seperation of complex
//     __shared__ double As[2][2][BLOCK_SIZE_K][BLOCK_SIZE_M];
//     __shared__ double Bs[2][2][BLOCK_SIZE_K][BLOCK_SIZE_N];

//     // registers for C
//     complex accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    
// #pragma unroll
//     for (int i = 0; i < THREAD_SIZE_Y; i++)
//     {
// #pragma unroll
//         for (int j = 0; j < THREAD_SIZE_X; j++)
//         {
//             accum[i][j].r = 0.0;
//             accum[i][j].i = 0.0;
//         }
//     }
//     // registers for A and B
//     double  frag_a[2][2][THREAD_SIZE_Y];
//     double  frag_b[2][2][THREAD_SIZE_X];
//     // registers load global memory
//     const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;
//     const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;
//     complex ldg_a_reg[ldg_num_a];
//     complex ldg_b_reg[ldg_num_b];

//     // threads number in one row
//     const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K;
//     const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N;

//     // row number and col number that needs to be loaded by this thread
//     const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
//     const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

//     const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW;
//     const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW;

//     // row stride that thread uses to load multiple rows of a tile
//     const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
//     const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

//     A = &A[(BLOCK_SIZE_M * by) * K];
//     B = &B[BLOCK_SIZE_N * bx];

// // transfer first tile from global mem to shared mem
// //  load A from global memory to shared memory
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//     {
//         int ldg_index = i / A_TILE_ROW_STRIDE;
//         ldg_a_reg[ldg_index] = A[OFFSET(
//             A_TILE_ROW_START + i, // row
//             A_TILE_COL,           // col
//             K)];
//         As[0][0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].r;
//         As[0][1][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].i;
//     }
//     // printf("000000000\n");
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//     {
//         int ldg_index = i / A_TILE_ROW_STRIDE;
//         ldg_b_reg[ldg_index] = B[OFFSET(
//             B_TILE_ROW_START + i, // row
//             B_TILE_COL,           // col
//             N)];
//         Bs[0][0][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].r;
//         Bs[0][1][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].i;
//     }

//     __syncthreads();

//     // load A from shared memory to register
//     int sa2reg_idx = ty * THREAD_SIZE_Y;
//     for (int i = 0; i < THREAD_SIZE_Y; i++)
//     {
//         frag_a[0][0][i] = As[0][0][0][i + sa2reg_idx];
//         frag_a[0][1][i] = As[0][1][0][i + sa2reg_idx];
//     }
//     // load B from shared memory to register
//     int sb2reg_idx = tx * THREAD_SIZE_X;
//     for (int i = 0; i < THREAD_SIZE_X; i++)
//     {
//         frag_b[0][0][i] = Bs[0][0][0][i + sb2reg_idx];
//         frag_b[0][1][i] = Bs[0][1][0][i + sb2reg_idx];
//     }

//     int write_stage_idx = 1;
//     int tile_idx = 0;
//     do
//     {
//         tile_idx += BLOCK_SIZE_K;
//         // load next tile from global mem
//         if (tile_idx < K)
//         {
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / A_TILE_ROW_STRIDE;
//                 ldg_a_reg[ldg_index] = A[OFFSET(
//                     A_TILE_ROW_START + i,  // row
//                     A_TILE_COL + tile_idx, // col
//                     K)];
//             }
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / A_TILE_ROW_STRIDE;
//                 ldg_b_reg[ldg_index] = B[OFFSET(
//                     tile_idx + B_TILE_ROW_START + i, // row
//                     B_TILE_COL,                      // col
//                     N)];
//             }
//         }

//         int load_stage_idx = write_stage_idx ^ 1;
//         int next_stage_flag = load_stage_idx;
// #ifdef PRINT_sharedmem

//         if (bx == 0 && by == 0 && tx == 0 && ty == 0)
//         {
//             printf("%d %d %d %d\n", bx, by, tx, ty);
//             printf("tileidx = %d\n", tile_idx);
//             printf("AS[0]:\n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_M; j++)
//                 {
//                     printf("%f ", As[0][0][i][j] );
//                     printf("+ %f i",As[0][1][i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("AS[1]:\n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_M; j++)
//                 {
//                     printf("%f ", As[1][0][i][j] );
//                     printf("+ %f i",As[1][1][i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("Bs[0]:  \n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_N; j++)
//                 {
//                     printf("%f ", Bs[0][0][i][j] );
//                     printf("+ %f i",Bs[0][1][i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("BS[1]:\n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_N; j++)
//                 {
//                     printf("%f ", Bs[1][0][i][j] );
//                     printf("+ %f i",Bs[1][1][i][j]);
//                 }
//                 printf("\n");
//             }
//         }
// #endif
// #pragma unroll
//         for (int j = 0; j < BLOCK_SIZE_K; ++j)
//         {
//             // load next tile from shared mem to register
//             //next_stage_flag = (j == BLOCK_SIZE_K - 1) ? load_stage_idx : write_stage_idx;
//             // next_stage_flag = (j == BLOCK_SIZE_K - 1) ? write_stage_idx : load_stage_idx;
//             next_stage_flag = load_stage_idx;
//             //  load A from shared memory to register
//             // if (j != BLOCK_SIZE_K - 1)
//             // {
//             for (int i = 0; i < THREAD_SIZE_Y; i++)
//             {
//                 frag_a[(j + 1) % 2][0][i] = As[next_stage_flag][0][(j + 1) % BLOCK_SIZE_K][i + sa2reg_idx];
//                 frag_a[(j + 1) % 2][1][i] = As[next_stage_flag][1][(j + 1) % BLOCK_SIZE_K][i + sa2reg_idx];
//             }
//             // load B from shared memory to register
//             for (int i = 0; i < THREAD_SIZE_X; i++)
//             {
//                 frag_b[(j + 1) % 2][0][i] = Bs[next_stage_flag][0][(j + 1) % BLOCK_SIZE_K][i + sb2reg_idx];
//                 frag_b[(j + 1) % 2][1][i] = Bs[next_stage_flag][1][(j + 1) % BLOCK_SIZE_K][i + sb2reg_idx];
//             }
//             // }
// #ifdef PRINT_REG
//             if (bx == 0 && by == 0 && tx == 0 && ty == 0)
//             {
//                 printf("next_stage_flag:%d\n", next_stage_flag);
//                 printf("j == %d\n", j);
//                 printf("regA[0]:\n");
//                 for (int g = 0; g < THREAD_SIZE_Y; g++)
//                 {
//                     printf("%f + %fi ", frag_a[0][0][g], frag_a[0][1][g]);
//                 }
//                 printf("\n");
//                 printf("regA[1]:\n");
//                 for (int g = 0; g < THREAD_SIZE_Y; g++)
//                 {
//                     printf("%f + %fi ", frag_a[1][0][g], frag_a[1][1][g]);
//                 }
//                 printf("\n");
//                 printf("regB[0]:\n");
//                 for (int g = 0; g < THREAD_SIZE_X; g++)
//                 {
//                     printf("%f + %fi ", frag_b[0][0][g], frag_b[0][1][g]);
//                 }
//                 printf("\n");
//                 printf("regB[1]:\n");
//                 for (int g = 0; g < THREAD_SIZE_X; g++)
//                 {
//                     printf("%f + %fi ", frag_b[1][0][g], frag_b[1][1][g]);
//                 }
//                 printf("\n");
//             }
// #endif
//             // compute C THREAD_SIZE_X x THREAD_SIZE_Y
//             // complex temp;

// //#pragma unroll
//             for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
//             {
// //#pragma unroll
//                 for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
//                 {
//                     // printf("4444444444444\n");
//                     accum[thread_y][thread_x].r += frag_a[j%2][0][thread_y] * frag_b[j%2][0][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].r);

//                     accum[thread_y][thread_x].r -= frag_a[j%2][1][thread_y] * frag_b[j%2][1][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].r);

//                     accum[thread_y][thread_x].i += frag_a[j%2][0][thread_y] * frag_b[j%2][1][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].i);

//                     accum[thread_y][thread_x].i += frag_a[j%2][1][thread_y] * frag_b[j%2][0][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].i);

//                     //printf("\n");
//                     //if (bx == 0 && by == 0 && tx == 0 && ty == 0){
//                         // printf("%f %f %f %f %f %f %f %f\n",
//                         // frag_a[j][0][thread_y],frag_b[j][0][thread_x],
//                         // frag_a[j][1][thread_y],frag_b[j][1][thread_x],
//                         // frag_a[j][0][thread_y],frag_b[j][1][thread_x],
//                         // frag_a[j][1][thread_y],frag_b[j][0][thread_x]
//                         // );
//                     // printf("%f , %f %f %f",frag_a[j][0][thread_y] * frag_b[j][0][thread_x],
//                     // frag_a[j][1][thread_y] * frag_b[j][1][thread_x],
//                     // frag_a[j][0][thread_y] * frag_b[j][1][thread_x],
//                     // frag_a[j][1][thread_y] * frag_b[j][0][thread_x]
//                     // );
                    
//                         //printf("%f + %f i ",accum[thread_y][thread_x].r,accum[thread_y][thread_x].i);
//                     //}
//                 }
//                 // if (bx == 0 && by == 0 && tx == 0 && ty == 0) 
//                 //printf("\n");
//             }
//             //if (bx == 0 && by == 0 && tx == 0 && ty == 0) printf("\n");
//         }

//         if (tile_idx < K)
//         {
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / A_TILE_ROW_STRIDE;
//                 As[write_stage_idx][0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].r;
//                 As[write_stage_idx][1][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].i;
//             }
// // printf("222222222222\n");
// //  load B from global memory to shared memory
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / A_TILE_ROW_STRIDE;
//                 Bs[write_stage_idx][0][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].r;
//                 Bs[write_stage_idx][1][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].i;
//             }
//             // use double buffer, only need one sync
//             __syncthreads();
//             // prefetch next reg
//             for (int i = 0; i < THREAD_SIZE_Y; i++)
//             {
//                 frag_a[0][0][i] = As[write_stage_idx][0][0][i + sa2reg_idx];
//                 frag_a[0][1][i] = As[write_stage_idx][0][1][i + sa2reg_idx];
//             }
//             // load B from shared memory to register
//             for (int i = 0; i < THREAD_SIZE_X; i++)
//             {
//                 frag_b[0][0][i] = Bs[write_stage_idx][0][0][i + sb2reg_idx];
//                 frag_b[0][1][i] = Bs[write_stage_idx][0][1][i + sb2reg_idx];
//             }

//             // switch
//             write_stage_idx ^= 1;
//         }
//     } while (tile_idx < K);

// // store back to C
// #pragma unroll
//     for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
//     {
// #pragma unroll
//         for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
//         {
//             C[OFFSET(
//                 BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
//                 BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
//                 N)] = accum[thread_y][thread_x];
//                 // if (bx == 0 && by == 0 && tx == 0 && ty == 0){
//                 //     printf("%f + %f i",accum[thread_y][thread_x].r,accum[thread_y][thread_x].i);
//                 // }
//         }
//         //if (bx == 0 && by == 0 && tx == 0 && ty == 0) printf("\n");
//     }
// }
// __global__ void Zgemm(
//     complex *__restrict__ A,
//     complex *__restrict__ B,
//     complex *__restrict__ C,
//     const int M,
//     const int N,
//     const int K)
// {
//     // Block index
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     // Thread index
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     // the threads number in Block of X,Y
//     const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;//16
//     const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;//8
//     const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

//     // thread id in cur Block
//     const int tid = ty * THREAD_X_PER_BLOCK + tx;

//     // shared memory
//     //the first 2 is double buffer the second is seperation of complex
//     __shared__ double As[2][2][BLOCK_SIZE_K][BLOCK_SIZE_M];
//     __shared__ double Bs[2][2][BLOCK_SIZE_K][BLOCK_SIZE_N];

//     // registers for C
//     complex accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    
// #pragma unroll
//     for (int i = 0; i < THREAD_SIZE_Y; i++)
//     {
// #pragma unroll
//         for (int j = 0; j < THREAD_SIZE_X; j++)
//         {
//             accum[i][j].r = 0.0;
//             accum[i][j].i = 0.0;
//         }
//     }
//     // registers for A and B
//     double  frag_a[2][2][THREAD_SIZE_Y];
//     double  frag_b[2][2][THREAD_SIZE_X];
//     // registers load global memory
//     const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;//2
//     const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;//4
//     complex ldg_a_reg[ldg_num_a];
//     complex ldg_b_reg[ldg_num_b];

//     // threads number in one row
//     const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K;//8
//     const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N;//64

//     // row number and col number that needs to be loaded by this thread
//     const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
//     const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

//     const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW;
//     const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW;

//     // row stride that thread uses to load multiple rows of a tile
//     const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
//     const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

//     //load double2
//     const int warp_id = tid / 32;
//     const int lane_id = tid % 32;
//     const int a_tile_index =  warp_id/2*8 + lane_id/8*2; //warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
//     const int b_tile_index =  warp_id%2*16 + lane_id%8*2; //(lane_id % 16) * 4; // (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;

//     A = &A[(BLOCK_SIZE_M * by) * K];
//     B = &B[BLOCK_SIZE_N * bx];

// // transfer first tile from global mem to shared mem
// //  load A from global memory to shared memory
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)//16
//     {
//         int ldg_index = i / A_TILE_ROW_STRIDE;
//         ldg_a_reg[ldg_index] = A[OFFSET(
//             A_TILE_ROW_START + i, // row
//             A_TILE_COL,           // col
//             K)];
//         As[0][0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].r;
//         As[0][1][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].i;
//     }
//     // printf("000000000\n");
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)//2
//     {
//         int ldg_index = i / B_TILE_ROW_STRIDE;
//         ldg_b_reg[ldg_index] = B[OFFSET(
//             B_TILE_ROW_START + i, // row
//             B_TILE_COL,           // col
//             N)];
//         Bs[0][0][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].r;
//         Bs[0][1][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].i;
//     }

//     __syncthreads();

//     // load A from shared memory to register
//     // int sa2reg_idx = ty * THREAD_SIZE_Y;
//     // for (int i = 0; i < THREAD_SIZE_Y; i++)
//     // {
//     //     frag_a[0][0][i] = As[0][0][0][i + sa2reg_idx];
//     //     frag_a[0][1][i] = As[0][1][0][i + sa2reg_idx];
//     // }
//     // // load B from shared memory to register
//     // int sb2reg_idx = tx * THREAD_SIZE_X;
//     // for (int i = 0; i < THREAD_SIZE_X; i++)
//     // {
//     //     frag_b[0][0][i] = Bs[0][0][0][i + sb2reg_idx];
//     //     frag_b[0][1][i] = Bs[0][1][0][i + sb2reg_idx];
//     // }

//     // load A from shared memory to register
//     FETCH_DOUBLE2(frag_a[0][0][0]) = FETCH_DOUBLE2(As[0][0][0][a_tile_index]);
//     FETCH_DOUBLE2(frag_a[0][1][0]) = FETCH_DOUBLE2(As[0][1][0][a_tile_index]);

//     FETCH_DOUBLE2(frag_a[0][0][2]) = FETCH_DOUBLE2(As[0][0][0][a_tile_index + 16]);
//     FETCH_DOUBLE2(frag_a[0][1][2]) = FETCH_DOUBLE2(As[0][1][0][a_tile_index + 16]);
    
//     // load B from shared memory to register
//     FETCH_DOUBLE2(frag_b[0][0][0]) = FETCH_DOUBLE2(Bs[0][0][0][b_tile_index]);
//     FETCH_DOUBLE2(frag_b[0][1][0]) = FETCH_DOUBLE2(Bs[0][1][0][b_tile_index]);

//     FETCH_DOUBLE2(frag_b[0][0][2]) = FETCH_DOUBLE2(Bs[0][0][0][b_tile_index + 32]);
//     FETCH_DOUBLE2(frag_b[0][1][2]) = FETCH_DOUBLE2(Bs[0][1][0][b_tile_index + 32]);

//     int write_stage_idx = 1;
//     int tile_idx = 0;
//     do
//     {
//         tile_idx += BLOCK_SIZE_K;
//         // load next tile from global mem
//         if (tile_idx < K)
//         {
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / A_TILE_ROW_STRIDE;
//                 ldg_a_reg[ldg_index] = A[OFFSET(
//                     A_TILE_ROW_START + i,  // row
//                     A_TILE_COL + tile_idx, // col
//                     K)];
//             }
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / B_TILE_ROW_STRIDE;
//                 ldg_b_reg[ldg_index] = B[OFFSET(
//                     tile_idx + B_TILE_ROW_START + i, // row
//                     B_TILE_COL,                      // col
//                     N)];
//             }
//         }

//         int load_stage_idx = write_stage_idx ^ 1;
// #ifdef PRINT_sharedmem

//         if (bx == 0 && by == 0 && tx == 0 && ty == 0)
//         {
//             printf("%d %d %d %d\n", bx, by, tx, ty);
//             printf("tileidx = %d\n", tile_idx);
//             printf("AS[0]:\n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_M; j++)
//                 {
//                     printf("%f ", As[0][0][i][j] );
//                     printf("+ %f i",As[0][1][i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("AS[1]:\n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_M; j++)
//                 {
//                     printf("%f ", As[1][0][i][j] );
//                     printf("+ %f i",As[1][1][i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("Bs[0]:  \n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_N; j++)
//                 {
//                     printf("%f ", Bs[0][0][i][j] );
//                     printf("+ %f i",Bs[0][1][i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("BS[1]:\n");
//             for (int i = 0; i < BLOCK_SIZE_K; i++)
//             {
//                 for (int j = 0; j < BLOCK_SIZE_N; j++)
//                 {
//                     printf("%f ", Bs[1][0][i][j] );
//                     printf("+ %f i",Bs[1][1][i][j]);
//                 }
//                 printf("\n");
//             }
//         }
// #endif
// #pragma unroll
//         for (int j = 0; j < BLOCK_SIZE_K-1; ++j)
//         {
//             // load next tile from shared mem to register
//             //next_stage_flag = (j == BLOCK_SIZE_K - 1) ? load_stage_idx : write_stage_idx;
//             // next_stage_flag = (j == BLOCK_SIZE_K - 1) ? write_stage_idx : load_stage_idx;
//             //next_stage_flag = load_stage_idx;
//             //  load A from shared memory to register
//             // if (j != BLOCK_SIZE_K - 1)
//             // {
//             // for (int i = 0; i < THREAD_SIZE_Y; i++)
//             // {
//             //     frag_a[(j + 1) % 2][0][i] = As[load_stage_idx][0][(j + 1) % BLOCK_SIZE_K][i + sa2reg_idx];
//             //     frag_a[(j + 1) % 2][1][i] = As[load_stage_idx][1][(j + 1) % BLOCK_SIZE_K][i + sa2reg_idx];
//             // }
//             // // load B from shared memory to register
//             // for (int i = 0; i < THREAD_SIZE_X; i++)
//             // {
//             //     frag_b[(j + 1) % 2][0][i] = Bs[load_stage_idx][0][(j + 1) % BLOCK_SIZE_K][i + sb2reg_idx];
//             //     frag_b[(j + 1) % 2][1][i] = Bs[load_stage_idx][1][(j + 1) % BLOCK_SIZE_K][i + sb2reg_idx];
//             // }
//             // }
//                 int jp1mod2 = (j + 1) % 2;
//                 int jp1modK = (j + 1) % BLOCK_SIZE_K;
                
//                // load A from shared memory to register
//                FETCH_DOUBLE2(frag_a[jp1mod2][0][0]) = FETCH_DOUBLE2(As[load_stage_idx][0][jp1modK][a_tile_index]);
//                FETCH_DOUBLE2(frag_a[jp1mod2][1][0]) = FETCH_DOUBLE2(As[load_stage_idx][1][jp1modK][a_tile_index]);
           
//                FETCH_DOUBLE2(frag_a[jp1mod2][0][2]) = FETCH_DOUBLE2(As[load_stage_idx][0][jp1modK][a_tile_index + 16]);
//                FETCH_DOUBLE2(frag_a[jp1mod2][1][2]) = FETCH_DOUBLE2(As[load_stage_idx][1][jp1modK][a_tile_index + 16]);
               
//                // load B from shared memory to register
//                FETCH_DOUBLE2(frag_b[jp1mod2][0][0]) = FETCH_DOUBLE2(Bs[load_stage_idx][0][jp1modK][b_tile_index]);
//                FETCH_DOUBLE2(frag_b[jp1mod2][1][0]) = FETCH_DOUBLE2(Bs[load_stage_idx][1][jp1modK][b_tile_index]);
           
//                FETCH_DOUBLE2(frag_b[jp1mod2][0][2]) = FETCH_DOUBLE2(Bs[load_stage_idx][0][jp1modK][b_tile_index + 32]);
//                FETCH_DOUBLE2(frag_b[jp1mod2][1][2]) = FETCH_DOUBLE2(Bs[load_stage_idx][1][jp1modK][b_tile_index + 32]);


// #ifdef PRINT_REG
//             if (bx == 0 && by == 0 && tx == 0 && ty == 0)
//             {
//                 printf("load_stage_idx:%d\n", load_stage_idx);
//                 printf("j == %d\n", j);
//                 printf("regA[0]:\n");
//                 for (int g = 0; g < THREAD_SIZE_Y; g++)
//                 {
//                     printf("%f + %fi ", frag_a[0][0][g], frag_a[0][1][g]);
//                 }
//                 printf("\n");
//                 printf("regA[1]:\n");
//                 for (int g = 0; g < THREAD_SIZE_Y; g++)
//                 {
//                     printf("%f + %fi ", frag_a[1][0][g], frag_a[1][1][g]);
//                 }
//                 printf("\n");
//                 printf("regB[0]:\n");
//                 for (int g = 0; g < THREAD_SIZE_X; g++)
//                 {
//                     printf("%f + %fi ", frag_b[0][0][g], frag_b[0][1][g]);
//                 }
//                 printf("\n");
//                 printf("regB[1]:\n");
//                 for (int g = 0; g < THREAD_SIZE_X; g++)
//                 {
//                     printf("%f + %fi ", frag_b[1][0][g], frag_b[1][1][g]);
//                 }
//                 printf("\n");
//             }
// #endif
//             // compute C THREAD_SIZE_X x THREAD_SIZE_Y
//             // complex temp;

// #pragma unroll
//             for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
//             {
// #pragma unroll
//                 for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
//                 {
//                     // printf("4444444444444\n");
//                     accum[thread_y][thread_x].r += frag_a[j%2][0][thread_y] * frag_b[j%2][0][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].r);

//                     accum[thread_y][thread_x].r -= frag_a[j%2][1][thread_y] * frag_b[j%2][1][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].r);

//                     accum[thread_y][thread_x].i += frag_a[j%2][0][thread_y] * frag_b[j%2][1][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].i);

//                     accum[thread_y][thread_x].i += frag_a[j%2][1][thread_y] * frag_b[j%2][0][thread_x];

//                     //printf("%f ",accum[thread_y][thread_x].i);

//                     //printf("\n");
//                     //if (bx == 0 && by == 0 && tx == 0 && ty == 0){
//                         // printf("%f %f %f %f %f %f %f %f\n",
//                         // frag_a[j][0][thread_y],frag_b[j][0][thread_x],
//                         // frag_a[j][1][thread_y],frag_b[j][1][thread_x],
//                         // frag_a[j][0][thread_y],frag_b[j][1][thread_x],
//                         // frag_a[j][1][thread_y],frag_b[j][0][thread_x]
//                         // );
//                     // printf("%f , %f %f %f",frag_a[j][0][thread_y] * frag_b[j][0][thread_x],
//                     // frag_a[j][1][thread_y] * frag_b[j][1][thread_x],
//                     // frag_a[j][0][thread_y] * frag_b[j][1][thread_x],
//                     // frag_a[j][1][thread_y] * frag_b[j][0][thread_x]
//                     // );
                    
//                         //printf("%f + %f i ",accum[thread_y][thread_x].r,accum[thread_y][thread_x].i);
//                     //}
//                 }
//                 // if (bx == 0 && by == 0 && tx == 0 && ty == 0) 
//                 //printf("\n");
//             }
//             //if (bx == 0 && by == 0 && tx == 0 && ty == 0) printf("\n");
//         }

//         if (tile_idx < K)
//         {
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / A_TILE_ROW_STRIDE;
//                 As[write_stage_idx][0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].r;
//                 As[write_stage_idx][1][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].i;
//             }
// // printf("222222222222\n");
// //  load B from global memory to shared memory
// #pragma unroll
//             for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//             {
//                 int ldg_index = i / B_TILE_ROW_STRIDE;
//                 Bs[write_stage_idx][0][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].r;
//                 Bs[write_stage_idx][1][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].i;
//             }
//             // use double buffer, only need one sync
//             __syncthreads();
//             // prefetch next reg
//             // for (int i = 0; i < THREAD_SIZE_Y; i++)
//             // {
//             //     frag_a[0][0][i] = As[write_stage_idx][0][0][i + sa2reg_idx];
//             //     frag_a[0][1][i] = As[write_stage_idx][0][1][i + sa2reg_idx];
//             // }
//             // // load B from shared memory to register
//             // for (int i = 0; i < THREAD_SIZE_X; i++)
//             // {
//             //     frag_b[0][0][i] = Bs[write_stage_idx][0][0][i + sb2reg_idx];
//             //     frag_b[0][1][i] = Bs[write_stage_idx][0][1][i + sb2reg_idx];
//             // }

//             // switch
//             write_stage_idx ^= 1;
//         }
//         //int sa2reg_idx = ty * THREAD_SIZE_Y;
//         // for (int i = 0; i < THREAD_SIZE_Y; i++)
//         // {
//         //     frag_a[0][0][i] = As[load_stage_idx^1][0][0][i + sa2reg_idx];
//         //     frag_a[0][1][i] = As[load_stage_idx^1][1][0][i + sa2reg_idx];
//         // }
     
//         // for (int i = 0; i < THREAD_SIZE_X; i++)
//         // {
//         //     frag_b[0][0][i] = Bs[load_stage_idx^1][0][0][i + sb2reg_idx];
//         //     frag_b[0][1][i] = Bs[load_stage_idx^1][1][0][i + sb2reg_idx];
//         // }
//                // load A from shared memory to register
//                FETCH_DOUBLE2(frag_a[0][0][0]) = FETCH_DOUBLE2(As[load_stage_idx^1][0][0][a_tile_index]);
//                FETCH_DOUBLE2(frag_a[0][1][0]) = FETCH_DOUBLE2(As[load_stage_idx^1][1][0][a_tile_index]);
           
//                FETCH_DOUBLE2(frag_a[0][0][2]) = FETCH_DOUBLE2(As[load_stage_idx^1][0][0][a_tile_index + 16]);
//                FETCH_DOUBLE2(frag_a[0][1][2]) = FETCH_DOUBLE2(As[load_stage_idx^1][1][0][a_tile_index + 16]);
               
//                // load B from shared0 to register
//                FETCH_DOUBLE2(frag_b[0][0][0]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][0][0][b_tile_index]);
//                FETCH_DOUBLE2(frag_b[0][1][0]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][1][0][b_tile_index]);
           
//                FETCH_DOUBLE2(frag_b[0][0][2]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][0][0][b_tile_index + 32]);
//                FETCH_DOUBLE2(frag_b[0][1][2]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][1][0][b_tile_index + 32]);

// #pragma unroll
//         for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
//         {
// #pragma unroll
//             for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
//             {
//                 // printf("4444444444444\n");
//                 accum[thread_y][thread_x].r += frag_a[1][0][thread_y] * frag_b[1][0][thread_x];

//                 //printf("%f ",accum[thread_y][thread_x].r);

//                 accum[thread_y][thread_x].r -= frag_a[1][1][thread_y] * frag_b[1][1][thread_x];

//                 //printf("%f ",accum[thread_y][thread_x].r);

//                 accum[thread_y][thread_x].i += frag_a[1][0][thread_y] * frag_b[1][1][thread_x];

//                 //printf("%f ",accum[thread_y][thread_x].i);

//                 accum[thread_y][thread_x].i += frag_a[1][1][thread_y] * frag_b[1][0][thread_x];
//             }
//         }
//     } while (tile_idx < K);

// // store back to C
// // #pragma unroll
// //     for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
// //     {
// // #pragma unroll
// //         for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
// //         {
// //             C[OFFSET(
// //                 BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
// //                 BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
// //                 N)] = accum[thread_y][thread_x];
// //                 // if (bx == 0 && by == 0 && tx == 0 && ty == 0){
// //                 //     printf("%f + %f i",accum[thread_y][thread_x].r,accum[thread_y][thread_x].i);
// //                 // }
// //         }
// //         //if (bx == 0 && by == 0 && tx == 0 && ty == 0) printf("\n");
// //     }

//     const int c_block_row = a_tile_index;
//     const int c_block_col = b_tile_index;

//     //store C00 block
//     // for(int i=0; i<2; i++){
//     //   FETCH_DOUBLE2(C[OFFSET(
//     //     BLOCK_SIZE_M * by + c_block_row + i,
//     //     BLOCK_SIZE_N * bx + c_block_col,
//     //     N)]) = FETCH_DOUBLE2(accum[i][0]);
//     // }

//     // //store C01 block
//     // for(int i=0; i<2; i++){
//     //   FETCH_DOUBLE2(C[OFFSET(
//     //     BLOCK_SIZE_M * by + c_block_row + i,
//     //     BLOCK_SIZE_N * bx + c_block_col + 64,
//     //     N)]) = FETCH_DOUBLE2(accum[i][4]);
//     // }
//     // //store C10 block
//     // for(int i=0; i<2; i++){
//     //   FETCH_DOUBLE2(C[OFFSET(
//     //     BLOCK_SIZE_M * by + c_block_row + 64 + i,
//     //     BLOCK_SIZE_N * bx + c_block_col,
//     //     N)]) = FETCH_DOUBLE2(accum[i+4][0]);
//     // }
//     // //store C11 block
//     // for(int i=0; i<2; i++){
//     //   FETCH_DOUBLE2(C[OFFSET(
//     //     BLOCK_SIZE_M * by + c_block_row + 64 + i,
//     //     BLOCK_SIZE_N * bx + c_block_col + 64,
//     //     N)]) = FETCH_DOUBLE2(accum[i+4][4]);
//     // }


//     // //store c00 blcok
//     for(int i = 0;i<2;i++)
//     {
//         for(int j = 0;j<2;j++)
//         {
//             C[OFFSET(
//             BLOCK_SIZE_M * by + c_block_row + i,
//             BLOCK_SIZE_N * bx + c_block_col + j,
//             N)] = accum[i][j];
//         }
//     }
//         //store c01 blcok
//         for(int i = 0;i<2;i++)
//         {
//             for(int j = 2;j<4;j++)
//             {
//                 C[OFFSET(
//                 BLOCK_SIZE_M * by + c_block_row + i,
//                 BLOCK_SIZE_N * bx + c_block_col + j,
//                 N)] = accum[i][j];
//             }
//         }
//             //store c10 blcok
//     for(int i = 2;i<4;i++)
//     {
//         for(int j = 0;j<2;j++)
//         {
//             C[OFFSET(
//             BLOCK_SIZE_M * by + c_block_row + i,
//             BLOCK_SIZE_N * bx + c_block_col + j,
//             N)] = accum[i][j];
//         }
//     }
//         //store c11 blcok
//         for(int i = 2;i<4;i++)
//         {
//             for(int j = 2;j<4;j++)
//             {
//                 C[OFFSET(
//                 BLOCK_SIZE_M * by + c_block_row + i,
//                 BLOCK_SIZE_N * bx + c_block_col + j,
//                 N)] = accum[i][j];
//             }
//         }



// }
// int main(int argc, char **argv)
// {
//     if (argc != 4)
//     {
//         printf("usage: ./main [M] [K] [N]\n");
//         exit(0);
//     }

//     // cudaSharedMemConfig *pconfig;
//     // cudaDeviceGetSharedMemConfig(pconfig);
//     // printf("%d\n",pconfig);

//     size_t M = atoi(argv[1]);
//     size_t K = atoi(argv[2]);
//     size_t N = atoi(argv[3]);

//     size_t bytes_A = sizeof(complex) * M * K;
//     size_t bytes_B = sizeof(complex) * K * N;
//     size_t bytes_C = sizeof(complex) * M * N;
//     complex *h_A = (complex *)malloc(bytes_A);
//     complex *h_B = (complex *)malloc(bytes_B);
//     complex *h_C = (complex *)malloc(bytes_C);
//     complex *h_C1 = (complex *)malloc(bytes_C);

//     complex *d_A;
//     complex *d_B;
//     complex *d_C;

//     checkCudaErrors(cudaMalloc(&d_A, bytes_A));
//     checkCudaErrors(cudaMalloc(&d_B, bytes_B));
//     checkCudaErrors(cudaMalloc(&d_C, bytes_C));
//     double msecPerMatrixMul[2] = {0, 0};
//     double gigaFlops[2] = {0, 0};
//     //存疑
//     double flopsPerMatrixMul = 8*M*N*K + 12*M*N;

//     // const int BLOCK_SIZE_M = 16;
//     // const int BLOCK_SIZE_K = 16;
//     // const int BLOCK_SIZE_N = 16;
//     // const int THREAD_SIZE_X = 2;
//     // const int THREAD_SIZE_Y = 2;
//     // const int BLOCK_SIZE_M = 4;
//     // const int BLOCK_SIZE_K = 4;
//     // const int BLOCK_SIZE_N = 4;
//     // const int THREAD_SIZE_X = 2;
//     // const int THREAD_SIZE_Y = 2;
//     // const int BLOCK_SIZE_M = 32;
//     // const int BLOCK_SIZE_K = 4;
//     // const int BLOCK_SIZE_N = 32;
//     // const int THREAD_SIZE_X = 2;
//     // const int THREAD_SIZE_Y = 2;
//     // const int BLOCK_SIZE_M = 16;
//     // const int BLOCK_SIZE_K = 8;
//     // const int BLOCK_SIZE_N = 16;
//     // const int THREAD_SIZE_X = 2;
//     // const int THREAD_SIZE_Y = 2;
//     const int BLOCK_SIZE_M = 32;
//     const int BLOCK_SIZE_K = 8;
//     const int BLOCK_SIZE_N = 64;
//     const int THREAD_SIZE_X = 4;
//     const int THREAD_SIZE_Y = 4;
//     // const int BLOCK_SIZE_M = 64;
//     // const int BLOCK_SIZE_K = 8;
//     // const int BLOCK_SIZE_N = 32;
//     // const int THREAD_SIZE_X = 4;
//     // const int THREAD_SIZE_Y = 4;
//     // const int BLOCK_SIZE_M = 32;
//     // const int BLOCK_SIZE_K = 8;
//     // const int BLOCK_SIZE_N = 32;
//     // const int THREAD_SIZE_X = 4;
//     // const int THREAD_SIZE_Y = 4;
//     // const int BLOCK_SIZE_M = 32;
//     // const int BLOCK_SIZE_K = 16;
//     // const int BLOCK_SIZE_N = 32;
//     // const int THREAD_SIZE_X = 4;
//     // const int THREAD_SIZE_Y = 4;
//     const bool ENABLE_DOUBLE_BUFFER = false;
//     int k_block = K / BLOCK_SIZE_K;
//     int stride = 2;

//     // 生成A的数据
// #if 1
//     // for (int i = 0; i < M * K; i++)
//     // {
//     //     int row = (i / K);
//     //     int col = (i % K);
//     //     int row_block = row / BLOCK_SIZE_M;
//     //     int col_block = col / BLOCK_SIZE_K;
//     //     if ((row_block * k_block + col_block) % stride == 0)
//     //     {
//     //         h_A[i].r = 1;
//     //         h_A[i].i = 2;
//     //     }
//     //     else
//     //     {
//     //         h_A[i].r = 3;
//     //         h_A[i].i = 4;
//     //     }
//     // }

//     // // 生成B的数据
//     // for (int i = 0; i < K * N; i++)
//     // {
//     //     if (i >= K * N / 2)
//     //     {
//     //         h_B[i].r = 5;
//     //         h_B[i].i = 6;
//     //     }
//     //     else
//     //     {
//     //         h_B[i].r = 7;
//     //         h_B[i].i = 8;
//     //     }
//     // }
//     for (int i = 0; i < M * K; i++)
//     {
//         int row = (i / K);
//         int col = (i % K);
//         int row_block = row / BLOCK_SIZE_M;
//         int col_block = col / BLOCK_SIZE_K;
        
//         h_A[i].r = double(i);
//         h_A[i].i = double(i);
//             // srand((unsigned)time(NULL));
//             // h_A[i].r = double(rand())/RAND_MAX;
//             // h_A[i].i = double(rand())/RAND_MAX;
        

//     }

//     // 生成B的数据
//     for (int i = K * N; i >0 ; i--)
//     {
//         // if (i >= K * N / 2)
//         // {
//         //     h_B[i].r = 5;
//         //     h_B[i].i = 6;
//         // }
//         // else
//         // {
//         //     h_B[i].r = 7;
//         //     h_B[i].i = 8;
//         // }
//         // srand((unsigned)time(NULL));
//         // h_B[i].r = double(rand())/RAND_MAX;
//         // h_B[i].i = double(rand())/RAND_MAX;

//         h_B[K * N-i].r = double(i);
//         h_B[K * N-i].i = double(i);
//     }
// #endif
// #ifdef PRINT_MAT
//     // printf("A:\n");
//     // for (int i = 0; i < M; i++)
//     // {
//     //     for (int j = 0; j < K; j++)
//     //     {
//     //         printf("%f + %f i", h_A[OFFSET(i, j, K)].r, h_A[OFFSET(i, j, K)].i);
//     //     }
//     //     printf("\n");
//     // }
//     // printf("B:\n");
//     // for (int i = 0; i < K; i++)
//     // {
//     //     for (int j = 0; j < N; j++)
//     //     {
//     //         printf("%f + %f i", h_B[OFFSET(i, j, N)].r, h_B[OFFSET(i, j, N)].i);
//     //     }
//     //     printf("\n");
//     // }
// #endif
//     // for( int i = 0; i < M * K; i++ ) {
//     //     h_A->i = i+3;
//     //     h_A->r = i+3;
//     // }

//     // // 生成B的数据
//     // for( int i = 0; i < K * N; i++ ) {
//     //     h_B->i = i;
//     //     h_B->r = i;
//     // }

//     checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

//     cudaEvent_t start, stop;
//     checkCudaErrors(cudaEventCreate(&start));
//     checkCudaErrors(cudaEventCreate(&stop));
//     float msecTotal = 0;
//     int nIter = 100;

//     checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaEventRecord(start));
//     for (int run = 0; run < nIter; run++)
//     {
//         dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
//         dim3 dimGrid((N - 1) / BLOCK_SIZE_N + 1, (M - 1) / BLOCK_SIZE_M + 1);
//         Zgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
//             <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
//     }
//     checkCudaErrors(cudaEventRecord(stop));
//     checkCudaErrors(cudaEventSynchronize(stop));
//     checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

//     checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

//     msecPerMatrixMul[0] = msecTotal / nIter;
//     gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
//     printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
//            gigaFlops[0],
//            msecPerMatrixMul[0],
//            flopsPerMatrixMul);

//     // cublas

//     cublasHandle_t blas_handle;
//     cublasCreate(&blas_handle);
//     // float alpha = 1.0;
//     // float beta = 0;
//     checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaEventRecord(start));
//     for (int run = 0; run < nIter; run++)
//     {
//         cublasZgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
//                     M, N, K, (cuDoubleComplex *)&zone,
//                     (cuDoubleComplex *)d_A, K, (cuDoubleComplex *)d_B, N, (cuDoubleComplex *)&zero, (cuDoubleComplex *)d_C, N);
//     }
//     checkCudaErrors(cudaEventRecord(stop));
//     checkCudaErrors(cudaEventSynchronize(stop));
//     checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

//     checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

//     msecPerMatrixMul[1] = msecTotal / nIter;
//     gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
//     printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
//            gigaFlops[1],
//            msecPerMatrixMul[1],
//            flopsPerMatrixMul);

//     cublasDestroy(blas_handle);

//     //transpose
//     // for(int i = 0;i<M;i++)
//     // {
//     //     for(int j = 0;j<N;j++)
//     //     {
//     //         complex temp;
//     //         temp =  h_C[OFFSET(i, j, N)];
//     //         h_C[OFFSET(i, j, N)] =  h_C[OFFSET(j, i, N)];
//     //         h_C[OFFSET(j, i, N)] = temp;
           
//     //     }
//     // }
//     double eps = 1.e-6; // machine zero
//     bool correct = true;
//     for (int i = 0; i < M * N; i++)
//     {
//         int row = i / N;
//         int col = i % N;
//         double abs_err_r = fabs(h_C[i].r - h_C1[col * M + row].r);
//         double abs_err_i = fabs(h_C[i].i - h_C1[col * M + row].i);
//         // double abs_err_r = fabs(h_C[i].r - h_C1[i].r);
//         // double abs_err_i = fabs(h_C[i].i - h_C1[i].i);
//         double dot_length = M;
//         double abs_val_r = fabs(h_C[i].r);
//         double abs_val_i = fabs(h_C[i].i);
//         double rel_err_r = abs_err_r / abs_val_r / dot_length;
//         double rel_err_i = abs_err_i / abs_val_i / dot_length;
//         if (rel_err_r > eps || rel_err_i > eps)
//         {
//             printf("Error! Matrix[%05d][%05d]=%.8f, ref_r=%.8f error term is > %E\n",
//                    row, col, h_C[i].r, h_C1[col * M + row].r, eps);
//             correct = false;
//             break;
//         }
//     }

//     printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
//     printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

// #ifdef PRINT_MAT
//     printf("my gemm:\n");
//     for (int i = 0; i < M; i++)
//     {
//         for (int j = 0; j < N; j++)
//         {
//             printf("%f + %f i", h_C[OFFSET(i, j, N)].r, h_C[OFFSET(i, j, N)].i);
//         }
//         printf("\n");
//     }

//     printf("cublas:\n");
//     for (int i = 0; i < M; i++)
//     {
//         for (int j = 0; j < N; j++)
//         {
//             printf("%f + %f i", h_C1[OFFSET(j, i, N)].r, h_C1[OFFSET(j, i, N)].i);
//         }
//         printf("\n");
//     }
// #endif
//     // Free Memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     free(h_A);
//     free(h_B);
//     free(h_C);
//     free(h_C1);
// }




template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
    const int THREAD_SIZE_X,        // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void Zgemm(
    complex *__restrict__ A,
    complex *__restrict__ B,
    complex *__restrict__ C,
    const int M,
    const int N,
    const int K)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;//16
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;//8
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    //the first 2 is double buffer the second is seperation of complex
    __shared__ double As[2][2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ double Bs[2][2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // registers for C
    complex accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    
#pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; i++)
    {
#pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; j++)
        {
            accum[i][j].r = 0.0;
            accum[i][j].i = 0.0;
        }
    }
    // registers for A and B
    double  frag_a[2][2][THREAD_SIZE_Y];
    double  frag_b[2][2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;//2
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK;//4
    complex ldg_a_reg[ldg_num_a];
    complex ldg_b_reg[ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K;//8
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N;//64

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[BLOCK_SIZE_N * bx];

    
    //load double2
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index =  warp_id/2*8 + lane_id/8*2; //warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
    const int b_tile_index =  warp_id%2*16 + lane_id%8*2; //(lane_id % 16) * 4; // (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;


// transfer first tile from global mem to shared mem
//  load A from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)//16
    {
        int ldg_index = i / A_TILE_ROW_STRIDE;
        ldg_a_reg[ldg_index] = A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL,           // col
            K)];
        As[0][0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].r;
        As[0][1][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].i;
    }
    // printf("000000000\n");
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)//2
    {
        int ldg_index = i / B_TILE_ROW_STRIDE;
        ldg_b_reg[ldg_index] = B[OFFSET(
            B_TILE_ROW_START + i, // row
            B_TILE_COL,           // col
            N)];
        Bs[0][0][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].r;
        Bs[0][1][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].i;
    }

    __syncthreads();

    // // load A from shared memory to register
    // int sa2reg_idx = ty * THREAD_SIZE_Y;
    // for (int i = 0; i < THREAD_SIZE_Y; i++)
    // {
    //     frag_a[0][0][i] = As[0][0][0][i + sa2reg_idx];
    //     frag_a[0][1][i] = As[0][1][0][i + sa2reg_idx];
    // }
    // // load B from shared memory to register
    // int sb2reg_idx = tx * THREAD_SIZE_X;
    // for (int i = 0; i < THREAD_SIZE_X; i++)
    // {
    //     frag_b[0][0][i] = Bs[0][0][0][i + sb2reg_idx];
    //     frag_b[0][1][i] = Bs[0][1][0][i + sb2reg_idx];
    // }
//     // load A from shared memory to register
    FETCH_DOUBLE2(frag_a[0][0][0]) = FETCH_DOUBLE2(As[0][0][0][a_tile_index]);
    FETCH_DOUBLE2(frag_a[0][1][0]) = FETCH_DOUBLE2(As[0][1][0][a_tile_index]);

    FETCH_DOUBLE2(frag_a[0][0][2]) = FETCH_DOUBLE2(As[0][0][0][a_tile_index + 16]);
    FETCH_DOUBLE2(frag_a[0][1][2]) = FETCH_DOUBLE2(As[0][1][0][a_tile_index + 16]);
    
    // load B from shared memory to register
    FETCH_DOUBLE2(frag_b[0][0][0]) = FETCH_DOUBLE2(Bs[0][0][0][b_tile_index]);
    FETCH_DOUBLE2(frag_b[0][1][0]) = FETCH_DOUBLE2(Bs[0][1][0][b_tile_index]);

    FETCH_DOUBLE2(frag_b[0][0][2]) = FETCH_DOUBLE2(Bs[0][0][0][b_tile_index + 32]);
    FETCH_DOUBLE2(frag_b[0][1][2]) = FETCH_DOUBLE2(Bs[0][1][0][b_tile_index + 32]);

    int write_stage_idx = 1;
    int tile_idx = 0;
    do
    {
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if (tile_idx < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE;
                ldg_a_reg[ldg_index] = A[OFFSET(
                    A_TILE_ROW_START + i,  // row
                    A_TILE_COL + tile_idx, // col
                    K)];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                int ldg_index = i / B_TILE_ROW_STRIDE;
                ldg_b_reg[ldg_index] = B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL,                      // col
                    N)];
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;
#ifdef PRINT_sharedmem

        if (bx == 0 && by == 0 && tx == 0 && ty == 0)
        {
            printf("%d %d %d %d\n", bx, by, tx, ty);
            printf("tileidx = %d\n", tile_idx);
            printf("AS[0]:\n");
            for (int i = 0; i < BLOCK_SIZE_K; i++)
            {
                for (int j = 0; j < BLOCK_SIZE_M; j++)
                {
                    printf("%f ", As[0][0][i][j] );
                    printf("+ %f i",As[0][1][i][j]);
                }
                printf("\n");
            }
            printf("AS[1]:\n");
            for (int i = 0; i < BLOCK_SIZE_K; i++)
            {
                for (int j = 0; j < BLOCK_SIZE_M; j++)
                {
                    printf("%f ", As[1][0][i][j] );
                    printf("+ %f i",As[1][1][i][j]);
                }
                printf("\n");
            }
            printf("Bs[0]:  \n");
            for (int i = 0; i < BLOCK_SIZE_K; i++)
            {
                for (int j = 0; j < BLOCK_SIZE_N; j++)
                {
                    printf("%f ", Bs[0][0][i][j] );
                    printf("+ %f i",Bs[0][1][i][j]);
                }
                printf("\n");
            }
            printf("BS[1]:\n");
            for (int i = 0; i < BLOCK_SIZE_K; i++)
            {
                for (int j = 0; j < BLOCK_SIZE_N; j++)
                {
                    printf("%f ", Bs[1][0][i][j] );
                    printf("+ %f i",Bs[1][1][i][j]);
                }
                printf("\n");
            }
        }
#endif
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K-1; ++j)
        {
            // load next tile from shared mem to register
            //next_stage_flag = (j == BLOCK_SIZE_K - 1) ? load_stage_idx : write_stage_idx;
            // next_stage_flag = (j == BLOCK_SIZE_K - 1) ? write_stage_idx : load_stage_idx;
            //next_stage_flag = load_stage_idx;
            //  load A from shared memory to register
            // if (j != BLOCK_SIZE_K - 1)
            // {
            // for (int i = 0; i < THREAD_SIZE_Y; i++)
            // {
            //     frag_a[(j + 1) % 2][0][i] = As[load_stage_idx][0][(j + 1) % BLOCK_SIZE_K][i + sa2reg_idx];
            //     frag_a[(j + 1) % 2][1][i] = As[load_stage_idx][1][(j + 1) % BLOCK_SIZE_K][i + sa2reg_idx];
            // }
            // // load B from shared memory to register
            // for (int i = 0; i < THREAD_SIZE_X; i++)
            // {
            //     frag_b[(j + 1) % 2][0][i] = Bs[load_stage_idx][0][(j + 1) % BLOCK_SIZE_K][i + sb2reg_idx];
            //     frag_b[(j + 1) % 2][1][i] = Bs[load_stage_idx][1][(j + 1) % BLOCK_SIZE_K][i + sb2reg_idx];
            // }
                int jp1mod2 = (j + 1) % 2;
                int jp1modK = (j + 1) % BLOCK_SIZE_K;
                
               // load A from shared memory to register
               FETCH_DOUBLE2(frag_a[jp1mod2][0][0]) = FETCH_DOUBLE2(As[load_stage_idx][0][jp1modK][a_tile_index]);
               FETCH_DOUBLE2(frag_a[jp1mod2][1][0]) = FETCH_DOUBLE2(As[load_stage_idx][1][jp1modK][a_tile_index]);
           
               FETCH_DOUBLE2(frag_a[jp1mod2][0][2]) = FETCH_DOUBLE2(As[load_stage_idx][0][jp1modK][a_tile_index + 16]);
               FETCH_DOUBLE2(frag_a[jp1mod2][1][2]) = FETCH_DOUBLE2(As[load_stage_idx][1][jp1modK][a_tile_index + 16]);
               
               // load B from shared memory to register
               FETCH_DOUBLE2(frag_b[jp1mod2][0][0]) = FETCH_DOUBLE2(Bs[load_stage_idx][0][jp1modK][b_tile_index]);
               FETCH_DOUBLE2(frag_b[jp1mod2][1][0]) = FETCH_DOUBLE2(Bs[load_stage_idx][1][jp1modK][b_tile_index]);
           
               FETCH_DOUBLE2(frag_b[jp1mod2][0][2]) = FETCH_DOUBLE2(Bs[load_stage_idx][0][jp1modK][b_tile_index + 32]);
               FETCH_DOUBLE2(frag_b[jp1mod2][1][2]) = FETCH_DOUBLE2(Bs[load_stage_idx][1][jp1modK][b_tile_index + 32]);

            // }
#ifdef PRINT_REG
            if (bx == 0 && by == 0 && tx == 0 && ty == 0)
            {
                printf("load_stage_idx:%d\n", load_stage_idx);
                printf("j == %d\n", j);
                printf("regA[0]:\n");
                for (int g = 0; g < THREAD_SIZE_Y; g++)
                {
                    printf("%f + %fi ", frag_a[0][0][g], frag_a[0][1][g]);
                }
                printf("\n");
                printf("regA[1]:\n");
                for (int g = 0; g < THREAD_SIZE_Y; g++)
                {
                    printf("%f + %fi ", frag_a[1][0][g], frag_a[1][1][g]);
                }
                printf("\n");
                printf("regB[0]:\n");
                for (int g = 0; g < THREAD_SIZE_X; g++)
                {
                    printf("%f + %fi ", frag_b[0][0][g], frag_b[0][1][g]);
                }
                printf("\n");
                printf("regB[1]:\n");
                for (int g = 0; g < THREAD_SIZE_X; g++)
                {
                    printf("%f + %fi ", frag_b[1][0][g], frag_b[1][1][g]);
                }
                printf("\n");
            }
#endif
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            // complex temp;

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                {
                    // printf("4444444444444\n");
                    accum[thread_y][thread_x].r += frag_a[j%2][0][thread_y] * frag_b[j%2][0][thread_x];

                    //printf("%f ",accum[thread_y][thread_x].r);

                    accum[thread_y][thread_x].r -= frag_a[j%2][1][thread_y] * frag_b[j%2][1][thread_x];

                    //printf("%f ",accum[thread_y][thread_x].r);

                    accum[thread_y][thread_x].i += frag_a[j%2][0][thread_y] * frag_b[j%2][1][thread_x];

                    //printf("%f ",accum[thread_y][thread_x].i);

                    accum[thread_y][thread_x].i += frag_a[j%2][1][thread_y] * frag_b[j%2][0][thread_x];

                    //printf("%f ",accum[thread_y][thread_x].i);

                    //printf("\n");
                    //if (bx == 0 && by == 0 && tx == 0 && ty == 0){
                        // printf("%f %f %f %f %f %f %f %f\n",
                        // frag_a[j][0][thread_y],frag_b[j][0][thread_x],
                        // frag_a[j][1][thread_y],frag_b[j][1][thread_x],
                        // frag_a[j][0][thread_y],frag_b[j][1][thread_x],
                        // frag_a[j][1][thread_y],frag_b[j][0][thread_x]
                        // );
                    // printf("%f , %f %f %f",frag_a[j][0][thread_y] * frag_b[j][0][thread_x],
                    // frag_a[j][1][thread_y] * frag_b[j][1][thread_x],
                    // frag_a[j][0][thread_y] * frag_b[j][1][thread_x],
                    // frag_a[j][1][thread_y] * frag_b[j][0][thread_x]
                    // );
                    
                        //printf("%f + %f i ",accum[thread_y][thread_x].r,accum[thread_y][thread_x].i);
                    //}
                }
                // if (bx == 0 && by == 0 && tx == 0 && ty == 0) 
                //printf("\n");
            }
            //if (bx == 0 && by == 0 && tx == 0 && ty == 0) printf("\n");
        }

        if (tile_idx < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE;
                As[write_stage_idx][0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].r;
                As[write_stage_idx][1][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index].i;
            }
// printf("222222222222\n");
//  load B from global memory to shared memory
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                int ldg_index = i / B_TILE_ROW_STRIDE;
                Bs[write_stage_idx][0][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].r;
                Bs[write_stage_idx][1][B_TILE_ROW_START + i][B_TILE_COL] = ldg_b_reg[ldg_index].i;
            }
            // use double buffer, only need one sync
            __syncthreads();
            // prefetch next reg
            // for (int i = 0; i < THREAD_SIZE_Y; i++)
            // {
            //     frag_a[0][0][i] = As[write_stage_idx][0][0][i + sa2reg_idx];
            //     frag_a[0][1][i] = As[write_stage_idx][0][1][i + sa2reg_idx];
            // }
            // // load B from shared memory to register
            // for (int i = 0; i < THREAD_SIZE_X; i++)
            // {
            //     frag_b[0][0][i] = Bs[write_stage_idx][0][0][i + sb2reg_idx];
            //     frag_b[0][1][i] = Bs[write_stage_idx][0][1][i + sb2reg_idx];
            // }

            // switch
            write_stage_idx ^= 1;
        }
        //int sa2reg_idx = ty * THREAD_SIZE_Y;
        // for (int i = 0; i < THREAD_SIZE_Y; i++)
        // {
        //     frag_a[0][0][i] = As[load_stage_idx^1][0][0][i + sa2reg_idx];
        //     frag_a[0][1][i] = As[load_stage_idx^1][1][0][i + sa2reg_idx];
        // }
     
        // for (int i = 0; i < THREAD_SIZE_X; i++)
        // {
        //     frag_b[0][0][i] = Bs[load_stage_idx^1][0][0][i + sb2reg_idx];
        //     frag_b[0][1][i] = Bs[load_stage_idx^1][1][0][i + sb2reg_idx];
        // }
                       // load A from shared memory to register
               FETCH_DOUBLE2(frag_a[0][0][0]) = FETCH_DOUBLE2(As[load_stage_idx^1][0][0][a_tile_index]);
               FETCH_DOUBLE2(frag_a[0][1][0]) = FETCH_DOUBLE2(As[load_stage_idx^1][1][0][a_tile_index]);
           
               FETCH_DOUBLE2(frag_a[0][0][2]) = FETCH_DOUBLE2(As[load_stage_idx^1][0][0][a_tile_index + 16]);
               FETCH_DOUBLE2(frag_a[0][1][2]) = FETCH_DOUBLE2(As[load_stage_idx^1][1][0][a_tile_index + 16]);
               
               // load B from shared0 to register
               FETCH_DOUBLE2(frag_b[0][0][0]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][0][0][b_tile_index]);
               FETCH_DOUBLE2(frag_b[0][1][0]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][1][0][b_tile_index]);
           
               FETCH_DOUBLE2(frag_b[0][0][2]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][0][0][b_tile_index + 32]);
               FETCH_DOUBLE2(frag_b[0][1][2]) = FETCH_DOUBLE2(Bs[load_stage_idx^1][1][0][b_tile_index + 32]);

#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
        {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
            {
                // printf("4444444444444\n");
                accum[thread_y][thread_x].r += frag_a[1][0][thread_y] * frag_b[1][0][thread_x];

                //printf("%f ",accum[thread_y][thread_x].r);

                accum[thread_y][thread_x].r -= frag_a[1][1][thread_y] * frag_b[1][1][thread_x];

                //printf("%f ",accum[thread_y][thread_x].r);

                accum[thread_y][thread_x].i += frag_a[1][0][thread_y] * frag_b[1][1][thread_x];

                //printf("%f ",accum[thread_y][thread_x].i);

                accum[thread_y][thread_x].i += frag_a[1][1][thread_y] * frag_b[1][0][thread_x];
            }
        }
    } while (tile_idx < K);

// store back to C
// #pragma unroll
//     for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
//     {
// #pragma unroll
//         for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
//         {
//             C[OFFSET(
//                 BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
//                 BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
//                 N)] = accum[thread_y][thread_x];
//                 // if (bx == 0 && by == 0 && tx == 0 && ty == 0){
//                 //     printf("%f + %f i",accum[thread_y][thread_x].r,accum[thread_y][thread_x].i);
//                 // }
//         }
//         //if (bx == 0 && by == 0 && tx == 0 && ty == 0) printf("\n");
//     }


    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;
    // //store c00 blcok
    for(int i = 0;i<2;i++)
    {
        for(int j = 0;j<2;j++)
        {
            C[OFFSET(
            BLOCK_SIZE_M * by + c_block_row + i,
            BLOCK_SIZE_N * bx + c_block_col + j,
            N)] = accum[i][j];
        }
    }
        //store c01 blcok
        for(int i = 0;i<2;i++)
        {
            for(int j = 2;j<4;j++)
            {
                C[OFFSET(
                BLOCK_SIZE_M * by + c_block_row + i,
                BLOCK_SIZE_N * bx + c_block_col + j+30,
                N)] = accum[i][j];
            }
        }
            //store c10 blcok
    for(int i = 2;i<4;i++)
    {
        for(int j = 0;j<2;j++)
        {
            C[OFFSET(
            BLOCK_SIZE_M * by + c_block_row + i+14,
            BLOCK_SIZE_N * bx + c_block_col + j,
            N)] = accum[i][j];
        }
    }
        //store c11 blcok
        for(int i = 2;i<4;i++)
        {
            for(int j = 2;j<4;j++)
            {
                C[OFFSET(
                BLOCK_SIZE_M * by + c_block_row + i+14,
                BLOCK_SIZE_N * bx + c_block_col + j+30,
                N)] = accum[i][j];
            }
        }


}
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }

    // cudaSharedMemConfig *pconfig;
    // cudaDeviceGetSharedMemConfig(pconfig);
    // printf("%d\n",pconfig);

    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes_A = sizeof(complex) * M * K;
    size_t bytes_B = sizeof(complex) * K * N;
    size_t bytes_C = sizeof(complex) * M * N;
    complex *h_A = (complex *)malloc(bytes_A);
    complex *h_B = (complex *)malloc(bytes_B);
    complex *h_C = (complex *)malloc(bytes_C);
    complex *h_C1 = (complex *)malloc(bytes_C);

    complex *d_A;
    complex *d_B;
    complex *d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    //存疑
    double flopsPerMatrixMul = 8*M*N*K + 12*M*N;

    // const int BLOCK_SIZE_M = 16;
    // const int BLOCK_SIZE_K = 16;
    // const int BLOCK_SIZE_N = 16;
    // const int THREAD_SIZE_X = 2;
    // const int THREAD_SIZE_Y = 2;
    // const int BLOCK_SIZE_M = 4;
    // const int BLOCK_SIZE_K = 4;
    // const int BLOCK_SIZE_N = 4;
    // const int THREAD_SIZE_X = 2;
    // const int THREAD_SIZE_Y = 2;
    // const int BLOCK_SIZE_M = 32;
    // const int BLOCK_SIZE_K = 4;
    // const int BLOCK_SIZE_N = 32;
    // const int THREAD_SIZE_X = 2;
    // const int THREAD_SIZE_Y = 2;
    // const int BLOCK_SIZE_M = 16;
    // const int BLOCK_SIZE_K = 8;
    // const int BLOCK_SIZE_N = 16;
    // const int THREAD_SIZE_X = 2;
    // const int THREAD_SIZE_Y = 2;
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    // const int BLOCK_SIZE_M = 64;
    // const int BLOCK_SIZE_K = 8;
    // const int BLOCK_SIZE_N = 32;
    // const int THREAD_SIZE_X = 4;
    // const int THREAD_SIZE_Y = 4;
    // const int BLOCK_SIZE_M = 32;
    // const int BLOCK_SIZE_K = 8;
    // const int BLOCK_SIZE_N = 32;
    // const int THREAD_SIZE_X = 4;
    // const int THREAD_SIZE_Y = 4;
    // const int BLOCK_SIZE_M = 32;
    // const int BLOCK_SIZE_K = 16;
    // const int BLOCK_SIZE_N = 32;
    // const int THREAD_SIZE_X = 4;
    // const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;
    int k_block = K / BLOCK_SIZE_K;
    int stride = 2;

    // 生成A的数据
#if 1
    // for (int i = 0; i < M * K; i++)
    // {
    //     int row = (i / K);
    //     int col = (i % K);
    //     int row_block = row / BLOCK_SIZE_M;
    //     int col_block = col / BLOCK_SIZE_K;
    //     if ((row_block * k_block + col_block) % stride == 0)
    //     {
    //         h_A[i].r = 1;
    //         h_A[i].i = 2;
    //     }
    //     else
    //     {
    //         h_A[i].r = 3;
    //         h_A[i].i = 4;
    //     }
    // }

    // // 生成B的数据
    // for (int i = 0; i < K * N; i++)
    // {
    //     if (i >= K * N / 2)
    //     {
    //         h_B[i].r = 5;
    //         h_B[i].i = 6;
    //     }
    //     else
    //     {
    //         h_B[i].r = 7;
    //         h_B[i].i = 8;
    //     }
    // }
    for (int i = 0; i < M * K; i++)
    {
        int row = (i / K);
        int col = (i % K);
        int row_block = row / BLOCK_SIZE_M;
        int col_block = col / BLOCK_SIZE_K;
        
        h_A[i].r = double(i);
        h_A[i].i = double(i);
            // srand((unsigned)time(NULL));
            // h_A[i].r = double(rand())/RAND_MAX;
            // h_A[i].i = double(rand())/RAND_MAX;
        

    }

    // 生成B的数据
    for (int i = K * N; i >0 ; i--)
    {
        // if (i >= K * N / 2)
        // {
        //     h_B[i].r = 5;
        //     h_B[i].i = 6;
        // }
        // else
        // {
        //     h_B[i].r = 7;
        //     h_B[i].i = 8;
        // }
        // srand((unsigned)time(NULL));
        // h_B[i].r = double(rand())/RAND_MAX;
        // h_B[i].i = double(rand())/RAND_MAX;

        h_B[K * N-i].r = double(i);
        h_B[K * N-i].i = double(i);
    }
#endif
#ifdef PRINT_MAT
    // printf("A:\n");
    // for (int i = 0; i < M; i++)
    // {
    //     for (int j = 0; j < K; j++)
    //     {
    //         printf("%f + %f i", h_A[OFFSET(i, j, K)].r, h_A[OFFSET(i, j, K)].i);
    //     }
    //     printf("\n");
    // }
    // printf("B:\n");
    // for (int i = 0; i < K; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%f + %f i", h_B[OFFSET(i, j, N)].r, h_B[OFFSET(i, j, N)].i);
    //     }
    //     printf("\n");
    // }
#endif
    // for( int i = 0; i < M * K; i++ ) {
    //     h_A->i = i+3;
    //     h_A->r = i+3;
    // }

    // // 生成B的数据
    // for( int i = 0; i < K * N; i++ ) {
    //     h_B->i = i;
    //     h_B->r = i;
    // }

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid((N - 1) / BLOCK_SIZE_N + 1, (M - 1) / BLOCK_SIZE_M + 1);
        Zgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
            <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[0],
           msecPerMatrixMul[0],
           flopsPerMatrixMul);

    // cublas

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    // float alpha = 1.0;
    // float beta = 0;
    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        cublasZgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    M, N, K, (cuDoubleComplex *)&zone,
                    (cuDoubleComplex *)d_A, K, (cuDoubleComplex *)d_B, N, (cuDoubleComplex *)&zero, (cuDoubleComplex *)d_C, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[1],
           msecPerMatrixMul[1],
           flopsPerMatrixMul);

    cublasDestroy(blas_handle);

    //transpose
    // for(int i = 0;i<M;i++)
    // {
    //     for(int j = 0;j<N;j++)
    //     {
    //         complex temp;
    //         temp =  h_C[OFFSET(i, j, N)];
    //         h_C[OFFSET(i, j, N)] =  h_C[OFFSET(j, i, N)];
    //         h_C[OFFSET(j, i, N)] = temp;
           
    //     }
    // }
    double eps = 1.e-6; // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        int row = i / N;
        int col = i % N;
        double abs_err_r = fabs(h_C[i].r - h_C1[col * M + row].r);
        double abs_err_i = fabs(h_C[i].i - h_C1[col * M + row].i);
        // double abs_err_r = fabs(h_C[i].r - h_C1[i].r);
        // double abs_err_i = fabs(h_C[i].i - h_C1[i].i);
        double dot_length = M;
        double abs_val_r = fabs(h_C[i].r);
        double abs_val_i = fabs(h_C[i].i);
        double rel_err_r = abs_err_r / abs_val_r / dot_length;
        double rel_err_i = abs_err_i / abs_val_i / dot_length;
        if (rel_err_r > eps || rel_err_i > eps)
        {
            printf("Error! Matrix[%05d][%05d]=%.8f, ref_r=%.8f error term is > %E\n",
                   row, col, h_C[i].r, h_C1[col * M + row].r, eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

#ifdef PRINT_MAT
    printf("my gemm:\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f + %f i", h_C[OFFSET(i, j, N)].r, h_C[OFFSET(i, j, N)].i);
        }
        printf("\n");
    }

    printf("cublas:\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f + %f i", h_C1[OFFSET(j, i, N)].r, h_C1[OFFSET(j, i, N)].i);
        }
        printf("\n");
    }
#endif
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}
