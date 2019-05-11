__global__ void PatchedSumImageKernel(double *A,
                                      double *summed_Arr,
                                      uint A_width,
                                      uint A_height,
                                      uint n_color,
                                      uint width,
                                      uint height)
{
    uint N_s_row = A_height - height + 1;
    uint N_s_col = A_width - width + 1;
    uint stride_x = blockDim.x * gridDim.x;
    uint stride_y = blockDim.y * gridDim.y;
    for (uint j = threadIdx.y + blockIdx.y * blockDim.y;
             j < N_s_row;
             j += stride_y)
    {
        for (uint i = threadIdx.x + blockIdx.x * blockDim.x;
             i < N_s_col;
             i += stride_x)
        {
            double sum = 0.;
            for (uint l = 0; l < height; l++){
                for (uint k = 0; k < width; k++){
                    for (uint m = 0; m < n_color; m++){
                        sum += A[(i + k) * n_color + 
                                 (j + l) * n_color * 
                                 A_width + m];
                    }
                }
            }
            summed_Arr[i + j * N_s_col] = sum;
        }
    }
}