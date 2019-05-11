__global__ void axpbyKernel(double* x, double* y, double a, double b, uint L){
    uint stride = gridDim.x * blockDim.x;
    uint t = threadIdx.x + blockIdx.x * blockDim.x;

    for (uint i = t; i < L; i += stride){
        x[i] = a * x[i] + b * y[i];
    }
}

__global__ void axpbyyKernel(double* x, double* y,
                             double a, double b, uint L){
    uint stride = gridDim.x * blockDim.x;
    uint t = threadIdx.x + blockIdx.x * blockDim.x;

    for (uint i = t; i < L; i += stride){
        x[i] = a * x[i] + b * y[i] * y[i];
    }
}

__global__ void gradModKernel(double* mhat, double* vhat, double* dev,
                              double eps, uint L){
    uint stride = gridDim.x * blockDim.x;
    uint t = threadIdx.x + blockIdx.x * blockDim.x;

    for (uint i = t; i < L; i += stride){
        dev[i] = mhat[i] / (sqrt(vhat[i]) + eps);
    }
}
