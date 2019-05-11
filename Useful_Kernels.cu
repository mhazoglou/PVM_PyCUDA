// using shuffle to reduce and do column vector-matrix multiplication
// these are all the useful kernels for calculating forward and 
// backward propagation
// a(m, 1) = sigmoid( W(m, n) x(n, 1) + B(m, 1) )
// 
// I'm programming this with block size of 32 by 32 in mind for dense
// matrix arrays
// you may need to change the warpsize if your blocksize cannot handle
// 1024 thread

//#define warpSize 32
#define CUDART_INF_D __longlong_as_double(0x7ff0000000000000)
#define CUDART_NINF_D __longlong_as_double(0xfff0000000000000)

// sums all the values of val in each thread of a warp using 
// __shfl_down
__inline__ __device__
double warpReduceSum(double val) {
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

// block reduction using warpReduceSum
__inline__ __device__
double blockReduceSum(double val) {

  static __shared__ double shared[32]; // Shared mem for 32 partial sums
  unsigned int lane = threadIdx.x % warpSize;
  unsigned int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void SumKernel(double *A, double *S, unsigned int L)
{
  double sum = 0.;
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < L;
       i += stride)
  {
    sum += A[i];
  }
  
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0)
    atomicAdd(S, sum);
}

// Elementwise addition with grid stride loops
__global__ void ArrayAddKernel(double *A1, double *A2, double *R,
                               unsigned int L)
{
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < L;
       i += stride)
  {
    R[i] = A1[i] + A2[i];
  }
}

// Elementwise subtraction with grid stride loops
__global__ void ArrayDifferenceKernel(double *A1, double *A2, double *R,
                                      unsigned int L)
{
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < L;
       i += stride)
  {
    R[i] = A1[i] - A2[i];
  }
}

// Elementwise multiplication with grid stride loops
__global__ void HadamardProductKernel(double *A1, double *A2, double *R,
                                      unsigned int L)
{
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < L;
       i += stride)
  {
    R[i] = A1[i] * A2[i];
  }
}

// Fills an array with zeros
__global__ void ZeroFillKernel(double *A, unsigned int L)
{
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < L;
       i += stride)
  {
    A[i] = 0.;
  }
}

// Takes a sparse matrix in CSR format and returns it's product with a vector
__global__  void
spmv_csr_kernel(const unsigned int num_rows,
                const unsigned int  *ptr,
                const unsigned int  *indices,
                const double *data,
                const double *x,
                double *y)
{
  unsigned int  start = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int row = start; row < num_rows; row += stride)
  {
    double dot = 0.;
    unsigned int row_start = ptr[row];
    unsigned int row_end   = ptr[row +1];
    for (unsigned int j = row_start; j < row_end; j++)
    {
      dot += data[j] * x[indices[j]];
    }
    y[row] = dot;
  }
}

// More efficient for doing independent multiple matrix vector multiplication
// using a CSR format
__global__ void 
Blocked1dCSRMatVecDotKernel(const unsigned int num_rows,
                const unsigned int  *ptr,
                const unsigned int  *indices,
                const double *data,
                const double *x,
                double *y)
{
    
    unsigned int  start_x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride_x = blockDim.x * gridDim.x;
    unsigned int  start_y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int stride_y = blockDim.y * gridDim.y;
    
    for (unsigned int row = start_y; row < num_rows; row += stride_y)
    {
        double sum = 0.;
        unsigned int row_start = ptr[row];
        //unsigned int row_end   = ptr[row +1];
        unsigned int row_length = ptr[row +1] - row_start;
        for (unsigned int j = start_x; j < row_length; j += stride_x)
        {
          sum += data[row_start + j] * x[indices[row_start + j]];
        }
        
        sum = blockReduceSum(sum);
        if (threadIdx.x==0)
        {
          atomicAdd(&y[row], sum);
        }
        
    }
}

// Takes two vectors and returns a CSR matrix made from outer products
// of sub-spaces of the vectors
__global__ void spvv_csr_outer_kernel(const unsigned int  num_rows,
                                      const unsigned int  *ptr,
                                      const unsigned int  *indices,
                                      const double *x,
                                      const double *y,
                                      double *data)
{
  unsigned int  start = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int row = start; row < num_rows; row += stride)
  {
    unsigned int row_start = ptr[row];
    unsigned int row_end   = ptr[row +1];
    for (unsigned int j = row_start; j < row_end; j++)
    {
      data[j] = y[row]*x[indices[j]];
    }
    
  }
}

// similar to the above kernel but take row indices instead of a
// pointer array in CSR format it's fast if you have COO information
__global__ void spvv_coo_outer_kernel(const unsigned int nnz,
                                      const unsigned int *row_idx,
                                      const unsigned int *col_idx,
                                      const double *x,
                                      const double *y,
                                      double *data)
{
  unsigned int stride = blockDim.x * gridDim.x;
  unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
  for (unsigned int el = start; el < nnz; el += stride)
  {
    data[el] = y[row_idx[el]] * x[col_idx[el]];
  }
}

// Takes a sparse matrix in CSR format and returns it's tranpose product with a vector
// needs y to be filled with zeros to return the right result (use ZeroFillKernel)
__global__  void
spmTv_csr_kernel(const unsigned int  num_rows,
                 const unsigned int  *ptr,
                 const unsigned int  *indices,
                 const double *data,
                 const double *x,
                 double *y)
{
  unsigned int  start = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int row = start; row < num_rows; row += stride)
  {
    unsigned int row_start = ptr[row];
    unsigned int row_end   = ptr[row +1];
    for (unsigned int j = row_start; j < row_end; j++)
    {
      atomicAdd(&y[indices[j]], data[j] * x[row]);
    }
  }
}

// Takes two vectors (one column Vy and one row Vx) and give a matrix M
__global__ void OuterProductKernel(double *Vx, double *Vy, double *M,
                                   unsigned int L_x, unsigned int L_y)
{
  unsigned int starting_col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride_y = blockDim.y * gridDim.y;
  unsigned int stride_x = blockDim.x * gridDim.x;
  for (unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
       row < L_y;
       row += stride_y)
  {
    for (unsigned int col = starting_col;
         col < L_x;
         col += stride_x)
    {
      M[col + row * L_x] = Vx[col] * Vy[row];
    }
  }
}

// product of a matrix with a vector
// R(L_R, 1) = A(L_R, L_V) V(L_V, 1)
// the result R needs to be initialized with zeros (use ZeroFillKernel) 
__global__ void MatVecDotKernel(const double *A,
                                const double *V,
                                double *R,
                                const unsigned int L_V,
                                const unsigned int L_R)
{
  unsigned int starting_col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride_y = blockDim.y * gridDim.y;
  unsigned int stride_x = blockDim.x * gridDim.x;
  for (unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
       row < L_R;
       row += stride_y) 
  {
    double sum = 0;
    //reduce multiple elements per thread
    for (unsigned int col = starting_col; 
         col < L_V; 
         col += stride_x) {
      sum += A[col + row * L_V] * V[col];
    }
    
    sum = warpReduceSum(sum);
    static __shared__ double shared[32]; // Shared mem for 32 partial sums
    shared[threadIdx.y] = sum;
     

    if (threadIdx.x==0)
    {
      atomicAdd(&R[row], shared[threadIdx.y]);
    }
  
  }
}

// product of transpose of a matrix times a vector
// A(L_V, L_R)
// R(L_R, 1) = A.transpose (L_R, L_V) V(L_V, 1)
// row and col below refer to the transposed matrix
// the result R needs to be initialized with zeros (use ZeroFillKernel) 
__global__ void MatTransposeVecDotKernel(const double *A,
                                         const double *V,
                                         double *R,
                                         const unsigned int L_V,
                                         const unsigned int L_R)
{
  unsigned int starting_col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride_y = blockDim.y * gridDim.y;
  unsigned int stride_x = blockDim.x * gridDim.x;
  for (unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
       row < L_R;
       row += stride_y) 
  {
    double sum = 0;
    //reduce multiple elements per thread
    for (unsigned int col = starting_col; 
         col < L_V; 
         col += stride_x) {
      sum += A[col * L_R + row] * V[col];
    }
    
  sum = warpReduceSum(sum);
  static __shared__ double shared[32]; // Shared mem for 32 partial sums
  shared[threadIdx.y] = sum;
    

  if (threadIdx.x==0)
  {
    atomicAdd(&R[row], shared[threadIdx.y]);
  }
  
  }
}

// Elementwise sigmoid of an array
__global__ void SigmoidKernel(const double *z,
                              double *a,
                              const unsigned int L_z)
{
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		a[i] = 1. / (1. + exp(-z[i]));
	}
}

// Elementwise derivative of a sigmoid for an array
__global__ void SigmoidPrimeKernel(const double *z,
                                   double *a_prime,
		                   const unsigned int L_z)
{
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		 double sigmoid = 1. / (1. + exp(-z[i]));
		 a_prime[i] = sigmoid * (1. - sigmoid);
	}
}

// Elementwise sigmoid of an array
__global__ void TanhKernel(const double *z,
                              double *a,
                              const unsigned int L_z)
{
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		a[i] = tanh(z[i]);
	}
}

// Elementwise derivative of a sigmoid for an array
__global__ void TanhPrimeKernel(const double *z,
                                   double *a_prime,
		                   const unsigned int L_z)
{
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		 double sec = 1. / cosh(z[i]);
		 a_prime[i] = sec * sec;
	}
}

//Elementwise Rectified linear function of an array
__global__ void ReLUKernel(const double *z,
                           double *a,
                           const uint L_z)
{
	uint stride = blockDim.x * gridDim.x;
	for (uint i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		a[i] = fmax(z[i], 0.);
	}
}

//Elementwise derivative of Rectified linear function of an array
__global__ void ReLUPrimeKernel(const double *z,
                                double *a,
                                const uint L_z)
{
	uint stride = blockDim.x * gridDim.x;
	for (uint i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		if (z[i] > 0.)
		{
			a[i] = 1.;
		}
		else
		{
			a[i] = 0.;
		}
	}
}

//Elementwise identity (linear) function of an array
__global__ void IdentityKernel(const double *z,
                               double *a,
                               const uint L_z)
{
	uint stride = blockDim.x * gridDim.x;
	for (uint i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		a[i] = z[i];
	}
}

//Elementwise derivative of identity (linear) function of an array
__global__ void OneFillKernel(const double *z,
                              double *a,
                              const uint L_z)
{
	uint stride = blockDim.x * gridDim.x;
	for (uint i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		a[i] = 1.;	
	}
}

// Elementwise SoftPlus of an array
// the derivative of this function is the sigmoid so I won't implement it
// as another kernel
__global__ void SoftPlusKernel(const double *z,
                               double *a,
                               const uint L_z)
{
	uint stride = blockDim.x * gridDim.x;
	for (uint i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_z;
		 i += stride)
	{
		a[i] = log1p(exp(z[i]));
	}
}


// Last step in backpropagation
__global__ void UpdateKernel(double *parameters, double *delta,
		double learning_rate, unsigned int L)
{
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L;
		 i += stride)
	{
		parameters[i] -= learning_rate * delta[i];
	}
}

__inline__ __device__
void warpReduceMax(double* val, uint* arg) 
{
    for (unsigned int offset = warpSize/2; offset > 0; offset /= 2)
        {
            double tmp_val = __shfl_down(*val, offset);
            uint   tmp_arg = __shfl_down(*arg, offset);
            /*
            tmp_arg can never actually be zero because of 
            __shfl_down resulting in a zero value indicates 
            a thread that is inactive (an undefined result)
            */
            if (tmp_val > *val)
            {
                *val = tmp_val;
                *arg = tmp_arg;
            }
        }
        
}


__inline__ __device__
void blockReduceMax(double *val, 
                    uint *arg) 
{

    static __shared__ double shared_val[32]; // Shared mem for 32 partial maxs
    static __shared__ uint shared_arg[32]; // shared mem for 32 partial argmaxs
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    warpReduceMax(val, arg);     // Each warp performs partial reduction

    if (lane==0)
    {
        shared_val[wid] = *val; // Write reduced value to shared memory
        shared_arg[wid] = *arg;
    }

    __syncthreads();              // Wait for all partial reductions

    // read from shared memory only if that warp existed
    // if we have an blockDim.x which is smaller than the
    // warpSize (32) we have a problem
    *val = (threadIdx.x < blockDim.x / warpSize) ? shared_val[lane] : CUDART_NINF_D;
    *arg = shared_arg[lane];


    if (wid==0) warpReduceMax(val, arg); //Final reduce within first warp

}

// does a reduction to arrays to find maximum at most needs to be run twice
// in sequence to do a full reduction
__global__ void MaxKernel(double* A,
                          double* Max,
                          uint* arg_Max,
                          uint L)
{
    uint grid_stride_arg_max;
    double grid_stride_max, a;
    
    uint start = threadIdx.x + blockIdx.x * blockDim.x;
    uint stride = blockDim.x * gridDim.x;
    
    // a way of picking out threads that are beyond
    // the length of the array
    if (start < L)
    {
        a = A[start];
        grid_stride_max = a;
        grid_stride_arg_max = start;
    }
    else
    {
        a = CUDART_NINF_D;
        grid_stride_max = CUDART_NINF_D;
        grid_stride_arg_max = 0;
    }
    
    // grid stride loop to find the max and argmax across
    // grid strides
    for (uint i = start;
         i < L;
         i += stride)
        {
            a = A[i];
            if (a > grid_stride_max)
            {
                grid_stride_max = a;
                grid_stride_arg_max = i;
            }
        }
    
    // blockReduceMax gives the maximum and argmax of 
    // each block in grid_stride_max and grid_stride_arg_max
    blockReduceMax(&grid_stride_max, &grid_stride_arg_max);
    if (threadIdx.x == 0)
    {
        Max[blockIdx.x] = grid_stride_max;
        arg_Max[blockIdx.x] = grid_stride_arg_max;
    }
    
}

__inline__ __device__
void warpReduceMin(double* val, uint* arg) 
{
    for (unsigned int offset = warpSize/2; offset > 0; offset /= 2)
        {
            double tmp_val = __shfl_down(*val, offset);
            uint   tmp_arg = __shfl_down(*arg, offset);
            /*
            tmp_arg can never actually be zero because of 
            __shfl_down resulting in a zero value indicates 
            a thread that is inactive (an undefined result)
            */
            if (tmp_val < *val)
            {
                *val = tmp_val;
                *arg = tmp_arg;
            }
        }
        
}


__inline__ __device__
void blockReduceMin(double *val, 
                    uint *arg) 
{

    static __shared__ double shared_val[32]; // Shared mem for 32 partial mins
    static __shared__ uint shared_arg[32]; // shared mem for 32 partial argmins
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    warpReduceMin(val, arg);     // Each warp performs partial reduction

    if (lane==0)
    {
        shared_val[wid] = *val; // Write reduced value to shared memory
        shared_arg[wid] = *arg;
    }

    __syncthreads();              // Wait for all partial reductions

    // read from shared memory only if that warp existed
    // if we have a BlockDim.x which is smaller than the warpSize (32) we have a problem  
    *val = (threadIdx.x < blockDim.x / warpSize) ? shared_val[lane] : CUDART_INF_D;
    *arg = shared_arg[lane];


    if (wid==0) warpReduceMin(val, arg); //Final reduce within first warp

}

// does a reduction to arrays to find minimum at most needs to be run twice
// in sequence to do a full reduction
__global__ void MinKernel(double* A,
                          double* Min,
                          uint* arg_Min,
                          uint L)
{
    uint grid_stride_arg_min;
    double grid_stride_min, a;
    
    uint start = threadIdx.x + blockIdx.x * blockDim.x;
    uint stride = blockDim.x * gridDim.x;
    
    // a way of picking out threads that are beyond
    // the length of the array
    if (start < L)
    {
        a = A[start];
        grid_stride_min = a;
        grid_stride_arg_min = start;
    }
    else
    {
        a = CUDART_INF_D;
        grid_stride_min = CUDART_INF_D;
        grid_stride_arg_min = 0;
    }
    
    // grid stride loop to find the min and argmin across
    // grid strides
    for (uint i = start;
         i < L;
         i += stride)
        {
            a = A[i];
            if (a < grid_stride_min)
            {
                grid_stride_min = a;
                grid_stride_arg_min = i;
            }
        }
    
    // blockReduceMin gives the minimum and argmin of 
    // each block in grid_stride_min and grid_stride_arg_min
    blockReduceMin(&grid_stride_min, &grid_stride_arg_min);
    if (threadIdx.x == 0)
    {
        Min[blockIdx.x] = grid_stride_min;
        arg_Min[blockIdx.x] = grid_stride_arg_min;
    }
    
}

