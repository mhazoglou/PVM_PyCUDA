// Useful functions for the PVM that are specific to the tracker

// derivative and error calculation
__global__ void der_and_error_kernel(double *A, double *B, double *C,
		unsigned int L)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;

	for (unsigned int i = start; i < L; i += stride)
	{
		C[i] = 0.5 * (1. + A[i] - B[i]);
	}
}

// integral calculation
__global__ void integral_kernel(double *A, double *B, double *C,
		unsigned int L, double tau = 0.5)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;

	for (unsigned int i = start; i < L; i += stride)
	{
		// this is the same as tau * A[i] + (1 - tau) * B[i]
		C[i] = tau * (A[i] - B[i]) + B[i];
	}
}

// appending the hidden values passed to upper layers as inputs to 
// the end of the inputs (you need to do this to use the integral_kernel
// and the der_and_error_kernel)
__global__ void hid_append_kernel(double *inputs, double *hidden, 
		double *concat_arr, unsigned int *map, 
		unsigned int L_inputs, unsigned int L_concat_arr)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;

	for (unsigned int i = start; i < L_concat_arr; i += stride)
	{
		if (i < L_inputs)
		{
			concat_arr[i] = inputs[i];
		}
		else
		{
			concat_arr[i] = hidden[map[i-L_inputs]];
		}
	}
}


// Average pooling of heatmaps from each layer
__global__ void tracker_avg_pool_kernel(double *heat_maps,
		double *avg_heat_map, unsigned int L_avg, unsigned int N_layers)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
	
	const unsigned int L_heat_maps = N_layers * L_avg;
	for (unsigned int i = start; i < L_avg; i += stride)
	{
		double sum = 0.;
		for (unsigned int j = i; j < L_heat_maps; j += L_avg)
		{
			sum += heat_maps[j];
		}
		avg_heat_map[i] = sum / N_layers;
	}
}

__global__ void full_input_map_kernel(double *full_input,
		double *sub_array, const unsigned int *map, unsigned int L_sub)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (unsigned int i = start; i < L_sub; i += stride)
	{
		full_input[map[i]] = sub_array[i];
	}
}

__global__ void hidden_map_to_full_input_kernel(double *full_input,
		const double *hidden, const int *map, unsigned int L_full)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (unsigned int i = start; i < L_full; i += stride)
	{
		int idx = map[i];
		if (idx != -1)
		{
			full_input[i] = hidden[idx];
		}
	}
}

__global__ void gradient_inv_hid_map_kernel(double *grad_wrt_hid,
                                            double *grad_wrt_full_input, 
                                            const unsigned int *map,
                                            const unsigned int L_full)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (unsigned int i = start; i < L_full; i += stride)
	{
		int idx = map[i];
		if (idx != -1)
		{		
			atomicAdd(&grad_wrt_hid[idx], grad_wrt_full_input[i]);
		}
	}
}

__global__ void output_pred_map_kernel(double *sub_array,
		double *out_and_pred, const unsigned int *map, unsigned int L_sub)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (unsigned int i = start; i < L_sub; i += stride)
	{
		sub_array[i] = out_and_pred[map[i]];
	}
}

__global__ void rev_output_pred_map_kernel(double *sub_array,
		double *out_and_pred, const unsigned int *map, unsigned int L_sub)
{
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int start = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (unsigned int i = start; i < L_sub; i += stride)
	{
		out_and_pred[map[i]] = sub_array[i];
	}
}

__global__ void SquareErrorDerTrackerKernel(double *avg_heatmap,
		double *gt_heatmap, double *delta_tracker,
		unsigned int L_avg, unsigned int N_layers)
{
	const unsigned int L_heat_maps = N_layers * L_avg;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < L_avg;
		 i += stride)
	{
		double tmp = (avg_heatmap[i] - gt_heatmap[i]) / N_layers;
		for (unsigned int j = i; j < L_heat_maps; j += L_avg)
			{
				delta_tracker[j] = tmp;
			}
	}
}


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
    for (uint i = threadIdx.x + blockIdx.x * blockDim.x;
         i < N_s_col;
         i += stride_x)
    {
        for (uint j = threadIdx.y + blockIdx.y * blockDim.y;
             j < N_s_row;
             j += stride_y)
        {
            double sum = 0.;
            for (uint k = 0; k < width; k++)
                {
                for (uint l = 0; l < height; l++)
                    {
                    for (uint m = 0; m < n_color; m++)
                        sum += A[(i + k) * n_color + (j + l) * n_color * A_width + m];
                    }
                }
            summed_Arr[i + j * N_s_col] = sum;
        }
    }
}
