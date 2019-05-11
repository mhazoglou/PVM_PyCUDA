from __future__ import absolute_import, print_function, division
from builtins import *
import math
import numpy as np
from pycuda import driver, compiler, gpuarray, tools


# -- initialize the device
# import pycuda.autoinit

fname = "./Useful_Kernels.cu"
tracker_filename = "./tracker_kernels.cu"

with open(fname) as fid:
    kernel_code = fid.read()

with open(tracker_filename) as fid:
    tracker_kernel_code = fid.read()

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# basic element-wise addition
# (I might want to compare the speed
# with the default addition hook on gpu arrays)
add = mod.get_function('ArrayAddKernel')
add.prepare(['P', 'P', 'P', np.int32])

# basic element-wise subtraction
# (I might want to compare the speed
# with the default subtraction hook on gpu arrays)
sub = mod.get_function('ArrayDifferenceKernel')
sub.prepare(['P', 'P', 'P', np.int32])

# basic element-wise multiplication
# (I might want to compare the speed
# with the default multiplication hook on gpu arrays)
hadamard = mod.get_function('HadamardProductKernel')
hadamard.prepare(['P', 'P', 'P', np.int32])

# All vectors are dense, all matrices are CSR

# sets all the elements of an array equal to zero
# this is needed for the spmTv_csr_kernel since atomic add is used
zerofill = mod.get_function('ZeroFillKernel')
zerofill.prepare(['P', np.int32])

# outer product that maps result to the shape of the sparse weight matrix
# (indices and pointers needed)
kron = mod.get_function('spvv_coo_outer_kernel')
kron.prepare([np.int32, 'P', 'P', 'P', 'P', 'P'])

# sparse matrix-vector multiplication using CSR format
dot = mod.get_function('spmv_csr_kernel')
dot.prepare([np.int32, 'P', 'P', 'P', 'P', 'P'])

# sparse matrix-transpose-vector multiplication using CSR format
Tdot = mod.get_function('spmTv_csr_kernel')
Tdot.prepare([np.int32, 'P', 'P', 'P', 'P', 'P'])

# element-wise application of the sigmoid function
sig = mod.get_function('SigmoidKernel')
sig.prepare(['P', 'P', np.int32])

# element-wise application of the derivative of the sigmoid
dsig = mod.get_function('SigmoidPrimeKernel')
dsig.prepare(['P', 'P', np.int32])

# updating weights and biases
update = mod.get_function('UpdateKernel')
update.prepare(['P', 'P', np.float64, np.int32])

# --------------------Tracker functions--------------------
# compile the kernel code
mod_tracker = compiler.SourceModule(tracker_kernel_code)

der_and_error = mod_tracker.get_function('der_and_error_kernel')
der_and_error.prepare(['P', 'P', 'P', np.int32])

integral = mod_tracker.get_function('integral_kernel')
integral.prepare(['P', 'P', 'P', np.int32])

append_hid = mod_tracker.get_function('hid_append_kernel')
append_hid.prepare(['P', 'P', 'P', 'P', np.int32, np.int32])

avg_pool = mod_tracker.get_function('tracker_avg_pool_kernel')
avg_pool.prepare(['P', 'P', np.int32, np.int32])

input_shuffling = mod_tracker.get_function('full_input_map_kernel')
input_shuffling.prepare(['P', 'P', 'P', np.int32])

input_hidden_shuffling = mod_tracker.get_function(
                                        'hidden_map_to_full_input_kernel')
input_hidden_shuffling.prepare(['P', 'P', 'P', np.int32])

output_pred_shuffling = mod_tracker.get_function('output_pred_map_kernel')
output_pred_shuffling.prepare(['P', 'P', 'P', np.int32])

rev_output_pred_shuffling = mod_tracker.get_function(
    'rev_output_pred_map_kernel')
rev_output_pred_shuffling.prepare(['P', 'P', 'P', np.int32])

sq_err_der_tracker = mod_tracker.get_function('SquareErrorDerTrackerKernel')
sq_err_der_tracker.prepare(['P', 'P', 'P', np.int32, np.int32])


class ParallelAutoencoder(object):
    def __init__(self, spec_dict, hidden_size, max_threads=1024, max_grid_size=1024):
        i2h_stuff, h2p_stuff = self.weight_initialize(spec_dict, hidden_size)
        i2h_pointers, i2h_indices, i2h_weights, i2h_bias = i2h_stuff
        h2p_pointers, h2p_indices, h2p_weights, h2p_bias = h2p_stuff

        self.i2h_pointers = gpuarray.to_gpu(i2h_pointers)
        self.i2h_indices = gpuarray.to_gpu(i2h_indices)
        self.i2h_weights = gpuarray.to_gpu(i2h_weights)
        self.i2h_bias = gpuarray.to_gpu(i2h_bias)

        self.i2h_row_idx = gpuarray.to_gpu(self.ptr_to_row(i2h_pointers))

        self.h2p_pointers = gpuarray.to_gpu(h2p_pointers)
        self.h2p_indices = gpuarray.to_gpu(h2p_indices)
        self.h2p_weights = gpuarray.to_gpu(h2p_weights)
        self.h2p_bias = gpuarray.to_gpu(h2p_bias)

        self.h2p_row_idx = gpuarray.to_gpu(self.ptr_to_row(h2p_pointers))

        self.L_hidden = np.int32(len(i2h_bias))
        self.L_pred = np.int32(len(h2p_bias))

        self.i2h_weights_nnz = np.int32(i2h_pointers[-1])
        self.h2p_weights_nnz = np.int32(h2p_pointers[-1])

        self.hidden_size = hidden_size

        self.hidden = gpuarray.to_gpu(np.zeros(self.L_hidden))
        self.pred = gpuarray.to_gpu(np.zeros(self.L_pred))

        self.hid_affine = gpuarray.empty_like(self.hidden)
        self.pred_affine = gpuarray.empty_like(self.pred)

        self.delta_h2p = gpuarray.to_gpu(np.zeros(self.L_pred))
        self.delta_i2h = gpuarray.to_gpu(np.zeros(self.L_hidden))

        self.a_prime_h2p = gpuarray.empty_like(self.pred)
        self.a_prime_i2h = gpuarray.empty_like(self.hidden)

        self.grad_weight_h2p = gpuarray.empty_like(self.h2p_weights)
        self.grad_weight_i2h = gpuarray.empty_like(self.i2h_weights)

        ceil = lambda x: int(math.ceil(x))
        self.threads = (max_threads, 1, 1)
        grid_x_i2h = min(ceil(self.L_hidden / max_threads), max_grid_size)
        grid_x_h2p = min(ceil(self.L_pred / max_threads), max_grid_size)
        grid_x_update_i2h = min(ceil(self.i2h_weights_nnz / max_threads), max_grid_size)
        grid_x_update_h2p = min(ceil(self.h2p_weights_nnz / max_threads), max_grid_size)

        self.grid_i2h = (grid_x_i2h, 1)
        self.grid_h2p = (grid_x_h2p, 1)
        self.grid_update_i2h = (grid_x_update_i2h, 1)
        self.grid_update_h2p = (grid_x_update_h2p, 1)

        self.stream1 = driver.Stream()
        self.stream2 = driver.Stream()
        self.stream3 = driver.Stream()
        self.stream4 = driver.Stream()
        self.stream5 = driver.Stream()

        self.sq_err = []

    def forward(self, input_):
        input_gpu = gpuarray.to_gpu(input_.astype(np.float64))

        dot.prepared_call(self.grid_i2h, self.threads,
                          # number of rows
                          self.L_hidden,
                          # CSR sparse matrix
                          self.i2h_pointers.gpudata,
                          self.i2h_indices.gpudata,
                          self.i2h_weights.gpudata,
                          # vector
                          input_gpu.gpudata,
                          # result
                          self.hid_affine.gpudata)

        # i2h_csr_weights = sp.csr_matrix((self.i2h_weights.get(),
        #                                  self.i2h_indices.get(),
        #                                  self.i2h_pointers.get()))
        #
        # print(i2h_csr_weights.dot(input_) - self.hid_affine.get())
        #
        # hid_aff = i2h_csr_weights.dot(input_) + self.i2h_bias.get()

        add.prepared_call(self.grid_i2h, self.threads,
                          self.hid_affine.gpudata, self.i2h_bias.gpudata,
                          self.hid_affine.gpudata, self.L_hidden)

        # print(self.hid_affine.get() - hid_aff)

        sig.prepared_call(self.grid_i2h, self.threads,
                          self.hid_affine.gpudata, self.hidden.gpudata,
                          self.L_hidden)

        # hid = 1. / (1. + np.exp(- hid_aff))
        #
        # print(self.hidden.get() - hid)

        dot.prepared_call(self.grid_h2p, self.threads,
                          self.L_pred,
                          self.h2p_pointers.gpudata,
                          self.h2p_indices.gpudata,
                          self.h2p_weights.gpudata,
                          self.hidden.gpudata,
                          self.pred_affine.gpudata)

        # h2p_csr_weights = sp.csr_matrix((self.h2p_weights.get(),
        #                                  self.h2p_indices.get(),
        #                                  self.h2p_pointers.get()))

        # pred_aff = h2p_csr_weights.dot(hid) + self.h2p_bias.get()

        add.prepared_call(self.grid_h2p, self.threads,
                          self.pred_affine.gpudata, self.h2p_bias.gpudata,
                          self.pred_affine.gpudata, self.L_pred)

        # print(self.pred_affine.get() - pred_aff)

        sig.prepared_call(self.grid_h2p, self.threads,
                          self.pred_affine.gpudata, self.pred.gpudata,
                          self.L_pred)

        # pred = 1. / (1. + np.exp(-pred_aff))
        #
        # print(self.pred.get() - pred)

    def _backward(self, input_, ideal_pred, learning_rate):
        # should be called only after forward is called first
        input_gpu = gpuarray.to_gpu_async(input_, stream=self.stream1)
        ideal_pred_gpu = gpuarray.to_gpu_async(ideal_pred, stream=self.stream2)

        dsig.prepared_async_call(self.grid_i2h, self.threads,
                                 self.stream3,
                                 self.hid_affine.gpudata, self.a_prime_i2h.gpudata,
                                 self.L_hidden)

        # hid_aff = self.hid_affine.get()
        # sigmoid = lambda x: 1. / (1. + np.exp(-x))
        # dsigmoid = lambda x: sigmoid(x) * (1. - sigmoid(x))
        # print('a_prime_i2h')
        # print(self.a_prime_i2h.get() - dsigmoid(hid_aff))

        dsig.prepared_async_call(self.grid_h2p, self.threads,
                                 self.stream4,
                                 self.pred_affine.gpudata, self.a_prime_h2p.gpudata,
                                 self.L_pred)

        # pred_aff = self.pred_affine.get()
        # print('a_prime_h2p')
        # print(self.a_prime_h2p.get() - dsigmoid(pred_aff))

        sub.prepared_async_call(self.grid_h2p, self.threads,
                                self.stream5,
                                self.pred.gpudata, ideal_pred_gpu.gpudata,
                                self.delta_h2p.gpudata,
                                self.L_pred)

        # print('dC/da')
        # print(self.pred.get() - ideal_pred - self.delta_h2p.get())
        #
        # del_h2p = self.delta_h2p.get() * dsigmoid(pred_aff)

        hadamard.prepared_call(self.grid_h2p, self.threads,
                               self.delta_h2p.gpudata, self.a_prime_h2p.gpudata,
                               self.delta_h2p.gpudata,
                               self.L_pred)

        # print('delta_h2p')
        # print(self.delta_h2p.get() - del_h2p)

        zerofill.prepared_call(self.grid_i2h, self.threads,
                               self.delta_i2h.gpudata, self.L_hidden)

        # print('This should be delta_i2h filled with zeros:\n')
        # print(self.delta_i2h.get())

        Tdot.prepared_call(self.grid_h2p, self.threads,
                           self.L_pred, self.h2p_pointers.gpudata,
                           self.h2p_indices.gpudata, self.h2p_weights.gpudata,
                           self.delta_h2p.gpudata, self.delta_i2h.gpudata)

        # print('\nThis should be delta_i2h.\n')
        # h2p_csr_weights = sp.csr_matrix((self.h2p_weights.get(),
        #                                  self.h2p_indices.get(),
        #                                  self.h2p_pointers.get()))

        # delta_i2h_before_hadamard = h2p_csr_weights.transpose().dot(self.delta_h2p.get())
        # print(self.delta_i2h.get() - delta_i2h_before_hadamard)
        #
        # del_i2h = self.delta_i2h.get() * self.a_prime_i2h.get()

        hadamard.prepared_call(self.grid_i2h, self.threads,
                               self.delta_i2h.gpudata, self.a_prime_i2h.gpudata,
                               self.delta_i2h.gpudata,
                               self.L_hidden)

        # print(del_i2h - self.delta_i2h.get())

        kron.prepared_async_call(self.grid_i2h, self.threads,
                                 self.stream1,
                                 self.i2h_weights_nnz,
                                 self.i2h_row_idx.gpudata,
                                 self.i2h_indices.gpudata,
                                 input_gpu.gpudata, self.delta_i2h.gpudata,
                                 self.grad_weight_i2h.gpudata)

        # grad_weight_i2h_cpu = cpu_kron(self.i2h_weights_nnz, self.i2h_row_idx.get(),
        #                                self.i2h_indices.get(), input_, self.delta_i2h.get())
        #
        # print(self.grad_weight_i2h.get() - grad_weight_i2h_cpu)

        kron.prepared_async_call(self.grid_h2p, self.threads,
                                 self.stream2,
                                 self.h2p_weights_nnz,
                                 self.h2p_row_idx.gpudata,
                                 self.h2p_indices.gpudata,
                                 self.hidden.gpudata,
                                 self.delta_h2p.gpudata,
                                 self.grad_weight_h2p.gpudata)

        # grad_weight_h2p_cpu = cpu_kron(self.h2p_weights_nnz, self.h2p_row_idx.get(),
        #                                self.h2p_indices.get(), self.hidden.get(),
        #                                self.delta_h2p.get())
        #
        # print(self.grad_weight_h2p.get() - grad_weight_h2p_cpu)
        #
        # i2h_weight_updated = self.i2h_weights.get() - learning_rate * self.grad_weight_i2h.get()
        # h2p_weight_updated = self.h2p_weights.get() - learning_rate * self.grad_weight_h2p.get()
        # i2h_bias_updated = self.i2h_bias.get() - learning_rate * self.delta_i2h.get()
        # h2p_bias_updated = self.h2p_bias.get() - learning_rate * self.delta_h2p.get()

        update.prepared_async_call(self.grid_update_i2h, self.threads,
                                   self.stream1,
                                   self.i2h_weights.gpudata,
                                   self.grad_weight_i2h.gpudata, learning_rate,
                                   self.i2h_weights_nnz)

        update.prepared_async_call(self.grid_update_h2p, self.threads,
                                   self.stream2,
                                   self.h2p_weights.gpudata,
                                   self.grad_weight_h2p.gpudata, learning_rate,
                                   self.h2p_weights_nnz)

        update.prepared_async_call(self.grid_i2h, self.threads,
                                   self.stream3,
                                   self.i2h_bias.gpudata, self.delta_i2h.gpudata,
                                   learning_rate, self.L_hidden)

        update.prepared_async_call(self.grid_h2p, self.threads,
                                   self.stream4,
                                   self.h2p_bias.gpudata,
                                   self.delta_h2p.gpudata, learning_rate,
                                   self.L_pred)

        # print('i2h_weights update')
        # print(self.i2h_weights.get() - i2h_weight_updated)
        #
        # print('h2p_weights update')
        # print(self.h2p_weights.get() - h2p_weight_updated)
        #
        # print('i2h_bias update')
        # print(self.i2h_bias.get() - i2h_bias_updated)
        #
        # print('h2p_bias update')
        # print(self.h2p_bias.get() - h2p_bias_updated)

    def train(self, train_dict, learning_rate_list, print_every=100):
        L_lr = len(learning_rate_list)
        n = 0
        while n < L_lr:
            for key, training_data in train_dict.items():
                input_, ideal_pred = training_data
                rows, cols = input_.shape

                for row in range(rows):
                    input_row = input_[row, :]
                    ideal_pred_row = ideal_pred[row, :]
                    self.forward(input_row)
                    self._backward(input_row, ideal_pred_row, np.float64(learning_rate_list[n]))
                    n += 1
                    se = sum((ideal_pred_row - self.pred.get()) ** 2) / 2.
                    self.sq_err.append(se)
                    if n % print_every == 0:
                        mse = sum(self.sq_err[-print_every:]) / \
                              print_every
                        print("{:>10} frames: {:>10}".format(n, mse))
                    if n == L_lr:
                        break
                if n == L_lr:
                    break

    @staticmethod
    def weight_initialize(spec_dict, hidden_size):
        # input to hidden
        i2h_pointers = [0]
        i2h_indices = []
        i2h_weights = []
        i2h_bias = []
        # hidden to predictions
        h2p_pointers = [0]
        h2p_indices = []
        h2p_weights = []
        h2p_bias = []

        sum_size = 0
        unit_count = 0
        for key, size in spec_dict.items():
            for _ in range(hidden_size):
                i2h_pointers.append(i2h_pointers[-1] + size)
                i2h_indices.append(sum_size + np.arange(size))
                i2h_weights.append(np.random.randn(size) / math.sqrt(size))
            i2h_bias.append(np.random.randn(hidden_size))
            sum_size += size

            for _ in range(size):
                h2p_pointers.append(h2p_pointers[-1] + hidden_size)
                h2p_indices.append(np.arange(unit_count * hidden_size,
                                             (unit_count + 1) * hidden_size))
                h2p_weights.append(np.random.randn(hidden_size) /
                                   math.sqrt(hidden_size))
            h2p_bias.append(np.random.randn(size))
            unit_count += 1

        i2h_pointers = np.array(i2h_pointers, dtype=np.int32)
        i2h_weights = np.concatenate(i2h_weights)
        i2h_bias = np.concatenate(i2h_bias)
        i2h_indices = np.concatenate(i2h_indices).astype(np.int32)

        h2p_pointers = np.array(h2p_pointers, dtype=np.int32)
        h2p_weights = np.concatenate(h2p_weights)
        h2p_bias = np.concatenate(h2p_bias)
        h2p_indices = np.concatenate(h2p_indices).astype(np.int32)

        return (i2h_pointers, i2h_indices, i2h_weights, i2h_bias), \
               (h2p_pointers, h2p_indices, h2p_weights, h2p_bias)

    @staticmethod
    def ptr_to_row(ptr):
        """
        Converts the input array-like of pointer (CSR format)
        to the equivalent array or rows for conversion of CSR
        to COO.

        :param ptr: An array like of pointers as is found in
        CSR format.
        :return: A numpy array for corresponding row indices of
        the input ptr
        """
        L_ptr = len(ptr)
        row_idx = []
        for idx, ptr_idx in enumerate(range(L_ptr - 1)):
            row_idx += [idx] * (ptr[ptr_idx + 1] - ptr[ptr_idx])
        return np.array(row_idx, dtype=np.int32)