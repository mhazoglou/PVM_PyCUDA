from __future__ import absolute_import, print_function, division
from builtins import *
import math
import cv2
import re
import pickle as cP
import numpy as np
import serial
import matplotlib.pyplot as plt
from collections import OrderedDict
from pycuda import driver, compiler, gpuarray, tools, curandom, cumath
from RectangularGridConstructor import make_connections
from WeightsAndBiasInitialization import weight_initialize,\
    tracker_weight_initialize, ptr_to_row

# -- initialize the device
# import pycuda.autoinit

fname = "./Useful_Kernels.cu"
tracker_filename = "./tracker_kernels.cu"
saccade_filename = "./Saccadic_kernels.cu"
adam_filename = "./adam_kernels.cu"

with open(fname) as file_id:
    kernel_code = file_id.read()

with open(tracker_filename) as file_id:
    tracker_kernel_code = file_id.read()

with open(saccade_filename) as file_id:
    saccadic_kernel_code = file_id.read()

with open(adam_filename) as file_id:
    adam_kernel_code = file_id.read()

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
dot = mod.get_function('Blocked1dCSRMatVecDotKernel')
dot.prepare([np.int32, 'P', 'P', 'P', 'P', 'P'])

# sparse matrix-transpose-vector multiplication using CSR format
Tdot = mod.get_function('spmTv_csr_kernel')
Tdot.prepare([np.int32, 'P', 'P', 'P', 'P', 'P'])

# dense outer product
dense_kron = mod.get_function('OuterProductKernel')
dense_kron.prepare(['P', 'P', 'P', np.int32, np.int32])

# dense matrix-vector multiplication using
# needs zero initialization for the result
dense_dot = mod.get_function('MatVecDotKernel')
dense_dot.prepare(['P', 'P', 'P', np.int32, np.int32])

# dense transpose-matrix-vector multiplication using
# needs zero initialization for the result
dense_Tdot = mod.get_function('MatTransposeVecDotKernel')
dense_Tdot.prepare(['P', 'P', 'P', np.int32, np.int32])

# element-wise application of the sigmoid function
sig = mod.get_function('SigmoidKernel')
sig.prepare(['P', 'P', np.int32])

# element-wise application of the derivative of the sigmoid
dsig = mod.get_function('SigmoidPrimeKernel')
dsig.prepare(['P', 'P', np.int32])

# element-wise application of the rectified linear function
tanh = mod.get_function('TanhKernel')
tanh.prepare(['P', 'P', np.int32])

# element-wise application of the derivative of the ReLU
dtanh = mod.get_function('TanhPrimeKernel')
dtanh.prepare(['P', 'P', np.int32])

# element-wise application of the rectified linear function
ReLU = mod.get_function('ReLUKernel')
ReLU.prepare(['P', 'P', np.int32])

# element-wise application of the derivative of the ReLU
dReLU = mod.get_function('ReLUPrimeKernel')
dReLU.prepare(['P', 'P', np.int32])

# element-wise application of the identity (linear) function
Identity = mod.get_function('IdentityKernel')
Identity.prepare(['P', 'P', np.int32])

# element-wise application of the derivative of the identity function
onefill = mod.get_function('OneFillKernel')
onefill.prepare(['P', 'P', np.int32])

# element-wise application of the identity (linear) function
SoftPlus = mod.get_function('SoftPlusKernel')
SoftPlus.prepare(['P', 'P', np.int32])

# updating weights and biases
update = mod.get_function('UpdateKernel')
update.prepare(['P', 'P', np.float64, np.int32])

# partial reduction of to find maximum and and argmax
max_gpu = mod.get_function('MaxKernel')
max_gpu.prepare(['P', 'P', 'P', np.int32])

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

grad_inv_hid_map = mod_tracker.get_function('gradient_inv_hid_map_kernel')
grad_inv_hid_map.prepare(['P', 'P', 'P', np.int32])

#---------------------------adam functions-------------------------------
mod_adam = compiler.SourceModule(adam_kernel_code)

axpby = mod_adam.get_function('axpbyKernel')
axpby.prepare(['P', 'P', np.float64, np.float64, np.int32])

axpbyy = mod_adam.get_function('axpbyyKernel')
axpbyy.prepare(['P', 'P', np.float64, np.float64, np.int32])

gradmod = mod_adam.get_function('gradModKernel')
gradmod.prepare(['P', 'P', 'P', np.float64, np.int32])

# --------------------saccade functions--------------------
# compile the kernel code
mod_saccade = compiler.SourceModule(saccadic_kernel_code)

err_patch_sum = mod_saccade.get_function('PatchedSumImageKernel')
err_patch_sum.prepare(['P', 'P', np.int32, np.int32, np.int32, np.int32,
                       np.int32])


def activation(name_of_activation_function):
    match_sig = re.fullmatch('sigmoid',
                             name_of_activation_function,
                             flags=re.IGNORECASE)
    match_tanh = re.fullmatch('tanh',
                             name_of_activation_function,
                             flags=re.IGNORECASE)
    match_relu = re.fullmatch('relu',
                              name_of_activation_function,
                              flags=re.IGNORECASE)
    match_id = re.fullmatch('id(entity)?',
                            name_of_activation_function,
                            flags=re.IGNORECASE)
    match_lin = re.fullmatch('lin(ear)?',
                             name_of_activation_function,
                             flags=re.IGNORECASE)
    match_soft_plus = re.fullmatch('soft_?plus',
                                   name_of_activation_function,
                                   flags=re.IGNORECASE)
    if match_sig:
        return sig, dsig
    elif match_tanh:
        return tanh, dtanh
    elif match_relu:
        return ReLU, dReLU
    elif match_id or match_lin:
        return Identity, onefill
    elif match_soft_plus:
        return SoftPlus, sig
    else:
        raise ValueError("Not a valid activation unit")


def regularize(x):
    if x > 1:
        return 1.
    elif x < 0:
        return 0.
    else:
        return x


def movingaverage(interval, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(interval, window, 'same')


def new_block_idx(new_idx_v_arr, new_idx_r_arr):
    block_idx_list = []
    prev_idx_V, prev_idx_R = 0, 0
    for idx_V, idx_R in zip(new_idx_v_arr, new_idx_r_arr):
        L_V = idx_V - prev_idx_V
        L_R = idx_R - prev_idx_R
        if block_idx_list != []:
            block_idx_list.append(block_idx_list[-1] +
                                  L_V * L_R)
            if L_V > max_L_V:
                max_L_V = L_V
            if L_R > max_L_R:
                max_L_R = L_R
        else:
            block_idx_list.append(L_V * L_R)
            max_L_V = L_V
            max_L_R = L_R
        prev_idx_V, prev_idx_R = idx_V, idx_R
    return block_idx_list, max_L_V, max_L_R


def base_pvm_gpu_forward(self, input_):
    # This step is to add the inputs for the upper layers which
    # are the hidden from the layers below
    try:
        append_hid.prepared_call(self.grid_int_der_err, self.threads,
                                 input_.gpudata, self.hidden.gpudata,
                                 self.in_and_hid.gpudata,
                                 self.hid_append_map.gpudata,
                                 self.L_input, self.L_pred)
    except:
        self.in_and_hid = input_[:]

    # calculating the derivatives, errors and integrals
    der_and_error.prepared_async_call(self.grid_int_der_err, self.threads,
                                      self.stream1,
                                      self.in_and_hid.gpudata,
                                      self.prev_input.gpudata,
                                      self.der.gpudata, self.L_pred)
    der_and_error.prepared_async_call(self.grid_int_der_err, self.threads,
                                      self.stream2,
                                      self.pred.gpudata,
                                      self.in_and_hid.gpudata,
                                      self.err.gpudata, self.L_pred)
    integral.prepared_async_call(self.grid_int_der_err, self.threads,
                                 self.stream3,
                                 self.in_and_hid.gpudata,
                                 self.int_.gpudata,
                                 self.int_.gpudata, self.L_pred)

    # shuffling the derivative, error, integral, and hidden for
    # the full_input
    input_shuffling.prepared_async_call(self.grid_int_der_err,
                                        self.threads, self.stream4,
                                        self.full_input.gpudata,
                                        input_.gpudata,
                                        self.input_map.gpudata,
                                        self.L_input)

    input_hidden_shuffling.prepared_async_call(self.grid_i2h,
                                               self.threads, self.stream5,
                                               self.full_input.gpudata,
                                               self.hidden.gpudata,
                                               self.hid_map.gpudata,
                                               self.L_full_input)

    input_shuffling.prepared_async_call(self.grid_int_der_err,
                                        self.threads, self.stream1,
                                        self.full_input.gpudata,
                                        self.der.gpudata,
                                        self.der_map.gpudata, self.L_pred)

    input_shuffling.prepared_async_call(self.grid_int_der_err,
                                        self.threads, self.stream2,
                                        self.full_input.gpudata,
                                        self.err.gpudata,
                                        self.err_map.gpudata, self.L_pred)

    input_shuffling.prepared_async_call(self.grid_int_der_err,
                                        self.threads, self.stream3,
                                        self.full_input.gpudata,
                                        self.int_.gpudata,
                                        self.int_map.gpudata, self.L_pred)
    zerofill.prepared_async_call(self.grid_i2h, self.threads,
                                 self.stream4,
                                 self.hid_affine.gpudata, self.L_hidden)
    zerofill.prepared_async_call(self.grid_h2op, self.threads,
                                 self.stream5,
                                 self.out_and_pred_affine.gpudata,
                                 self.L_op)

    # storing the
    self.prev_input = self.in_and_hid.copy()

    dot.prepared_call(self.grid_i2h_dot, self.threads,
                      # number of rows
                      self.L_hidden,
                      # CSR sparse matrix
                      self.i2h_pointers.gpudata, self.i2h_indices.gpudata,
                      self.i2h_weights.gpudata,
                      # vector
                      self.full_input.gpudata,
                      # result
                      self.hid_affine.gpudata)

    add.prepared_call(self.grid_i2h, self.threads,
                      self.hid_affine.gpudata, self.i2h_bias.gpudata,
                      self.hid_affine.gpudata, self.L_hidden)

    self.act.prepared_call(self.grid_i2h, self.threads,
                           self.hid_affine.gpudata, self.hidden.gpudata,
                           self.L_hidden)

    dot.prepared_call(self.grid_h2op_dot, self.threads,
                      # number of rows
                      self.L_op,
                      # CSR sparse matrix
                      self.h2op_pointers.gpudata,
                      self.h2op_indices.gpudata,
                      self.h2op_weights.gpudata,
                      # vector
                      self.hidden.gpudata,
                      # results
                      self.out_and_pred_affine.gpudata)

    add.prepared_call(self.grid_h2op, self.threads,
                      self.out_and_pred_affine.gpudata,
                      self.h2op_bias.gpudata,
                      self.out_and_pred_affine.gpudata,
                      self.L_op)

    self.act.prepared_call(self.grid_h2op, self.threads,
                           self.out_and_pred_affine.gpudata,
                           self.out_and_pred.gpudata,
                           self.L_op)

    if self.L_out > 0:
        output_pred_shuffling.prepared_async_call(
            self.grid_o_shuf,
            self.threads,
            self.stream4,
            self.output.gpudata,
            self.out_and_pred.gpudata,
            self.out_map.gpudata,
            self.L_out
        )

    output_pred_shuffling.prepared_async_call(
        self.grid_int_der_err,
        self.threads, self.stream5,
        self.pred.gpudata,
        self.out_and_pred.gpudata,
        self.pred_map.gpudata,
        self.L_pred
    )


def base_pvm_gpu_backprop(self, next_input):
    self.dact.prepared_async_call(self.grid_i2h, self.threads,
                                  self.stream1,
                                  self.hid_affine.gpudata,
                                  self.a_prime_i2h.gpudata,
                                  self.L_hidden)

    self.dact.prepared_async_call(self.grid_h2op, self.threads,
                                  self.stream2,
                                  self.out_and_pred_affine.gpudata,
                                  self.a_prime_h2op.gpudata, self.L_op)

    # ignore appending the hidden values fed to upper layers to the input
    try:
        append_hid.prepared_call(self.grid_int_der_err, self.threads,
                                 next_input.gpudata, self.hidden.gpudata,
                                 self.ideal_pred.gpudata,
                                 self.hid_append_map.gpudata,
                                 self.L_input,
                                 self.L_pred)
    except:
        self.ideal_pred = next_input[:]

    # calculating the gradients in error using MSE
    sub.prepared_async_call(self.grid_int_der_err, self.threads,
                            self.stream1,
                            self.pred.gpudata, self.ideal_pred.gpudata,
                            self.delta_pred.gpudata, self.L_pred)

    hadamard.prepared_async_call(self.grid_int_der_err, self.threads,
                                 self.stream1,
                                 self.delta_pred.gpudata,
                                 self.pred_mask.gpudata,
                                 self.delta_pred.gpudata, self.L_pred)

    if self.L_out > 0:
        zerofill.prepared_async_call(
            self.grid_o_shuf,
            self.threads,
            self.stream2,
            self.delta_output.gpudata,
            self.L_out
        )

        rev_output_pred_shuffling.prepared_async_call(
            self.grid_o_shuf,
            self.threads,
            self.stream2,
            self.delta_output.gpudata,
            self.delta_h2op.gpudata,
            self.out_map.gpudata,
            self.L_out
        )

    rev_output_pred_shuffling.prepared_async_call(self.grid_int_der_err,
                                                  self.threads,
                                                  self.stream1,
                                                  self.delta_pred.gpudata,
                                                  self.delta_h2op.gpudata,
                                                  self.pred_map.gpudata,
                                                  self.L_pred)

    hadamard.prepared_call(self.grid_h2op, self.threads,
                           self.delta_h2op.gpudata,
                           self.a_prime_h2op.gpudata,
                           self.delta_h2op.gpudata, self.L_op)

    zerofill.prepared_call(self.grid_i2h, self.threads,
                           self.delta_i2h.gpudata, self.L_hidden)

    Tdot.prepared_call(self.grid_h2op, self.threads,
                       self.L_op,
                       self.h2op_pointers.gpudata,
                       self.h2op_indices.gpudata,
                       self.h2op_weights.gpudata,
                       self.delta_h2op.gpudata,
                       self.delta_i2h.gpudata)

    hadamard.prepared_call(self.grid_i2h, self.threads,
                           self.delta_i2h.gpudata,
                           self.a_prime_i2h.gpudata,
                           self.delta_i2h.gpudata, self.L_hidden)

    kron.prepared_async_call(self.grid_i2h, self.threads,
                             self.stream1,
                             self.i2h_weights_nnz,
                             self.i2h_row_idx.gpudata,
                             self.i2h_indices.gpudata,
                             self.full_input.gpudata, self.delta_i2h.gpudata,
                             self.grad_weight_i2h.gpudata)

    kron.prepared_async_call(self.grid_h2op, self.threads,
                             self.stream2,
                             self.h2op_weights_nnz,
                             self.h2op_row_idx.gpudata,
                             self.h2op_indices.gpudata,
                             self.hidden.gpudata,
                             self.delta_h2op.gpudata,
                             self.grad_weight_h2op.gpudata)


def update_parameters(self, learning_rate, reg_factor=0):
    """Called after base_pvm_backprop for updating
    parameter values"""
    f = 2 * learning_rate * reg_factor
    update.prepared_async_call(self.grid_i2h, self.threads,
                               self.stream1,
                               self.i2h_bias.gpudata,
                               self.delta_i2h.gpudata,
                               learning_rate, self.L_hidden)

    update.prepared_async_call(self.grid_h2op, self.threads,
                               self.stream2,
                               self.h2op_bias.gpudata,
                               self.delta_h2op.gpudata,
                               learning_rate, self.L_op)

    update.prepared_async_call(self.grid_update_i2h, self.threads,
                               self.stream1,
                               self.i2h_weights.gpudata,
                               self.i2h_weights.gpudata,
                               f,
                               self.i2h_weights_nnz)

    update.prepared_async_call(self.grid_update_i2h, self.threads,
                               self.stream1,
                               self.i2h_weights.gpudata,
                               self.grad_weight_i2h.gpudata,
                               learning_rate, self.i2h_weights_nnz)

    update.prepared_async_call(self.grid_update_h2op, self.threads,
                               self.stream2,
                               self.h2op_weights.gpudata,
                               self.h2op_weights.gpudata,
                               f,
                               self.h2op_weights_nnz)

    update.prepared_async_call(self.grid_update_h2op, self.threads,
                               self.stream2,
                               self.h2op_weights.gpudata,
                               self.grad_weight_h2op.gpudata,
                               learning_rate, self.h2op_weights_nnz)


def update_weights(self, learning_rate, reg_factor=0):
    """Called after base_pvm_backprop for updating
    weight values"""
    f = 2 * learning_rate * reg_factor

    update.prepared_async_call(self.grid_update_i2h, self.threads,
                               self.stream1,
                               self.i2h_weights.gpudata,
                               self.i2h_weights.gpudata,
                               f,
                               self.i2h_weights_nnz)

    update.prepared_async_call(self.grid_update_i2h, self.threads,
                               self.stream1,
                               self.i2h_weights.gpudata,
                               self.grad_weight_i2h.gpudata,
                               learning_rate, self.i2h_weights_nnz)

    update.prepared_async_call(self.grid_update_h2op, self.threads,
                               self.stream2,
                               self.h2op_weights.gpudata,
                               self.h2op_weights.gpudata,
                               f,
                               self.h2op_weights_nnz)

    update.prepared_async_call(self.grid_update_h2op, self.threads,
                               self.stream2,
                               self.h2op_weights.gpudata,
                               self.grad_weight_h2op.gpudata,
                               learning_rate, self.h2op_weights_nnz)


def adam_update_parameters(self, t, alpha=0.0001, beta1=0.9,
                           beta2=0.999, eps=1e-8, reg_factor=0.):
    axpby.prepared_async_call(self.grid_h2op, self.threads,
                              self.stream1,
                              self.m_t_bias_h2op.gpudata,
                              self.grad_bias_h2op.gpudata,
                              beta1, 1. - beta1,
                              self.L_op)
    axpbyy.prepared_async_call(self.grid_h2op, self.threads,
                               self.stream2,
                               self.v_t_bias_h2op.gpudata,
                               self.grad_bias_h2op.gpudata,
                               beta2, 1. - beta2,
                               self.L_op)
    axpby.prepared_async_call(self.grid_h2op, self.threads,
                              self.stream1,
                              self.mhat_t_bias_h2op.gpudata,
                              self.m_t_bias_h2op.gpudata,
                              0, 1. / (1. - beta1 ** t),
                              self.L_op)
    axpby.prepared_async_call(self.grid_h2op, self.threads,
                              self.stream1,
                              self.vhat_t_bias_h2op.gpudata,
                              self.v_t_bias_h2op.gpudata,
                              0, 1. / (1. - beta2 ** t),
                              self.L_op)

    axpby.prepared_async_call(self.grid_i2h, self.threads,
                              self.stream3,
                              self.m_t_bias_i2h.gpudata,
                              self.grad_bias_i2h.gpudata,
                              beta1, 1. - beta1,
                              self.L_hidden)
    axpbyy.prepared_async_call(self.grid_i2h, self.threads,
                               self.stream4,
                               self.v_t_bias_i2h.gpudata,
                               self.grad_bias_i2h.gpudata,
                               beta2, 1. - beta2,
                               self.L_hidden)
    axpby.prepared_async_call(self.grid_i2h, self.threads,
                              self.stream3,
                              self.mhat_t_bias_i2h.gpudata,
                              self.m_t_bias_i2h.gpudata,
                              0, 1. / (1. - beta1 ** t),
                              self.L_hidden)
    axpby.prepared_async_call(self.grid_i2h, self.threads,
                              self.stream4,
                              self.vhat_t_bias_i2h.gpudata,
                              self.v_t_bias_i2h.gpudata,
                              0, 1. / (1. - beta2 ** t),
                              self.L_hidden)


    gradmod.prepared_call(self.grid_h2op, self.threads,
                          self.mhat_t_bias_h2op.gpudata,
                          self.vhat_t_bias_h2op.gpudata,
                          self.dev_bias_h2op.gpudata,
                          eps, self.L_op)
    gradmod.prepared_call(self.grid_i2h, self.threads,
                          self.mhat_t_bias_i2h.gpudata,
                          self.vhat_t_bias_i2h.gpudata,
                          self.dev_bias_i2h.gpudata,
                          eps, self.L_hidden)


    reg_h2op_weights_grad = (
        self.grad_weight_h2op *
        (1. + 2 * reg_factor * self.h2op_weights)
    )

    reg_i2h_weights_grad = (
        self.grad_weight_i2h *
        (1. + 2 * reg_factor * self.i2h_weights)
    )

    axpby.prepared_async_call(self.grid_update_h2op, self.threads,
                              self.stream1,
                              self.m_t_weights_h2op.gpudata,
                              reg_h2op_weights_grad.gpudata,
                              beta1, 1. - beta1,
                              self.h2op_weights_nnz)
    axpbyy.prepared_async_call(self.grid_update_h2op, self.threads,
                               self.stream2,
                               self.v_t_weights_h2op.gpudata,
                               reg_h2op_weights_grad.gpudata,
                               beta2, 1. - beta2,
                               self.h2op_weights_nnz)
    axpby.prepared_async_call(self.grid_update_h2op, self.threads,
                              self.stream1,
                              self.mhat_t_weights_h2op.gpudata,
                              self.m_t_weights_h2op.gpudata,
                              0, 1. / (1. - beta1 ** t),
                              self.h2op_weights_nnz)
    axpby.prepared_async_call(self.grid_update_h2op, self.threads,
                              self.stream2,
                              self.vhat_t_weights_h2op.gpudata,
                              self.v_t_weights_h2op.gpudata,
                              0, 1. / (1. - beta2 ** t),
                              self.h2op_weights_nnz)

    axpby.prepared_async_call(self.grid_update_i2h, self.threads,
                              self.stream3,
                              self.m_t_weights_i2h.gpudata,
                              reg_i2h_weights_grad.gpudata,
                              beta1, 1. - beta1,
                              self.i2h_weights_nnz)
    axpbyy.prepared_async_call(self.grid_update_i2h, self.threads,
                               self.stream4,
                               self.v_t_weights_i2h.gpudata,
                               reg_i2h_weights_grad.gpudata,
                               beta2, 1. - beta2,
                               self.i2h_weights_nnz)
    axpby.prepared_async_call(self.grid_update_i2h, self.threads,
                              self.stream3,
                              self.mhat_t_weights_i2h.gpudata,
                              self.m_t_weights_i2h.gpudata,
                              0, 1. / (1. - beta1 ** t),
                              self.i2h_weights_nnz)
    axpby.prepared_async_call(self.grid_update_i2h, self.threads,
                              self.stream4,
                              self.vhat_t_weights_i2h.gpudata,
                              self.v_t_weights_i2h.gpudata,
                              0, 1. / (1. - beta2 ** t),
                              self.i2h_weights_nnz)


    gradmod.prepared_call(self.grid_update_h2op, self.threads,
                          self.mhat_t_weights_h2op.gpudata,
                          self.vhat_t_weights_h2op.gpudata,
                          self.dev_weights_h2op.gpudata,
                          eps, self.h2op_weights_nnz)
    gradmod.prepared_call(self.grid_update_i2h, self.threads,
                          self.mhat_t_weights_i2h.gpudata,
                          self.vhat_t_weights_i2h.gpudata,
                          self.dev_weights_i2h.gpudata,
                          eps, self.i2h_weights_nnz)


    update.prepared_async_call(self.grid_i2h, self.threads,
                               self.stream1,
                               self.i2h_bias.gpudata,
                               self.dev_bias_i2h.gpudata,
                               alpha, self.L_hidden)

    update.prepared_async_call(self.grid_h2op, self.threads,
                               self.stream2,
                               self.h2op_bias.gpudata,
                               self.dev_bias_h2op.gpudata,
                               alpha, self.L_op)

    update.prepared_async_call(self.grid_update_i2h, self.threads,
                               self.stream1,
                               self.i2h_weights.gpudata,
                               self.dev_weights_i2h.gpudata,
                               alpha, self.i2h_weights_nnz)

    update.prepared_async_call(self.grid_update_h2op, self.threads,
                               self.stream2,
                               self.h2op_weights.gpudata,
                               self.dev_weights_h2op.gpudata,
                               alpha, self.h2op_weights_nnz)


class PVM_CUDA(object):
    """
    A class for the basic implementation of the PVM on Nvidia GPUs using
    PyCUDA

    Attributes:
        mse: mean square error in a list, it is available only after
            using the train method
        connect_dict: An ordered dictionary that
            maps a string representing a PVM unit to a tuple with

            (unit_count, hidden_size, output_size, fedfrom_list, latsup_list)

            unit_count is a numerical value associated with the key
            hidden_size is the same as the parameter hidden_size it is the
            hidden_size of each PVM unit
        i2h_pointers: A gpuarray abstraction for the input to hidden
            calculation. It is a long int array stored on the GPU, it
            contains the pointers for a CSR array representation of the
            i2h_weights
        i2h_indices: A gpuarray abstraction for the input to hidden
            calculation. It is a long int array stored on the GPU of the
            column indices for a CSR array representation of the i2h_weights
        i2h_weights: A gpuarray abstraction for the input to hidden
            calculation. It is a double precision float array stored on the
            GPU of the weights for a CSR array of the matrix-vector product.
        i2h_bias: A gpuarray abstraction for the input to hidden calculation.
            It is a double precision float array stored on the GPU of the
            biases for the input to hidden calculation.
        i2h_row_idx: A gpuarray abstraction for the conversion of the CSR
            array representation of the input to hidden weights into a COO
            format, it contains the row indices of the elements in
            i2h_weights.
        h2op_pointers: A gpuarray abstraction for the hidden to output and
            prediction calculation. It is a long int array stored on the GPU,
            it contains the pointers of the CSR format of the weights.
        h2op_indices: A gpuarray abstraction for the hidden to output and
            prediction calculation. It is a long int array stored on the GPU,
            it contains the column indices of the elements in h2op_weights.
        h2op_weights: A gpuarray abstraction for the hidden to output and
            prediction calculation. It is a double precision float array
            stored on the GPU, it contains the non-zero values of the
            weights in a CSR format.
        h2op_bias: A gpuarray abstraction for the hidden to output and
            prediction calculation. It is a double precision float array
            stored on the GPU, it contains the biases used in the hidden
            to output and prediction calculation
        h2op_row_idx: A gpuarray abstraction for the conversion of the CSR
            array representation of the input to hidden weights into a COO
            format, it contains the row indices of the elements in
            h2op_weights.
        input_map: A gpuarray abstraction for mapping the inputs to the
            correct location for the corresponding PVM unit.
        der_map: A gpuarray abstraction for mapping the derivatives to the
            correct location for the corresponding PVM unit.
        int_map: A gpuarray abstraction for mapping the integral to the
            correct location for the corresponding PVM unit.
        err_map: A gpuarray abstraction for mapping the errors to the
            correct location for the corresponding PVM unit.
        hid_map: A gpuarray abstraction for mapping the hiddens to the
            correct location for the corresponding PVM unit.
        hid_append_map: A gpuarray abstraction for appending the hiddens
            to the end of the input for a more convenient error,
            derivative and integral calculation
        out_map: A gpuarray abstraction for mapping the output from
            out_and_pred
        pred_map: A gpuaray abstraction for mapping the prediction from
            out_and_pred
        L_input: Length of the input in np.int32 format
        i2h_weights_nnz: The number of non-zero elements in the input to
            hidden weight matrix in np.int32 format
        h2op_weights_nnz: The number of non-zero elements in the hidden
            to output and prediction calculation in np.int32 format.
        L_hidden_inputs: The length of
    """
    def __init__(self, connect_dict, act_type='sigmoid',
                 maxthreads=1024, max_grid_size=65535, zero_bias=False):
        self.mse = []
        self.img_mse = []
        self.connect_dict = connect_dict

        act, dact = activation(act_type)

        self.act = act
        self.dact = dact

        i2h_stuff, h2op_stuff, map_stuff, input_new_unit, op_new_unit,\
            hid_new_unit = weight_initialize(connect_dict)

        # input (plus context) to hidden weights and biases
        # the weights are in a CSR format
        self.i2h_pointers = gpuarray.to_gpu(i2h_stuff[0])
        self.i2h_indices  = gpuarray.to_gpu(i2h_stuff[1])
        self.i2h_weights  = gpuarray.to_gpu(i2h_stuff[2])
        self.i2h_bias     = gpuarray.to_gpu(i2h_stuff[3])

        # for efficient conversion to COO format which improves outer
        # product performance in back propagation
        self.i2h_row_idx  = gpuarray.to_gpu(ptr_to_row(i2h_stuff[0]))

        # hidden to output plus prediction weights and biases
        # the weights are in CSR format
        self.h2op_pointers = gpuarray.to_gpu(h2op_stuff[0])
        self.h2op_indices  = gpuarray.to_gpu(h2op_stuff[1])
        self.h2op_weights  = gpuarray.to_gpu(h2op_stuff[2])
        self.h2op_bias     = gpuarray.to_gpu(h2op_stuff[3])

        # for efficient conversion to COO format which improves outer
        # product performance in back propagation
        self.h2op_row_idx  = gpuarray.to_gpu(ptr_to_row(h2op_stuff[0]))

        if zero_bias:
            self._update = update_weights
            self.i2h_bias  = gpuarray.zeros_like(self.i2h_bias)
            self.h2op_bias = gpuarray.zeros_like(self.h2op_bias)
        else:
            self._update = update_parameters

        # all the different mapping arrays
        self.input_map      = gpuarray.to_gpu(np.array(map_stuff[0],
                                                       dtype=np.int32))
        self.der_map        = gpuarray.to_gpu(np.array(map_stuff[1],
                                                       dtype=np.int32))
        self.int_map        = gpuarray.to_gpu(np.array(map_stuff[2],
                                                       dtype=np.int32))
        self.err_map        = gpuarray.to_gpu(np.array(map_stuff[3],
                                                       dtype=np.int32))
        self.hid_map        = gpuarray.to_gpu(np.array(map_stuff[4],
                                                       dtype=np.int32))
        self.hid_append_map = gpuarray.to_gpu(np.array(map_stuff[5],
                                                       dtype=np.int32))
        self.out_map        = gpuarray.to_gpu(np.array(map_stuff[6],
                                                       dtype=np.int32))
        self.pred_map       = gpuarray.to_gpu(np.array(map_stuff[7],
                                                       dtype=np.int32))

        self.L_input          = np.int32(len(self.input_map))
        self.i2h_weights_nnz  = np.int32(i2h_stuff[0][-1])
        self.h2op_weights_nnz = np.int32(h2op_stuff[0][-1])

        # length of hidden as inputs
        self.L_hidden_inputs = np.int32(len(self.hid_append_map))

        # L_pred == len(der_map) == len(int_map) == len(err_map)
        # same as L_hidden_inputs + L_input
        self.L_pred = np.int32(len(self.pred_map))
        # length of inputs plus context
        self.L_full_input = np.int32(len(self.hid_map))
        self.L_out        = np.int32(len(self.out_map))
        self.L_op         = np.int32(self.L_out + self.L_pred)

        # mask that allows you to ignore a particular input during training
        # by default it will not remove anything from the prediction
        # unless changed
        self.pred_mask = gpuarray.zeros(self.L_pred, dtype=np.float64) + 1.

        self.N_units  = np.int32(len(connect_dict))
        self.L_hidden = hid_new_unit[-1]

        block_idx_list, max_L_V, max_L_R = new_block_idx(input_new_unit,
                                                         hid_new_unit)

        self.threads = (maxthreads, 1, 1)
        grid_x_i2h = min(int(math.ceil(self.L_hidden / maxthreads)),
                         max_grid_size)
        grid_x_i2h_dot = min(int(math.ceil(max_L_V / maxthreads)),
                             max_grid_size)
        grid_y_i2h_dot = min(int(self.L_hidden), max_grid_size)
        grid_x_input = min(int(math.ceil(self.L_input / maxthreads)),
                           max_grid_size)
        grid_x_int_der_err = min(int(math.ceil(self.L_pred / maxthreads)),
                                 max_grid_size)
        grid_x_h2op = min(int(math.ceil(self.L_op / maxthreads)),
                          max_grid_size)
        grid_x_h2op_dot = min(int(math.ceil(max_L_R / maxthreads)),
                              max_grid_size)
        grid_y_h2op_dot = min(int(self.L_op), max_grid_size)
        grid_x_o_shuf = min(int(math.ceil(self.L_out / maxthreads)),
                            max_grid_size)
        grid_x_update_i2h = min(int(math.ceil(self.i2h_weights_nnz /
                                              maxthreads)),
                                max_grid_size)
        grid_x_update_h2op = min(int(math.ceil(self.h2op_weights_nnz /
                                               maxthreads)),
                                 max_grid_size)
        self.grid_i2h         = (grid_x_i2h, 1)
        self.grid_i2h_dot     = (grid_x_i2h_dot, grid_y_i2h_dot)
        self.grid_input       = (grid_x_input, 1)
        self.grid_int_der_err = (grid_x_int_der_err, 1)
        self.grid_h2op        = (grid_x_h2op, 1)
        self.grid_h2op_dot    = (grid_x_h2op_dot, grid_y_h2op_dot)
        self.grid_o_shuf      = (grid_x_o_shuf, 1)
        self.grid_update_i2h  = (grid_x_update_i2h, 1)
        self.grid_update_h2op = (grid_x_update_h2op, 1)

        self.stream1 = driver.Stream()
        self.stream2 = driver.Stream()
        self.stream3 = driver.Stream()
        self.stream4 = driver.Stream()
        self.stream5 = driver.Stream()
        # self.stream6 = driver.Stream()
        # self.stream7 = driver.Stream()
        # self.stream8 = driver.Stream()

        self.hidden     = gpuarray.to_gpu(np.zeros(self.L_hidden))
        self.hid_affine = gpuarray.empty_like(self.hidden)

        self.hidden_inputs = gpuarray.to_gpu(np.zeros(self.L_hidden_inputs))
        self.in_and_hid    = gpuarray.to_gpu(np.zeros(self.L_pred))

        self.der  = gpuarray.to_gpu(np.zeros(self.L_pred))
        self.err  = gpuarray.to_gpu(np.zeros(self.L_pred))
        self.int_ = gpuarray.to_gpu(np.zeros(self.L_pred))

        self.out_and_pred        = gpuarray.to_gpu(np.zeros(self.L_op))
        self.out_and_pred_affine = gpuarray.empty_like(self.out_and_pred)

        self.pred   = gpuarray.to_gpu(np.zeros(self.L_pred))
        self.output = gpuarray.to_gpu(np.zeros(self.L_out))

        self.prev_input = gpuarray.to_gpu(np.zeros(self.L_pred))
        self.full_input = gpuarray.to_gpu(np.zeros(self.L_full_input))

        # variables for back-propagation
        self.ideal_pred = gpuarray.to_gpu(np.zeros(self.L_pred))

        self.delta_output = gpuarray.to_gpu(np.zeros(self.L_out))
        self.delta_pred   = gpuarray.to_gpu(np.zeros(self.L_pred))
        self.delta_h2op   = gpuarray.to_gpu(np.zeros(self.L_op))
        self.delta_i2h    = gpuarray.to_gpu(np.zeros(self.L_hidden))

        self.a_prime_h2op = gpuarray.empty_like(self.out_and_pred)
        self.a_prime_i2h  = gpuarray.empty_like(self.hidden)

        self.grad_weight_h2op = gpuarray.empty_like(self.h2op_weights)
        self.grad_weight_i2h  = gpuarray.empty_like(self.i2h_weights)

        self.grad_bias_h2op = self.delta_h2op
        self.grad_bias_i2h  = self.delta_i2h


    def reset_state(self):
        zerofill.prepared_async_call(self.grid_i2h, self.threads,
                                     self.stream1,
                                     self.hidden.gpudata, self.L_hidden)
        zerofill.prepared_async_call(self.grid_i2h, self.threads,
                                     self.stream2,
                                     self.hid_affine.gpudata, self.L_hidden)
        zerofill.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream3,
                                     self.prev_input.gpudata, self.L_pred)
        zerofill.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream4,
                                     self.int_.gpudata, self.L_pred)
        zerofill.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream5,
                                     self.pred.gpudata, self.L_pred)
        zerofill.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream1,
                                     self.err.gpudata, self.L_pred)
        zerofill.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream2,
                                     self.der.gpudata, self.L_pred)
        zerofill.prepared_async_call(self.grid_h2op, self.threads,
                                     self.stream3,
                                     self.out_and_pred.gpudata, self.L_op)
        zerofill.prepared_async_call(self.grid_h2op, self.threads,
                                     self.stream4,
                                     self.out_and_pred_affine.gpudata,
                                     self.L_op)
        if self.L_out > 0:
            zerofill.prepared_async_call(self.grid_o_shuf, self.threads,
                                         self.stream5,
                                         self.output.gpudata, self.L_out)

    def forward(self, single_frame):
        assert single_frame.size == self.L_input,\
            "Input frame must match specified input size"
        input_ = gpuarray.to_gpu(single_frame)

        base_pvm_gpu_forward(self, input_)

    def backward(self, single_frame, next_frame, learning_rate,
                 L2_norm_reg=0):
        assert len(single_frame) == len(next_frame),\
            "Both frames must be of the same size"
        next_input = gpuarray.to_gpu(next_frame)

        self.forward(single_frame)

        # input_ = gpuarray.to_gpu(single_frame)
        base_pvm_gpu_backprop(self, next_input)
        self._update(self, learning_rate, reg_factor=L2_norm_reg)

    def train(self, dict_training_data, learning_rate_list, L2_norm_reg=0,
              print_every=100000, save_every_print=False,
              filename='default', interval=100):
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        N_frames = len(learning_rate_list)
        while n < N_frames:
            for key, data in dict_training_data.items():
                n_frames, *_ = data.shape
                self.reset_state()

                for i in range(n_frames - 1):
                    learning_rate = learning_rate_list[n]

                    n += 1
                    single_frame = data[i, ...]
                    next_frame   = data[i+1, ...]
                    self.backward(single_frame, next_frame,
                                  learning_rate, L2_norm_reg=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred)**2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) /\
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse,
                                                   interval), 'b')
                            plt.plot(movingaverage(self.img_mse,
                                                   interval), 'r')
                            plt.legend(['Training error',
                                        'Training error (image only)'])
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()
                    if n == N_frames:
                        break
                if n == N_frames:
                    break

    def adam_train(self, dict_training_data, N_epoch, L2_norm_reg=0,
              print_every=100000, save_every_print=False,
              filename='default', interval=100,
              alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-8):
        # attribute outside of __init__ but nobody cares
        self.m_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.v_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.mhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.vhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.dev_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)

        self.m_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.v_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.mhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.vhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.dev_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)

        self.m_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.v_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.mhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.vhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.dev_weights_h2op =gpuarray.zeros_like(self.grad_weight_h2op)

        self.m_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.v_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.mhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.vhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.dev_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        for _ in range(N_epoch):
            for key, data in dict_training_data.items():
                n_frames, *junk = data.shape
                self.reset_state()

                self.forward(data[0, ...])
                for i in range(1, n_frames - 1):
                    n += 1
                    single_frame = data[i, ...]
                    next_frame = data[i + 1, ...]
                    next_input = gpuarray.to_gpu(next_frame)


                    self.forward(single_frame)

                    base_pvm_gpu_backprop(self, next_input)
                    adam_update_parameters(self, n,
                                           alpha=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           eps=eps,
                                           reg_factor=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred)**2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) /\
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse,
                                                   interval), 'b')
                            plt.plot(movingaverage(self.img_mse,
                                                   interval), 'r')
                            plt.legend(['Training error',
                                        'Training error (image only)'])
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()

    def save_parameters(self, filename):
        with open(filename + '_connections.pkl', 'wb') as fid:
            cP.dump(self.connect_dict, fid)
        np.savez(filename, i2h_pointers=self.i2h_pointers.get(),
                 i2h_indices=self.i2h_indices.get(),
                 i2h_weights=self.i2h_weights.get(),
                 i2h_bias=self.i2h_bias.get(),
                 h2op_pointers=self.h2op_pointers.get(),
                 h2op_indices=self.h2op_indices.get(),
                 h2op_weights=self.h2op_weights.get(),
                 h2op_bias=self.h2op_bias.get())

    def load_parameters(self, filename):
        with open(filename + '_connections.pkl', 'rb') as fid:
            should_be_connect_dict = cP.load(fid)
        if self.connect_dict != dict(should_be_connect_dict):
            raise ValueError("The model you are trying to load has "
                             + "different connections. "
                             + "It's incompatible with the current instance.")
        par_dict = np.load(filename + '.npz')
        self.i2h_pointers     = gpuarray.to_gpu(par_dict['i2h_pointers'])
        self.i2h_indices      = gpuarray.to_gpu(par_dict['i2h_indices'])
        self.i2h_weights      = gpuarray.to_gpu(par_dict['i2h_weights'])
        self.i2h_bias         = gpuarray.to_gpu(par_dict['i2h_bias'])
        self.h2op_pointers    = gpuarray.to_gpu(par_dict['h2op_pointers'])
        self.h2op_indices     = gpuarray.to_gpu(par_dict['h2op_indices'])
        self.h2op_weights     = gpuarray.to_gpu(par_dict['h2op_weights'])
        self.h2op_bias        = gpuarray.to_gpu(par_dict['h2op_bias'])

    def save_mse(self, filename):
        with open(filename + '_mse_list.pkl', 'wb') as fid:
            cP.dump(self.mse, fid)


class OnTheFlyPVM(PVM_CUDA):
    def __init__(self, connect_dict, flat_map, norm=1., act_type='sigmoid',
                 maxthreads=1024, max_grid_size=65535, zero_bias=False):
        super(OnTheFlyPVM, self).__init__(connect_dict,
                                          act_type,
                                          maxthreads,
                                          max_grid_size,
                                          zero_bias
                                          )
        self.input_frame_shuffle = gpuarray.to_gpu(flat_map.astype(np.uint32))
        self.norm = norm

    def forward(self, single_frame):
        assert single_frame.size == self.L_input, \
            "Input frame must match specified input size"

        input_pre_map = gpuarray.to_gpu(single_frame) / self.norm
        input_ = gpuarray.zeros_like(input_pre_map)
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            input_.gpudata,
                                            input_pre_map.gpudata,
                                            self.input_frame_shuffle.gpudata,
                                            self.L_input)
        base_pvm_gpu_forward(self, input_)

    def capture(self, capture, dim, wait_time=17, scale=1):
        height, width, n_colors = dim
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(*dim)

        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        def resize(img):
            return cv2.resize(img, (width, height))

        cv2.namedWindow('Input frame',
                        cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
        cv2.namedWindow('Prediction',
                        cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
        cv2.namedWindow('Error in Prediction',
                        cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
        try:
            while True:
                ret, frame = capture.read()

                input_frame = resize(frame)

                self.forward(input_frame)

                cv2.imshow(
                    "Input Frame",
                    rescale(
                        np.array(
                            frame,
                            dtype=np.uint8
                        )
                    )
                )
                cv2.imshow(
                    "Prediction",
                    rescale(
                        np.array(
                            self.norm * self.pred[:self.L_input].get(),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                cv2.imshow(
                    "Error previous frame",
                    rescale(
                        np.array(
                            self.norm * abs(
                                self.err[:self.L_input].get() - 0.5
                            ),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                cv2.waitKey(wait_time)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    def backward(self, single_frame, next_frame, learning_rate,
                 L2_norm_reg=0):
        self.forward(single_frame)

        next_input_pre_map = gpuarray.to_gpu(next_frame) / self.norm
        next_input = gpuarray.zeros_like(next_input_pre_map)
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            next_input.gpudata,
                                            next_input_pre_map.gpudata,
                                            self.input_frame_shuffle.gpudata,
                                            self.L_input)
        base_pvm_gpu_backprop(self, next_input)
        self._update(self, learning_rate, reg_factor=L2_norm_reg)

    def adam_train(self, dict_training_data, N_epoch, L2_norm_reg=0,
                   print_every_epoch=True, save_every_print=False,
                   filename='default', interval=100,
                   alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-8):
        # nobody cares
        self.m_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.v_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.mhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.vhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.dev_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)

        self.m_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.v_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.mhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.vhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.dev_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)

        self.m_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.v_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.mhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.vhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.dev_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)

        self.m_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.v_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.mhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.vhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.dev_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        print("-" * 80)
        n = 0
        for epoch in range(N_epoch):
            for key, data in dict_training_data.items():
                n_frames, *junk = data.shape
                self.reset_state()

                for i in range(n_frames - 1):
                    if epoch == 0:
                        n += 1
                    single_frame = data[i, ...]
                    next_frame = data[i + 1, ...]

                    self.forward(single_frame)

                    next_input_pre_map = gpuarray.to_gpu(next_frame) \
                                         / self.norm
                    next_input = gpuarray.zeros_like(next_input_pre_map)
                    output_pred_shuffling.prepared_call(
                        self.grid_input,
                        self.threads,
                        next_input.gpudata,
                        next_input_pre_map.gpudata,
                        self.input_frame_shuffle.gpudata,
                        self.L_input)
                    base_pvm_gpu_backprop(self, next_input)
                    adam_update_parameters(self, n,
                                           alpha=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           eps=eps,
                                           reg_factor=L2_norm_reg)

                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred) ** 2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
            print(" " * 19 + 'AVG MSE over last {} frames'.format(n))
            if print_every_epoch:
                mse_avg = sum(self.mse[-n:]) / n
                print("{:>10} Epoch: {:>10}".format(epoch, mse_avg))
                if save_every_print:
                    self.save_parameters(filename)
                    plt.plot(movingaverage(self.mse, interval), 'b')
                    plt.plot(movingaverage(self.img_mse, interval), 'r')
                    plt.legend(['Training error',
                                'Training error (image only)'])
                    plt.savefig(filename +
                                'training_moving_avg' +
                                '_MSE_vs_frames.pdf',
                                transparent=True)
                    plt.close()

    def train_capture(self, capture, dim, learning_rate,
                      wait_time=17, scale=1):
        ret, frame = capture.read()
        height, width, n_colors = dim
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(*dim)
        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        def resize(img):
            return cv2.resize(img, (width, height))
        input_frame = resize(frame)
        try:
            while True:
                ret, new_frame = capture.read()

                input_new_frame = resize(new_frame)

                # self.forward(input_frame)
                self.backward(input_frame, input_new_frame, learning_rate)
                input_frame = input_new_frame.copy()

                cv2.imshow(
                    "Input Frame",
                    rescale(
                        np.array(
                            input_new_frame,
                            dtype=np.uint8
                        )
                    )
                )
                cv2.imshow(
                    "Prediction",
                    rescale(
                        np.array(
                            self.norm * self.pred[:self.L_input].get(),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                cv2.imshow(
                    "Error previous frame",
                    rescale(
                        np.array(
                            self.norm * abs(
                                self.err[:self.L_input].get() - 0.5
                            ),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                cv2.waitKey(wait_time)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    def quick_animate(self, dict_testing_data, wait_time=17, scale=1):
        print("-" * 80 + "\n" + "Animating testing data" + "\n" + "-" * 80)
        print("Use a Keyboard interruption to exit early.")
        print('-' * 80)
        datum = dict_testing_data[list(dict_testing_data.keys())[0]]
        _, height, width, n_colors = datum.shape
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(height, width, n_colors)
        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        try:
            for key, data in dict_testing_data.items():
                n_frames, height, width, n_colors = data.shape
                self.reset_state()

                for i in range(n_frames):
                    single_frame = data[i, ...]
                    self.forward(single_frame)
                    cv2.imshow(
                        "Input Frame",
                        rescale(
                            np.array(
                                single_frame,
                            )[..., ::-1] / self.norm
                        )
                    )
                    cv2.imshow(
                        "Prediction",
                        rescale(
                            np.array(
                                self.pred[:self.L_input].get()
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.imshow(
                        "Error previous frame",
                        rescale(
                            np.array(
                                abs(
                                    self.err[:self.L_input].get() - 0.5
                                )
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    def test(self, dict_testing_data):
        tot_frames_in_epoch = 0
        img_test_error = gpuarray.to_gpu(np.zeros(self.L_input, dtype=np.float64))
        test_error = gpuarray.to_gpu(np.zeros(self.L_pred, dtype=np.float64))
        for key, data in dict_testing_data.items():
            n_frame, row, column, n_colors = data.shape

            for i in range(n_frame - 1):
                self.forward(data[i, ...])
                next_frame = data[i + 1, ...]
                next_input_pre_map = gpuarray.to_gpu(next_frame) / self.norm
                next_input = gpuarray.zeros_like(next_input_pre_map)
                output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                                    next_input.gpudata,
                                                    next_input_pre_map.gpudata,
                                                    self.input_frame_shuffle.gpudata,
                                                    self.L_input)
                # ignore appending the hidden values fed to upper layers to the input
                try:
                    append_hid.prepared_call(self.grid_int_der_err, self.threads,
                                             next_input.gpudata, self.hidden.gpudata,
                                             self.ideal_pred.gpudata,
                                             self.hid_append_map.gpudata,
                                             self.L_input,
                                             self.L_pred)
                except:
                    self.ideal_pred = next_input[:]

                # calculating the gradients in error using MSE
                sub.prepared_async_call(self.grid_int_der_err, self.threads,
                                        self.stream1,
                                        self.pred.gpudata, self.ideal_pred.gpudata,
                                        self.delta_pred.gpudata, self.L_pred)
                mask = self.pred_mask.get()
                test_error += (self.delta_pred) ** 2 / sum(mask)
                img_test_error += (self.delta_pred[:self.L_input]) ** 2 \
                                  / sum(mask[:self.L_input])
                tot_frames_in_epoch += 1
        avg_err = sum(test_error) / tot_frames_in_epoch
        img_avg_err = sum(img_test_error) / tot_frames_in_epoch
        return (tot_frames_in_epoch, np.asscalar(avg_err.get()),
                np.asscalar(img_avg_err.get()))

    def adam_train_and_validate(self, dict_training_data, dict_valid_data,
                                N_epoch, L2_norm_reg=0,
                                print_every_epoch=True,
                                save_every_print=False, filename='default',
                                interval=100, alpha=0.0001, beta1=0.9,
                                beta2=0.999, eps=1e-8):
        # nobody cares these aren't used anywhere besides training with
        # ADAM
        self.m_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.v_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.mhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.vhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.dev_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)

        self.m_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.v_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.mhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.vhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.dev_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)

        self.m_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.v_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.mhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.vhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.dev_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)

        self.m_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.v_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.mhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.vhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.dev_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        print("-" * 80)
        n = 0
        valid_err_list = []
        img_valid_err_list = []
        for epoch in range(N_epoch):
            for key, data in dict_training_data.items():
                n_frames, *junk = data.shape
                self.reset_state()

                for i in range(n_frames - 1):
                    if epoch == 0:
                        n += 1
                    single_frame = data[i, ...]
                    next_frame = data[i + 1, ...]

                    self.forward(single_frame)

                    next_input_pre_map = gpuarray.to_gpu(next_frame) \
                                         / self.norm
                    next_input = gpuarray.zeros_like(next_input_pre_map)
                    output_pred_shuffling.prepared_call(
                        self.grid_input,
                        self.threads,
                        next_input.gpudata,
                        next_input_pre_map.gpudata,
                        self.input_frame_shuffle.gpudata,
                        self.L_input)
                    base_pvm_gpu_backprop(self, next_input)
                    adam_update_parameters(self, n,
                                           alpha=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           eps=eps,
                                           reg_factor=L2_norm_reg)

                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred) ** 2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
            # print(" " * 19 + 'AVG MSE over last {} frames'.format(n))
            if print_every_epoch:
                mse_avg = sum(self.mse[-n:]) / n
                print("{:>10} Epoch Training Error: {:>10}".format(epoch,
                                                                   mse_avg))
                valid_tot_frames, val_err, img_val_err = \
                    self.test(dict_valid_data)
                valid_err_list.append(val_err)
                img_valid_err_list.append(img_val_err)
                print("{:>10} Epoch Validation Error: {:>10}".format(epoch,
                                                                     val_err))
                if save_every_print:
                    self.save_parameters(filename)
                    plt.plot(movingaverage(self.mse, interval), 'b')
                    plt.plot(movingaverage(self.img_mse, interval), 'r')
                    plt.plot(n * (np.arange(1, epoch + 2) - 0.5), valid_err_list,
                             '.--b', markersize=10)
                    plt.plot(n * (np.arange(1, epoch + 2) - 0.5), img_valid_err_list,
                             '.--r', markersize=10)
                    plt.legend(['Training error',
                                'Training error (image only)',
                                'Validation error',
                                'Validation error (image only)'])
                    plt.savefig(filename +
                                'training_moving_avg' +
                                '_MSE_vs_frames.pdf',
                                transparent=True)
                    plt.close()


class PVMtracker(PVM_CUDA):
    def __init__(self, structure, output_sizes, heat_map_size, input_size,
                 hidden_size, act_type='sigmoid', maxthreads=1024,
                 max_grid_size=65535, context_from_top_0_0=True):

        self.structure      = structure
        self.output_sizes   = output_sizes
        self.heat_map_size  = heat_map_size
        self.input_size     = input_size
        self.hidden_size    = hidden_size

        self.connect_dict = make_connections(structure, input_size,
                                             hidden_size,
                                             output_sizes,
                                             context_from_top_0_0=
                                             context_from_top_0_0)

        super(PVMtracker, self).__init__(self.connect_dict,
                                         act_type,
                                         maxthreads,
                                         max_grid_size)
        tracker_stuff = tracker_weight_initialize(structure,
                                                  output_sizes,
                                                  heat_map_size)

        tracker_new_layer = tracker_stuff[4]
        tracker_new_heatmap = list(range(
            0, heat_map_size * len(tracker_new_layer), heat_map_size))
        block_idx, max_L_tracker_input, heatmapsize = new_block_idx(
            tracker_new_layer, tracker_new_heatmap
        )

        self.tracker_pointers = gpuarray.to_gpu(tracker_stuff[0])
        self.tracker_indices  = gpuarray.to_gpu(tracker_stuff[1])
        self.tracker_weights  = gpuarray.to_gpu(tracker_stuff[2])
        self.tracker_bias     = gpuarray.to_gpu(tracker_stuff[3])

        self.tracker_row_idx = gpuarray.to_gpu(ptr_to_row(tracker_stuff[0]))
        self.tracker_weights_nnz = np.int32(tracker_stuff[0][-1])

        self.N_layers = np.int32(len(structure))
        self.L_heatmaps = np.int32(self.N_layers * heat_map_size)
        self.L_avg_heatmap = np.int32(heat_map_size)

        grid_x_tracker = min(int(math.ceil(self.L_heatmaps / maxthreads)),
                             max_grid_size)
        grid_x_tracker_dot = min(int(math.ceil(
            max_L_tracker_input / maxthreads
        )), max_grid_size)
        grid_y_tracker_dot = min(int(self.L_heatmaps), max_grid_size)
        grid_x_avgpool = min(int(math.ceil(self.L_avg_heatmap / maxthreads)),
                             max_grid_size)
        grid_x_update_tracker = min(int(math.ceil(self.tracker_weights_nnz
                                                  / maxthreads)),
                                    max_grid_size)
        self.grid_tracker = (grid_x_tracker, 1)
        self.grid_tracker_dot = (grid_x_tracker_dot, grid_y_tracker_dot)
        self.grid_avgpool = (grid_x_avgpool, 1)
        self.grid_update_tracker = (grid_x_update_tracker, 1)

        self.heatmaps = gpuarray.to_gpu(np.zeros(self.L_heatmaps))
        self.heatmaps_affine = gpuarray.empty_like(self.heatmaps)
        self.avg_heatmap = gpuarray.to_gpu(np.zeros(self.L_avg_heatmap))

        self.delta_tracker = gpuarray.to_gpu(np.zeros(self.L_heatmaps))
        self.a_prime_tracker = gpuarray.empty_like(self.heatmaps)
        self.grad_weight_tracker = gpuarray.empty_like(self.tracker_weights)
        self.grad_bias_tracker = gpuarray.empty_like(self.tracker_bias)

    def reset_state(self):
        super(PVMtracker, self).reset_state()

        zerofill.prepared_async_call(self.grid_tracker, self.threads,
                                     self.stream1,
                                     self.heatmaps.gpudata, self.L_heatmaps)
        zerofill.prepared_async_call(self.grid_tracker, self.threads,
                                     self.stream2,
                                     self.heatmaps_affine.gpudata,
                                     self.L_heatmaps)
        zerofill.prepared_async_call(self.grid_avgpool, self.threads,
                                     self.stream3,
                                     self.avg_heatmap.gpudata,
                                     self.L_avg_heatmap)

    def forward(self, single_frame):
        super(PVMtracker, self).forward(single_frame)
        zerofill.prepared_call(self.grid_tracker, self.threads,
                               self.heatmaps_affine.gpudata,
                               self.L_heatmaps)

        dot.prepared_call(self.grid_tracker_dot, self.threads,
                          # number of rows
                          self.L_heatmaps,
                          # CSR sparse matrix
                          self.tracker_pointers.gpudata,
                          self.tracker_indices.gpudata,
                          self.tracker_weights.gpudata,
                          # vector
                          self.output.gpudata,
                          # results
                          self.heatmaps_affine.gpudata)

        add.prepared_call(self.grid_tracker, self.threads,
                          self.heatmaps_affine.gpudata,
                          self.tracker_bias.gpudata,
                          self.heatmaps_affine.gpudata,
                          self.L_heatmaps)

        self.act.prepared_call(self.grid_tracker, self.threads,
                               self.heatmaps_affine.gpudata, self.heatmaps.gpudata,
                               self.L_heatmaps)

        avg_pool.prepared_call(self.grid_avgpool, self.threads,
                               self.heatmaps.gpudata,
                               self.avg_heatmap.gpudata,
                               self.L_avg_heatmap, self.N_layers)

    def backward(self, single_frame, next_frame, ground_truth_heat_map,
                   learning_rate):
        assert len(single_frame) == len(next_frame),\
            "Both frames must be of the same size"
        assert len(ground_truth_heat_map) == self.L_avg_heatmap,\
            "Ground truth heat map must be the size specified during"\
            + " initialization"
        next_input = gpuarray.to_gpu(next_frame)
        gt_heatmap = gpuarray.to_gpu(ground_truth_heat_map)

        self.forward(single_frame)

        self.dact.prepared_async_call(self.grid_i2h, self.threads,
                                      self.stream1,
                                      self.hid_affine.gpudata,
                                      self.a_prime_i2h.gpudata,
                                      self.L_hidden)

        self.dact.prepared_async_call(self.grid_h2op, self.threads,
                                      self.stream2,
                                      self.out_and_pred_affine.gpudata,
                                      self.a_prime_h2op.gpudata, self.L_op)

        self.dact.prepared_async_call(self.grid_tracker, self.threads,
                                      self.stream3,
                                      self.heatmaps_affine.gpudata,
                                      self.a_prime_tracker.gpudata,
                                      self.L_heatmaps)

        # ignore appending the hidden values fed to upper layers to the input
        try:
            append_hid.prepared_call(self.grid_int_der_err, self.threads,
                                     next_input.gpudata, self.hidden.gpudata,
                                     self.ideal_pred.gpudata,
                                     self.hid_append_map.gpudata,
                                     self.L_input,
                                     self.L_pred)
        except:
            self.ideal_pred = next_input[:]

        # calculating the gradients in error using MSE
        sub.prepared_async_call(self.grid_int_der_err, self.threads,
                                self.stream1,
                                self.pred.gpudata, self.ideal_pred.gpudata,
                                self.delta_pred.gpudata, self.L_pred)

        hadamard.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream1,
                                     self.delta_pred.gpudata,
                                     self.pred_mask.gpudata,
                                     self.delta_pred.gpudata, self.L_pred)

        sq_err_der_tracker.prepared_async_call(self.grid_avgpool,
                                               self.threads,
                                               self.stream2,
                                               self.avg_heatmap.gpudata,
                                               gt_heatmap.gpudata,
                                               self.delta_tracker.gpudata,
                                               self.L_avg_heatmap,
                                               self.N_layers)

        # multiplying elementwise by the derivation of the activation
        hadamard.prepared_async_call(self.grid_tracker, self.threads,
                                     self.stream2,
                                     self.delta_tracker.gpudata,
                                     self.a_prime_tracker.gpudata,
                                     self.delta_tracker.gpudata,
                                     self.L_heatmaps)

        zerofill.prepared_async_call(self.grid_o_shuf, self.threads,
                                     self.stream2,
                                     self.delta_output.gpudata, self.L_out)

        # data type issue 446464 is the values pointers goes up to
        # but 65535 is unsigned int range
        # need to change unsigned int to unsigned long int
        Tdot.prepared_async_call(self.grid_tracker, self.threads,
                                 self.stream2,
                                 self.L_heatmaps,
                                 self.tracker_pointers.gpudata,
                                 self.tracker_indices.gpudata,
                                 self.tracker_weights.gpudata,
                                 # vector
                                 self.delta_tracker.gpudata,
                                 # results
                                 self.delta_output.gpudata)

        rev_output_pred_shuffling.prepared_async_call(self.grid_int_der_err,
                                                      self.threads,
                                                      self.stream1,
                                                      self.delta_pred.gpudata,
                                                      self.delta_h2op.gpudata,
                                                      self.pred_map.gpudata,
                                                      self.L_pred)

        rev_output_pred_shuffling.prepared_async_call(self.grid_o_shuf,
                                                      self.threads,
                                                      self.stream2,
                                                      self.delta_output.gpudata,
                                                      self.delta_h2op.gpudata,
                                                      self.out_map.gpudata,
                                                      self.L_out)

        hadamard.prepared_call(self.grid_h2op, self.threads,
                               self.delta_h2op.gpudata,
                               self.a_prime_h2op.gpudata,
                               self.delta_h2op.gpudata, self.L_op)

        zerofill.prepared_call(self.grid_i2h, self.threads,
                               self.delta_i2h.gpudata, self.L_hidden)

        Tdot.prepared_call(self.grid_h2op, self.threads,
                           self.L_op,
                           self.h2op_pointers.gpudata,
                           self.h2op_indices.gpudata,
                           self.h2op_weights.gpudata,
                           self.delta_h2op.gpudata,
                           self.delta_i2h.gpudata)

        hadamard.prepared_call(self.grid_i2h, self.threads,
                               self.delta_i2h.gpudata,
                               self.a_prime_i2h.gpudata,
                               self.delta_i2h.gpudata, self.L_hidden)

        kron.prepared_async_call(self.grid_i2h, self.threads,
                                 self.stream1,
                                 self.i2h_weights_nnz,
                                 self.i2h_row_idx.gpudata,
                                 self.i2h_indices.gpudata,
                                 self.full_input.gpudata, self.delta_i2h.gpudata,
                                 self.grad_weight_i2h.gpudata)

        kron.prepared_async_call(self.grid_h2op, self.threads,
                                 self.stream2,
                                 self.h2op_weights_nnz,
                                 self.h2op_row_idx.gpudata,
                                 self.h2op_indices.gpudata,
                                 self.hidden.gpudata,
                                 self.delta_h2op.gpudata,
                                 self.grad_weight_h2op.gpudata)

        kron.prepared_async_call(self.grid_tracker, self.threads,
                                 self.stream3,
                                 self.tracker_weights_nnz,
                                 self.tracker_row_idx.gpudata,
                                 self.tracker_indices.gpudata,
                                 self.output.gpudata,
                                 self.delta_tracker.gpudata,
                                 self.grad_weight_tracker.gpudata)

        update.prepared_async_call(self.grid_i2h, self.threads,
                                   self.stream1,
                                   self.i2h_bias.gpudata,
                                   self.delta_i2h.gpudata,
                                   learning_rate, self.L_hidden)

        update.prepared_async_call(self.grid_h2op, self.threads,
                                   self.stream2,
                                   self.h2op_bias.gpudata,
                                   self.delta_h2op.gpudata,
                                   learning_rate, self.L_op)

        update.prepared_async_call(self.grid_tracker, self.threads,
                                   self.stream3,
                                   self.tracker_bias.gpudata,
                                   self.delta_tracker.gpudata,
                                   learning_rate, self.L_heatmaps)

        update.prepared_async_call(self.grid_update_i2h, self.threads,
                                   self.stream1,
                                   self.i2h_weights.gpudata,
                                   self.grad_weight_i2h.gpudata,
                                   learning_rate, self.i2h_weights_nnz)

        update.prepared_async_call(self.grid_update_h2op, self.threads,
                                   self.stream2,
                                   self.h2op_weights.gpudata,
                                   self.grad_weight_h2op.gpudata,
                                   learning_rate, self.h2op_weights_nnz)

        update.prepared_async_call(self.grid_update_tracker, self.threads,
                                   self.stream3,
                                   self.tracker_weights.gpudata,
                                   self.grad_weight_tracker.gpudata,
                                   learning_rate, self.tracker_weights_nnz)

    def train(self, dict_training_data, learning_rate_list,
              print_every=100000, save_every_print=False,
              filename='tracker', interval=100):
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        N_frames = len(learning_rate_list)
        while n < N_frames:
            for key, data in dict_training_data.items():
                rescale_arr, ground_truth_heat_map_list = data
                self.reset_state()

                for i, ground_truth_heat_map in \
                        enumerate(ground_truth_heat_map_list[:-1]):
                    learning_rate = learning_rate_list[n]

                    n += 1
                    next_frame   = rescale_arr[i+1, :]
                    single_frame = rescale_arr[i, :]
                    self.backward(single_frame, next_frame,
                                    ground_truth_heat_map,
                                    learning_rate)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred)**2).get()
                                  )
                    heat_map_mse_sum = sum((self.avg_heatmap.get() -
                                           ground_truth_heat_map) ** 2)
                    mse_sum += heat_map_mse_sum
                    mse_sum /= (sum(self.pred_mask.get()) + self.L_avg_heatmap)
                    # pred_mse_sum /= self.L_pred
                    # heat_map_mse_sum /= self.L_avg_heatmap
                    self.mse.append(mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) /\
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse, interval))
                            plt.xlabel('Frames')
                            plt.ylabel('MSE')
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()
                    if n == N_frames:
                        break
                if n == N_frames:
                    break

    def get_bounding_box(self, heatmap_shape, threshold=32/255):

        bounding_box = np.zeros(heatmap_shape, dtype=np.bool)

        heatmap = self.avg_heatmap.get().reshape(heatmap_shape)
        max_val = np.max(heatmap)
        med_val = np.median(heatmap)
        # peak = np.argmax(heatmap)
        if max_val > med_val + threshold:
            cutoff = (max_val - med_val) * 0.5 + med_val
            exceed_cutoff = heatmap > cutoff
            ar_y, ar_x = np.where(exceed_cutoff)
            x_min, x_max = min(ar_x), max(ar_x)
            y_min, y_max = min(ar_y), max(ar_y)
            bounding_box[y_min:y_max + 1, x_min:x_max + 1] = True
        return bounding_box

    def test(self, dict_testing_data, heatmap_shape, success_threshold,
             precision_threshold, accuracy_threshold=1.0,
             bbox_threshold=32/255):
        heatmap_x, heatmap_y = heatmap_shape
        if heatmap_x * heatmap_y != self.heat_map_size:
            raise ValueError('Heat map shape does not match size '
                             + 'specified during initialization.')

        avoid_div_by0 = lambda x: x if x != 0 else 1

        success_list   = []
        precision_list = []
        accuracy_list  = []
        for key, data in dict_testing_data.items():
            rescale_arr, ground_truth_heat_map_list = data
            self.reset_state()

            for i, ground_truth_heat_map in \
                    enumerate(ground_truth_heat_map_list):
                frame = rescale_arr[i, :]
                self.forward(frame)
                bbox = self.get_bounding_box(heatmap_shape,
                                             threshold=bbox_threshold)
                reshaped_gt = ground_truth_heat_map.reshape(heatmap_shape)
                success_frac = np.sum(bbox * reshaped_gt) /\
                    avoid_div_by0(np.sum(reshaped_gt))
                # using centroid as definition of center
                idx_arr = np.mgrid[0:heatmap_x, 0:heatmap_y]
                X_arr = idx_arr[1]
                Y_arr = idx_arr[0]
                X = np.sum(X_arr * bbox) / avoid_div_by0(np.sum(bbox))
                Y = np.sum(Y_arr * bbox) / avoid_div_by0(np.sum(bbox))
                X_gt_arr = X_arr * reshaped_gt
                Y_gt_arr = Y_arr * reshaped_gt
                X_gt = np.sum(X_gt_arr) / avoid_div_by0(np.sum(reshaped_gt))
                Y_gt = np.sum(Y_gt_arr) / avoid_div_by0(np.sum(reshaped_gt))
                x_w_gt = np.max(X_gt_arr) - np.min(X_gt_arr)
                y_w_gt = np.max(Y_gt_arr) - np.min(Y_gt_arr)
                dX_gt = accuracy_threshold * x_w_gt / 2
                dY_gt = accuracy_threshold * y_w_gt / 2
                # checking if tracker center in the appropriate range
                inX = (X <= X_gt + dX_gt) and (X >= X_gt - dX_gt)
                inY = (Y <= Y_gt + dY_gt) and (Y >= Y_gt - dY_gt)
                true_neg = not (np.any(reshaped_gt) or
                                np.any(bbox))
                # distance in pixels from centroid of the ground truth
                # bounding box and the heat map
                dist = np.sqrt((X - X_gt)**2 + (Y - Y_gt)**2)

                if success_frac >= success_threshold:
                    success_list.append(True)
                else:
                    success_list.append(False)
                if dist < precision_threshold:
                    precision_list.append(True)
                else:
                    precision_list.append(False)
                if (inX and inY) or true_neg:
                    accuracy_list.append(True)
                else:
                    accuracy_list.append(False)
        return success_list, precision_list, accuracy_list

    def save_parameters(self, filename):
        with open(filename + '_connections.pkl', 'wb') as fid:
            cP.dump(self.connect_dict, fid)
        np.savez(filename, i2h_pointers=self.i2h_pointers.get(),
                 i2h_indices=self.i2h_indices.get(),
                 i2h_weights=self.i2h_weights.get(),
                 i2h_bias=self.i2h_bias.get(),
                 h2op_pointers=self.h2op_pointers.get(),
                 h2op_indices=self.h2op_indices.get(),
                 h2op_weights=self.h2op_weights.get(),
                 h2op_bias=self.h2op_bias.get(),
                 tracker_pointers=self.tracker_pointers.get(),
                 tracker_indices=self.tracker_indices.get(),
                 tracker_weights=self.tracker_weights.get(),
                 tracker_bias=self.tracker_bias.get())

    def load_parameters(self, filename):
        with open(filename + '_connections.pkl', 'rb') as fid:
            should_be_connect_dict = cP.load(fid)
        if self.connect_dict != dict(should_be_connect_dict):
            raise ValueError("The model you are trying to load has "
                             + "different connections. "
                             + "It's incompatible with the current instance.")
        par_dict = np.load(filename + '.npz')
        self.i2h_pointers     = gpuarray.to_gpu(par_dict['i2h_pointers'])
        self.i2h_indices      = gpuarray.to_gpu(par_dict['i2h_indices'])
        self.i2h_weights      = gpuarray.to_gpu(par_dict['i2h_weights'])
        self.i2h_bias         = gpuarray.to_gpu(par_dict['i2h_bias'])
        self.h2op_pointers    = gpuarray.to_gpu(par_dict['h2op_pointers'])
        self.h2op_indices     = gpuarray.to_gpu(par_dict['h2op_indices'])
        self.h2op_weights     = gpuarray.to_gpu(par_dict['h2op_weights'])
        self.h2op_bias        = gpuarray.to_gpu(par_dict['h2op_bias'])
        self.tracker_pointers = gpuarray.to_gpu(par_dict['tracker_pointers'])
        self.tracker_indices  = gpuarray.to_gpu(par_dict['tracker_indices'])
        self.tracker_weights  = gpuarray.to_gpu(par_dict['tracker_weights'])
        self.tracker_bias     = gpuarray.to_gpu(par_dict['tracker_bias'])


class PhantomXTurretPVM(OnTheFlyPVM):
    def __init__(self, connect_dict, flat_map, dim, x_w, y_w, fov,
                 fov_horizontal=True, norm=1, noise=1, act_type='sigmoid',
                 threshold=0.01, maxthreads=1024, max_grid_size=65535,
                 omega=0.8, gamma=0.9, threadsize=(32, 32, 1), zero_bias=False):
        super(PhantomXTurretPVM, self).__init__(connect_dict, flat_map, norm,
                                                act_type, maxthreads,
                                                max_grid_size, zero_bias)
        height, width, n_colors = dim
        self.height = np.int32(height)
        self.width = np.int32(width)
        self.n_color = np.int32(n_colors)
        # field of view should be divided by half in the equations below
        # but that's why it's 360 instead of 180 in the conversion to
        # radian from degrees
        if fov_horizontal:
            self.pixel_focal_length = width / (2 * np.tan(np.pi *
                                                          fov / 360))
        else:
            self.pixel_focal_length = height / (2 * np.tan(np.pi *
                                                           fov / 360))
        self.x_w = np.int32(x_w)
        self.y_w = np.int32(y_w)

        L_patchwise_err = (height - y_w + 1) * (width - x_w + 1)
        self.L_patchwise_err = np.int32(L_patchwise_err)

        self.patchwise_err = gpuarray.to_gpu(np.zeros(L_patchwise_err))
        self.err_sq = gpuarray.to_gpu(np.zeros(self.L_input))
        self.err_sub_frame = gpuarray.to_gpu(np.zeros(self.L_input))
        self.err_sub_frame_pre_map = gpuarray.zeros_like(self.err_sub_frame)

        ceil = lambda x: int(np.ceil(x))
        self.threadsize = threadsize
        self.gridsize = (min(ceil(height / threadsize[0]),
                             max_grid_size),
                         min(ceil(width / threadsize[1]),
                             max_grid_size))
        self.grid_max1 = (min(ceil(L_patchwise_err / maxthreads),
                              max_grid_size), 1)
        self.grid_max2 = (min(ceil(self.grid_max1[0] / maxthreads),
                              max_grid_size), 1)

        self.L_max2 = np.int32(min(self.grid_max1[0],
                                   max_grid_size))
        self.argmax1 = gpuarray.to_gpu(np.zeros(self.L_max2,
                                                dtype=np.uint32))
        self.max1 = gpuarray.to_gpu(np.zeros(self.L_max2,
                                             dtype=np.float64))
        self.argmax2 = gpuarray.to_gpu(np.zeros(1, dtype=np.uint32))
        self.max2 = gpuarray.to_gpu(np.zeros(1, dtype=np.float64))

        rev_map = np.argsort(flat_map).reshape(*dim)
        self.input_frame_rev_shuf = gpuarray.to_gpu(rev_map.astype(np.uint32))
        self.x_shift = 0 # horizontal shift in pixels from the left of the image
        self.y_shift = 0 # vertical shift in pixels from the top of the image
        self.prev_x_shift = 0
        self.prev_y_shift = 0
        self.real_x_shift = 0
        self.real_y_shift = 0
        self.omega = omega  # omega * delta_t
        self.omega_sq = omega**2
        self.gamma = gamma
        self.gamma_omega = gamma * omega
        self.prev_max = 0.

        self.pan = 512 # center value of pan servo
        self.tilt = 512 # center value of tilt servo

        self.noise = noise
        self.threshold = threshold * x_w * y_w * n_colors

    def reset_state(self, x_pos=0, y_pos=0):
        super(PhantomXTurretPVM, self).reset_state()
        self.prev_max = 0.
        self.x_shift = 0.
        self.y_shift = 0.
        self.prev_x_shift = 0
        self.prev_y_shift = 0
        self.real_x_shift = 0
        self.real_y_shift = 0

    def forward(self, single_frame):
        super(PhantomXTurretPVM, self).forward(single_frame)

        self.err_sub_frame_pre_map = self.err[:self.L_input].copy()
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            self.err_sub_frame.gpudata,
                                            self.err_sub_frame_pre_map.gpudata,
                                            self.input_frame_rev_shuf.gpudata,
                                            self.L_input)

        hadamard.prepared_call(self.grid_input, self.threads,
                               self.err_sub_frame.gpudata,
                               self.err_sub_frame.gpudata,
                               self.err_sq.gpudata, self.L_input)

        err_patch_sum.prepared_call(self.gridsize, self.threadsize,
                                    self.err_sq.gpudata,
                                    self.patchwise_err.gpudata,
                                    self.width,
                                    self.height,
                                    self.n_color,
                                    self.x_w,
                                    self.y_w)

        self.max1 = self.patchwise_err[:self.L_patchwise_err]

        max_gpu.prepared_call(self.grid_max1, self.threads,
                              self.patchwise_err.gpudata,
                              self.max1.gpudata,
                              self.argmax1.gpudata,
                              self.L_patchwise_err)

        max_gpu.prepared_call(self.grid_max2, self.threads,
                              self.max1.gpudata,
                              self.max2.gpudata,
                              self.argmax2.gpudata,
                              self.L_max2)

        arg_max = self.argmax1.get()[self.argmax2.get()[0]]
        max_val = self.max2.get()[0]

        if max_val > max(self.prev_max, self.threshold):
            array_width = (self.width - self.x_w)
            x_shift = arg_max % (array_width + 1) \
                      - array_width / 2.
            y_shift = arg_max // (array_width + 1) \
                      - (self.height - self.y_w) / 2.
        else:
            x_shift, y_shift = 0., 0.

        self.prev_max = (max_val + self.prev_max) / 2.

        self.x_shift = x_shift
        self.y_shift = y_shift

    def evolve(self):
        self.real_x_shift, self.real_y_shift,\
        self.prev_x_shift, self.prev_y_shift = (
            (self.prev_x_shift * (self.gamma_omega - 1)
             + self.x_shift * self.omega_sq) / (1 + self.gamma_omega),
            (self.prev_y_shift * (self.gamma_omega - 1)
             + self.y_shift * self.omega_sq) / (1 + self.gamma_omega),
            self.real_x_shift, self.real_y_shift
        )

        self.pan -= int(3.41 * 180 * np.arctan(self.real_x_shift
                                               / self.pixel_focal_length)
                        / np.pi
                        )
        self.tilt -= int(3.41 * 180 * np.arctan(self.real_y_shift
                                                / self.pixel_focal_length)
                         / np.pi
                         )

        self.pan = max(self.pan, 0)
        self.pan = min(self.pan, 1023)

        self.tilt = max(self.tilt, 256)
        self.tilt = min(self.tilt, 768)

    def capture(self, capture, wait_time=17, scale=1):
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(self.height,
                  self.width,
                  self.n_color)

        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        def resize(img):
            return cv2.resize(img, (self.width, self.height))

        cv2.namedWindow('Input frame',
                        cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
        cv2.namedWindow('Prediction',
                        cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
        cv2.namedWindow('Error in Prediction',
                        cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
        try:
            while True:
                ret, frame = capture.read()

                input_frame = resize(frame)

                self.forward(input_frame)

                cv2.imshow(
                    "Input Frame",
                    rescale(
                        np.array(
                            frame,
                            dtype=np.uint8
                        )
                    )
                )
                cv2.imshow(
                    "Prediction",
                    rescale(
                        np.array(
                            self.norm * self.pred[:self.L_input].get(),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                err = np.array(
                    self.norm * abs(
                        self.err[:self.L_input].get() - 0.5
                    ),
                    dtype=np.uint8
                )[rev_map]
                y_shift_center, x_shift_center = int(self.y_shift +
                                                     self.height / 2), \
                                                 int(self.x_shift +
                                                     self.width / 2)
                err[y_shift_center, x_shift_center, 0] = 0
                err[y_shift_center, x_shift_center, 1] = 0
                err[y_shift_center, x_shift_center, 2] = 255
                cv2.imshow(
                    "Error previous frame",
                    rescale(err)
                )
                cv2.waitKey(wait_time)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    def serial_communication_and_capture(self, ser, capture,
                                         wait_time=17, scale=1):
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(self.height, self.width, self.n_color)

        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        def resize(img):
            return cv2.resize(img, (self.width, self.height))

        cv2.namedWindow('Input frame', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.namedWindow('Prediction', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.namedWindow('Error previous frame', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        try:
            while True:
                ret, frame = capture.read()

                input_frame = resize(frame)

                self.forward(input_frame)
                self.evolve()
                ser.write('{},{},'.format(self.pan, self.tilt).encode())

                cv2.imshow(
                    "Input frame",
                    rescale(
                        np.array(
                            frame,
                            dtype=np.uint8
                        )
                    )
                )
                cv2.imshow(
                    "Prediction",
                    rescale(
                        np.array(
                            self.norm * self.pred[:self.L_input].get(),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                err = np.array(
                    self.norm * abs(
                        self.err[:self.L_input].get() - 0.5
                    ),
                    dtype=np.uint8
                )[rev_map]
                y_shift_center, x_shift_center = int(self.y_shift +
                                                     self.height / 2), \
                                                 int(self.x_shift +
                                                     self.width / 2)
                err[y_shift_center, x_shift_center, 0] = 0
                err[y_shift_center, x_shift_center, 1] = 0
                err[y_shift_center, x_shift_center, 2] = 255
                cv2.imshow(
                    "Error previous frame",
                    rescale(err)
                )
                cv2.waitKey(wait_time)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            ser.write('512,512,'.encode())
            self.pan, self.tilt = 512, 512
            ser.readline()


class TransformMotionIntegrationPhantomXTurretPVM(PhantomXTurretPVM):
    def __init__(self, connect_dict, flat_map, dim, x_w, y_w, fov,
                 fov_horizontal=True, norm=1, noise=1, act_type='sigmoid',
                 threshold=0.01, maxthreads=1024, max_grid_size=65535,
                 omega=0.8, gamma=0.9, threadsize=(32, 32, 1), zero_bias=False):
        super(TransformMotionIntegrationPhantomXTurretPVM, self).__init__(
            connect_dict, flat_map, dim, x_w, y_w, fov, fov_horizontal,
            norm, noise, act_type, threshold, maxthreads, max_grid_size,
            omega, gamma, threadsize, zero_bias
        )

    def forward(self, single_frame, pose, prev_pose):
        height, width, n_color = single_frame.shape  # might want to use self.height etc.
        theta1, phi1 = (prev_pose[:2] / 1023. - 512. / 1023.) * 300. * np.pi / 180.
        theta2, phi2 = (pose[:2] / 1023. - 512. / 1023.) * 300. * np.pi / 180.

        cosphi1 = np.cos(phi1)
        sinphi1 = np.sin(phi1)

        costheta = np.cos(theta2 - theta1)
        sintheta = np.sin(theta2 - theta1)

        cosphi2 = np.cos(phi2)
        sinphi2 = np.sin(phi2)

        M = np.array(
            [[costheta,
              sinphi1 * sintheta,
              -cosphi1 * sintheta],
             [-sinphi2 * sintheta,
              sinphi2 * sinphi1 * costheta + cosphi2 * cosphi1,
              -sinphi2 * cosphi1 * costheta + cosphi2 * sinphi1],
             [cosphi2 * sintheta,
              -sinphi1 * cosphi2 * costheta + sinphi2 * cosphi1,
              cosphi1 * cosphi2 * costheta + sinphi2 * sinphi1]
             ]
        )

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        xx = (xx - width / 2) / self.pixel_focal_length
        yy = (yy - height / 2) / self.pixel_focal_length

        XX = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
        YY = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]

        X = (XX * self.pixel_focal_length + width / 2).astype(np.int32)
        Y = (YY * self.pixel_focal_length + height / 2).astype(np.int32)
        X[np.less(X, 0)] = 0
        X[np.greater_equal(X, width)] = width - 1
        Y[np.less(Y, 0)] = 0
        Y[np.greater_equal(Y, height)] = height - 1

        transform_frame = single_frame[Y, X, :]

        super(TransformMotionIntegrationPhantomXTurretPVM,
              self).forward(transform_frame)

    def backward(self, single_frame, pose, prev_pose, next_frame,
                 learning_rate, L2_norm_reg=0):
        self.forward(single_frame, pose, prev_pose)

        next_input_pre_map = gpuarray.to_gpu(
            next_frame.ravel()) / self.norm

        next_input = gpuarray.zeros_like(next_input_pre_map)

        output_pred_shuffling.prepared_call(
            self.grid_input, self.threads,
            next_input.gpudata,
            next_input_pre_map.gpudata,
            self.input_frame_shuffle.gpudata,
            self.L_input
        )

        base_pvm_gpu_backprop(self, next_input)

    def serial_communication_and_capture(self, ser, capture,
                                         wait_time=17, scale=1):
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(self.height, self.width, self.n_color)

        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)

        def resize(img):
            return cv2.resize(img, (self.width, self.height))

        cv2.namedWindow('Input Frame', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.namedWindow('Prediction', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.namedWindow('Error previous frame', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        ser.write('{},{},'.format(self.pan, self.tilt).encode())
        try:
            while True:
                ret, frame = capture.read()

                input_frame = resize(frame)

                pan_tilt = ser.readline()
                pan_tilt = pan_tilt.decode('utf-8').split('\r\n')[0]
                prev_pose = np.array([int(i) for i in pan_tilt.split(', ')])
                pose = np.array([self.pan, self.tilt])
                self.forward(input_frame, pose, prev_pose)
                self.evolve()
                ser.write('{},{},'.format(self.pan, self.tilt).encode())

                cv2.imshow(
                    "Input Frame",
                    rescale(
                        np.array(
                            frame,
                            dtype=np.uint8
                        )
                    )
                )
                cv2.imshow(
                    "Prediction",
                    rescale(
                        np.array(
                            self.norm * self.pred[:self.L_input].get(),
                            dtype=np.uint8
                        )[rev_map]
                    )
                )
                err = np.array(
                    self.norm * abs(
                        self.err[:self.L_input].get() - 0.5
                    ),
                    dtype=np.uint8
                )[rev_map]
                y_shift_center, x_shift_center = int(self.y_shift +
                                                     self.height / 2), \
                                                 int(self.x_shift +
                                                     self.width / 2)
                err[y_shift_center, x_shift_center, 0] = 0
                err[y_shift_center, x_shift_center, 1] = 0
                err[y_shift_center, x_shift_center, 2] = 255
                cv2.imshow(
                    "Error previous frame",
                    rescale(err)
                )
                cv2.waitKey(wait_time)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            ser.write('512,512,'.encode())
            self.pan, self.tilt = 512, 512
            ser.readline()


class PinholeSaccadicPVM(OnTheFlyPVM):
    def __init__(self, connect_dict, flat_map, dim, x_w, y_w,
                 omega, gamma, norm=1, noise=1, act_type='sigmoid',
                 maxthreads=1024, max_grid_size=65535,
                 threadsize=(32, 32, 1), zero_bias=False):
        super(PinholeSaccadicPVM, self).__init__(connect_dict, flat_map, norm,
                                                 act_type, maxthreads,
                                                 max_grid_size, zero_bias)

        self.x_w = np.int32(x_w)
        self.y_w = np.int32(y_w)
        height, width, n_color = dim
        self.sub_frame_height = np.int32(height)
        self.sub_frame_width  = np.int32(width)
        self.n_color          = np.int32(n_color)

        L_patchwise_err = (height - y_w + 1) * (width - x_w + 1)
        self.L_patchwise_err = np.int32(L_patchwise_err)

        self.patchwise_err = gpuarray.to_gpu(np.zeros(L_patchwise_err))
        self.err_sq        = gpuarray.to_gpu(np.zeros(self.L_input))
        self.err_sub_frame = gpuarray.to_gpu(np.zeros(self.L_input))
        self.err_sub_frame_pre_map = gpuarray.zeros_like(self.err_sub_frame)

        ceil = lambda x: int(np.ceil(x))
        self.threadsize = threadsize
        self.gridsize = (min(ceil(height / threadsize[0]),
                             max_grid_size),
                         min(ceil(width / threadsize[1]),
                             max_grid_size))
        self.grid_max1 = (min(ceil(L_patchwise_err / maxthreads),
                              max_grid_size), 1)
        self.grid_max2 = (min(ceil(self.grid_max1[0] / maxthreads),
                              max_grid_size), 1)

        self.L_max2  = np.int32(min(self.grid_max1[0],
                                    max_grid_size))
        self.argmax1 = gpuarray.to_gpu(np.zeros(self.L_max2,
                                                dtype=np.uint32))
        self.max1    = gpuarray.to_gpu(np.zeros(self.L_max2,
                                                dtype=np.float64))
        self.argmax2 = gpuarray.to_gpu(np.zeros(1, dtype=np.uint32))
        self.max2    = gpuarray.to_gpu(np.zeros(1, dtype=np.float64))

        rev_map = np.argsort(flat_map).reshape(*dim)
        self.input_frame_rev_shuf = gpuarray.to_gpu(rev_map.astype(np.uint32))
        self.x = 0  # horizontal position from left of the image
        self.y = 0  # vertical position from the top of the image
        self.x_shift = 0
        self.y_shift = 0
        self.x_prev = 0
        self.y_prev = 0
        self.omega = omega  # omega * delta_t
        self.omega_sq = omega**2
        self.gamma = gamma
        self.gamma_omega = gamma * omega
        self.prev_max = 0.

        self.noise = noise

    def reset_state(self, x_pos=0, y_pos=0):
        super(PinholeSaccadicPVM, self).reset_state()
        self.x_prev = x_pos
        self.y_prev = y_pos
        self.prev_max = 0.
        self.x = x_pos
        self.y = y_pos
        self.x_shift = 0.
        self.y_shift = 0.

    def __evolve(self, x_eq, y_eq, max_x, max_y):
        """
        evolving the harmonic oscillator forward in time with error of order
        delta_t**2
        :param x_eq: horizontal equilibrium measured from the left of the
        :param y_eq: vertical equilibrium measured from the top of the
        :return:
        """
        self.x, self.y, self.x_prev, self.y_prev = (
            int(round(((2 - self.omega_sq) * self.x + (self.gamma_omega - 1) *
                      self.x_prev + self.omega_sq * x_eq) /
                      (1 + self.gamma_omega)) +
            np.random.randint(-self.noise, self.noise + 1)),
            int(round(((2 - self.omega_sq) * self.y + (self.gamma_omega - 1) *
                      self.y_prev + self.omega_sq * y_eq) /
                      (1 + self.gamma_omega)) +
            np.random.randint(-self.noise, self.noise + 1)),
            self.x,
            self.y
        )
        if self.x < 0:
            self.x = 0
        elif self.x >= max_x:
            self.x = max_x

        if self.y < 0:
            self.y = 0
        elif self.y >= max_y:
            self.y = max_y

    def forward(self, single_frame):
        full_frame_height, full_frame_width, n_color = single_frame.shape
        x = self.x
        y = self.y
        max_x = full_frame_width - self.sub_frame_width
        max_y = full_frame_height - self.sub_frame_height

        sub_frame = single_frame[y:y + self.sub_frame_height,
                                 x:x + self.sub_frame_width,
                                 :].flatten()

        super(PinholeSaccadicPVM, self).forward(sub_frame)

        self.err_sub_frame_pre_map = self.err[:self.L_input].copy()
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            self.err_sub_frame.gpudata,
                                            self.err_sub_frame_pre_map.gpudata,
                                            self.input_frame_rev_shuf.gpudata,
                                            self.L_input)

        hadamard.prepared_call(self.grid_input, self.threads,
                               self.err_sub_frame.gpudata,
                               self.err_sub_frame.gpudata,
                               self.err_sq.gpudata, self.L_input)

        err_patch_sum.prepared_call(self.gridsize, self.threadsize,
                                    self.err_sq.gpudata,
                                    self.patchwise_err.gpudata,
                                    self.sub_frame_width,
                                    self.sub_frame_height,
                                    self.n_color,
                                    self.x_w,
                                    self.y_w)

        self.max1 = self.patchwise_err[:self.L_patchwise_err]

        max_gpu.prepared_call(self.grid_max1, self.threads,
                              self.patchwise_err.gpudata,
                              self.max1.gpudata,
                              self.argmax1.gpudata,
                              self.L_patchwise_err)

        max_gpu.prepared_call(self.grid_max2, self.threads,
                              self.max1.gpudata,
                              self.max2.gpudata,
                              self.argmax2.gpudata,
                              self.L_max2)

        arg_max = self.argmax1.get()[self.argmax2.get()[0]]
        max_val = self.max2.get()[0]

        if max_val > self.prev_max:
            sub_array_width = (self.sub_frame_width - self.x_w)
            x_shift = arg_max % (sub_array_width + 1)\
                - sub_array_width / 2.
            y_shift = arg_max // (sub_array_width + 1)\
                - (self.sub_frame_height - self.y_w) / 2.
        else:
            x_shift, y_shift = 0., 0.

        self.prev_max = (max_val + self.prev_max) / 2.

        self.x_shift = x_shift
        self.y_shift = y_shift
        self.__evolve(x + x_shift, y + y_shift, max_x, max_y)

    def backward(self, single_frame, next_frame,
                 learning_rate, L2_norm_reg=0):
        self.forward(single_frame)

        x = int(self.x)
        y = int(self.y)

        sub_frame = next_frame[y:y + self.sub_frame_height,
                    x:x + self.sub_frame_width,
                    :].flatten()
        next_input_pre_map = gpuarray.to_gpu(sub_frame) / self.norm
        next_input = gpuarray.zeros_like(next_input_pre_map)
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            next_input.gpudata,
                                            next_input_pre_map.gpudata,
                                            self.input_frame_shuffle.gpudata,
                                            self.L_input)
        base_pvm_gpu_backprop(self, next_input)
        update_parameters(self, learning_rate, reg_factor=L2_norm_reg)

    def train(self, dict_training_data, learning_rate_list, L2_norm_reg=0,
              print_every=100000, save_every_print=False, filename='default',
              interval=100):
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        N_frames = len(learning_rate_list)
        while n < N_frames:
            for key, data in dict_training_data.items():
                n_frames, rows, cols, n_colors = data.shape
                self.reset_state(x_pos=(cols - self.sub_frame_width) // 2,
                                 y_pos=(rows - self.sub_frame_height) // 2)

                for i in range(n_frames - 1):
                    learning_rate = learning_rate_list[n]

                    n += 1
                    single_frame = data[i, ...]
                    next_frame = data[i + 1, ...]
                    self.backward(single_frame, next_frame,
                                  learning_rate,
                                  L2_norm_reg=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred)**2).get()
                                  )
                    mse_sum /= sum(self.pred_mask.get())
                    self.mse.append(mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) / \
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse, interval))
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()
                    if n == N_frames:
                        break
                if n == N_frames:
                    break

    def quick_animate(self, dict_testing_data, wait_time=17, scale=1):
        print("-" * 80 + "\n" + "Animating testing data" + "\n" + "-" * 80)
        print("Use a Keyboard interruption to exit early.")
        print('-' * 80)
        rev_map = self.input_frame_rev_shuf.get()
        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        try:
            for key, data in dict_testing_data.items():
                n_frames, height, width, n_colors = data.shape
                self.reset_state(x_pos=(width - self.sub_frame_width) // 2,
                                 y_pos=(height - self.sub_frame_height) // 2)

                for i in range(n_frames):
                    single_frame = data[i, ...]
                    self.forward(single_frame)
                    cv2.imshow(
                        "Input Frame",
                        rescale(
                            np.array(
                                single_frame,
                                dtype=np.uint8
                            )[..., ::-1]
                        )
                    )
                    cv2.imshow(
                        "Prediction",
                        rescale(
                            np.array(
                                self.norm * self.pred[:self.L_input].get(),
                                dtype=np.uint8
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.imshow(
                        "Error previous frame",
                        rescale(
                            np.array(
                                self.norm * abs(
                                    self.err[:self.L_input].get() - 0.5
                                ),
                                dtype=np.uint8
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()


class MotionIntegrationPVM(OnTheFlyPVM):
    def __init__(self, connect_dict, flat_map, norm_frame=1., norm_pose=1023.,
                 dof=2, act_type='sigmoid', maxthreads=1024,
                 max_grid_size=65535, zero_bias=False):
        super(MotionIntegrationPVM, self).__init__(
            connect_dict, flat_map, norm=norm_frame, act_type=act_type,
            maxthreads=maxthreads, max_grid_size=max_grid_size,
            zero_bias=zero_bias)
        self.norm_pose = norm_pose
        self.dof = np.int32(dof)
        self.pose_weights_size = self.dof * self.L_hidden
        self.pose = gpuarray.to_gpu(np.zeros([dof], dtype=np.float64))
        self.pose_weights = gpuarray.to_gpu(
            np.random.randn(self.L_hidden, dof) / math.sqrt(dof),
        )
        self.grad_pose_weights = gpuarray.zeros_like(self.pose_weights)
        pow = int(math.log2(dof))
        self.pose_threads = (1024//min(2**pow, 32), min(2**pow, 32), 1)
        self.pose_grid = (
            min(int(math.ceil(self.L_hidden / self.pose_threads[0])),
                max_grid_size),
            min(int(math.ceil(dof / self.pose_threads[1])), max_grid_size)
        )
        self.grid_update_pose_weights = (
            min(int(math.ceil(self.pose_weights_size / maxthreads)),
                max_grid_size),
            1
        )

    def forward(self, single_frame, pose):
        assert single_frame.size == self.L_input, \
            "Input frame plus pose must match specified input size"

        self.pose[:] = gpuarray.to_gpu(pose.ravel()) / self.norm_pose
        input_pre_map = gpuarray.to_gpu(single_frame) / self.norm
        input_ = gpuarray.zeros_like(input_pre_map)
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            input_.gpudata,
                                            input_pre_map.gpudata,
                                            self.input_frame_shuffle.gpudata,
                                            self.L_input)


        # This step is to add the inputs for the upper layers which
        # are the hidden from the layers below
        try:
            append_hid.prepared_call(self.grid_int_der_err, self.threads,
                                     input_.gpudata, self.hidden.gpudata,
                                     self.in_and_hid.gpudata,
                                     self.hid_append_map.gpudata,
                                     self.L_input, self.L_pred)
        except:
            self.in_and_hid = input_[:]

        # calculating the derivatives, errors and integrals
        der_and_error.prepared_async_call(self.grid_int_der_err, self.threads,
                                          self.stream1,
                                          self.in_and_hid.gpudata,
                                          self.prev_input.gpudata,
                                          self.der.gpudata, self.L_pred)
        der_and_error.prepared_async_call(self.grid_int_der_err, self.threads,
                                          self.stream2,
                                          self.pred.gpudata,
                                          self.in_and_hid.gpudata,
                                          self.err.gpudata, self.L_pred)
        integral.prepared_async_call(self.grid_int_der_err, self.threads,
                                     self.stream3,
                                     self.in_and_hid.gpudata,
                                     self.int_.gpudata,
                                     self.int_.gpudata, self.L_pred)

        # shuffling the derivative, error, integral, and hidden for
        # the full_input
        input_shuffling.prepared_async_call(self.grid_int_der_err,
                                            self.threads, self.stream4,
                                            self.full_input.gpudata,
                                            input_.gpudata,
                                            self.input_map.gpudata,
                                            self.L_input)

        input_hidden_shuffling.prepared_async_call(self.grid_i2h,
                                                   self.threads, self.stream5,
                                                   self.full_input.gpudata,
                                                   self.hidden.gpudata,
                                                   self.hid_map.gpudata,
                                                   self.L_full_input)

        input_shuffling.prepared_async_call(self.grid_int_der_err,
                                            self.threads, self.stream1,
                                            self.full_input.gpudata,
                                            self.der.gpudata,
                                            self.der_map.gpudata, self.L_pred)

        input_shuffling.prepared_async_call(self.grid_int_der_err,
                                            self.threads, self.stream2,
                                            self.full_input.gpudata,
                                            self.err.gpudata,
                                            self.err_map.gpudata, self.L_pred)

        input_shuffling.prepared_async_call(self.grid_int_der_err,
                                            self.threads, self.stream3,
                                            self.full_input.gpudata,
                                            self.int_.gpudata,
                                            self.int_map.gpudata, self.L_pred)
        zerofill.prepared_async_call(self.grid_i2h, self.threads,
                                     self.stream4,
                                     self.hid_affine.gpudata, self.L_hidden)
        zerofill.prepared_async_call(self.grid_h2op, self.threads,
                                     self.stream5,
                                     self.out_and_pred_affine.gpudata,
                                     self.L_op)

        # storing the
        self.prev_input = self.in_and_hid.copy()

        dot.prepared_call(self.grid_i2h_dot, self.threads,
                          # number of rows
                          self.L_hidden,
                          # CSR sparse matrix
                          self.i2h_pointers.gpudata, self.i2h_indices.gpudata,
                          self.i2h_weights.gpudata,
                          # vector
                          self.full_input.gpudata,
                          # result
                          self.hid_affine.gpudata)

        add.prepared_call(self.grid_i2h, self.threads,
                          self.hid_affine.gpudata, self.i2h_bias.gpudata,
                          self.hid_affine.gpudata, self.L_hidden)

        dense_dot.prepared_call(self.pose_grid, self.pose_threads,
                                self.pose_weights.gpudata,
                                self.pose.gpudata,
                                self.hid_affine.gpudata,
                                self.dof,
                                self.L_hidden)

        self.act.prepared_call(self.grid_i2h, self.threads,
                               self.hid_affine.gpudata, self.hidden.gpudata,
                               self.L_hidden)

        dot.prepared_call(self.grid_h2op_dot, self.threads,
                          # number of rows
                          self.L_op,
                          # CSR sparse matrix
                          self.h2op_pointers.gpudata,
                          self.h2op_indices.gpudata,
                          self.h2op_weights.gpudata,
                          # vector
                          self.hidden.gpudata,
                          # results
                          self.out_and_pred_affine.gpudata)

        add.prepared_call(self.grid_h2op, self.threads,
                          self.out_and_pred_affine.gpudata,
                          self.h2op_bias.gpudata,
                          self.out_and_pred_affine.gpudata,
                          self.L_op)

        self.act.prepared_call(self.grid_h2op, self.threads,
                               self.out_and_pred_affine.gpudata,
                               self.out_and_pred.gpudata,
                               self.L_op)

        if self.L_out > 0:
            output_pred_shuffling.prepared_async_call(
                self.grid_o_shuf,
                self.threads,
                self.stream4,
                self.output.gpudata,
                self.out_and_pred.gpudata,
                self.out_map.gpudata,
                self.L_out
            )

        output_pred_shuffling.prepared_async_call(
            self.grid_int_der_err,
            self.threads, self.stream5,
            self.pred.gpudata,
            self.out_and_pred.gpudata,
            self.pred_map.gpudata,
            self.L_pred
        )

    def backward(self, single_frame, pose,
                 next_frame, learning_rate,
                 L2_norm_reg=0):
        self.forward(single_frame, pose)

        next_input_pre_map = gpuarray.to_gpu(
            next_frame.ravel()) / self.norm
        next_input = gpuarray.zeros_like(next_input_pre_map)
        output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                            next_input.gpudata,
                                            next_input_pre_map.gpudata,
                                            self.input_frame_shuffle.gpudata,
                                            self.L_input)
        base_pvm_gpu_backprop(self, next_input)
        dense_kron.prepared_call(self.pose_grid, self.pose_threads,
                                 self.pose.gpudata,
                                 self.delta_i2h.gpudata,
                                 self.grad_pose_weights.gpudata,
                                 self.dof, self.L_hidden)

        f = 2 * learning_rate * L2_norm_reg
        update.prepared_async_call(self.grid_update_pose_weights,
                                   self.threads,
                                   self.stream3,
                                   self.pose_weights.gpudata,
                                   self.pose_weights.gpudata,
                                   f,
                                   self.pose_weights_size)

        update.prepared_async_call(self.grid_update_pose_weights,
                                   self.threads,
                                   self.stream3,
                                   self.pose_weights.gpudata,
                                   self.grad_pose_weights.gpudata,
                                   learning_rate, self.pose_weights_size)

        self._update(self, learning_rate, reg_factor=L2_norm_reg)

    def save_parameters(self, filename):
        with open(filename + '_connections.pkl', 'wb') as fid:
            cP.dump((self.connect_dict, self.dof), fid)
        np.savez(filename, i2h_pointers=self.i2h_pointers.get(),
                 i2h_indices=self.i2h_indices.get(),
                 i2h_weights=self.i2h_weights.get(),
                 i2h_bias=self.i2h_bias.get(),
                 h2op_pointers=self.h2op_pointers.get(),
                 h2op_indices=self.h2op_indices.get(),
                 h2op_weights=self.h2op_weights.get(),
                 h2op_bias=self.h2op_bias.get(),
                 pose_weights=self.pose_weights.get())

    def transfer_parameters(self, filename):
        self.pred_mask[self.L_input:] -= 1.
        with open(filename + '_connections.pkl', 'rb') as fid:
            should_be_connect_dict = cP.load(fid)
        if self.connect_dict != dict(should_be_connect_dict):
            raise ValueError("The model you are trying to load has "
                             + "different connections "
                             + "It's incompatible with the current instance.")
        par_dict = np.load(filename + '.npz')
        self.i2h_pointers     = gpuarray.to_gpu(par_dict['i2h_pointers'])
        self.i2h_indices      = gpuarray.to_gpu(par_dict['i2h_indices'])
        self.i2h_weights      = gpuarray.to_gpu(par_dict['i2h_weights'])
        self.i2h_bias         = gpuarray.to_gpu(par_dict['i2h_bias'])
        self.h2op_pointers    = gpuarray.to_gpu(par_dict['h2op_pointers'])
        self.h2op_indices     = gpuarray.to_gpu(par_dict['h2op_indices'])
        self.h2op_weights     = gpuarray.to_gpu(par_dict['h2op_weights'])
        self.h2op_bias        = gpuarray.to_gpu(par_dict['h2op_bias'])

    def load_parameters(self, filename):
        with open(filename + '_connections.pkl', 'rb') as fid:
            should_be_connect_dict, should_be_dof = cP.load(fid)
        if self.connect_dict != dict(should_be_connect_dict) or \
                self.dof != should_be_dof:
            raise ValueError("The model you are trying to load has "
                             + "different connections or degrees of freedom.\n"
                             + "It's incompatible with the current instance.")
        par_dict = np.load(filename + '.npz')
        self.i2h_pointers     = gpuarray.to_gpu(par_dict['i2h_pointers'])
        self.i2h_indices      = gpuarray.to_gpu(par_dict['i2h_indices'])
        self.i2h_weights      = gpuarray.to_gpu(par_dict['i2h_weights'])
        self.i2h_bias         = gpuarray.to_gpu(par_dict['i2h_bias'])
        self.h2op_pointers    = gpuarray.to_gpu(par_dict['h2op_pointers'])
        self.h2op_indices     = gpuarray.to_gpu(par_dict['h2op_indices'])
        self.h2op_weights     = gpuarray.to_gpu(par_dict['h2op_weights'])
        self.h2op_bias        = gpuarray.to_gpu(par_dict['h2op_bias'])
        self.pose_weights     = gpuarray.to_gpu(par_dict['pose_weights'])

    def train(self, dict_training_data, learning_rate_list, L2_norm_reg=0,
              print_every=100000, save_every_print=False,
              filename='default', interval=100):
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        N_frames = len(learning_rate_list)
        while n < N_frames:
            for key in dict_training_data.keys():
                frame_data = dict_training_data[key]['images']
                pose_data = dict_training_data[key]['position']
                n_frames, rows, cols, n_colors = frame_data.shape
                self.reset_state()

                for i in range(n_frames - 1):
                    learning_rate = learning_rate_list[n]

                    n += 1
                    pan_tilt_pose = pose_data[i, ...]
                    single_frame = frame_data[i, ...]
                    next_frame = frame_data[i + 1, ...]
                    self.backward(single_frame, pan_tilt_pose,
                                  next_frame,
                                  learning_rate,
                                  L2_norm_reg=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred) ** 2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) / \
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse, interval), 'b')
                            plt.plot(movingaverage(self.img_mse, interval), 'r')
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()
                    if n == N_frames:
                        break
                if n == N_frames:
                    break

    def adam_train(self, dict_training_data, N_epoch, L2_norm_reg=0,
              print_every=100000, save_every_print=False,
              filename='default', interval=100,
              alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-8):
        # attribute outside of __init__ but nobody cares
        self.m_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.v_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.mhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.vhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.dev_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)

        self.m_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.v_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.mhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.vhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.dev_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)

        self.m_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.v_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.mhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.vhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.dev_weights_h2op =gpuarray.zeros_like(self.grad_weight_h2op)

        self.m_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.v_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.mhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.vhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.dev_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)

        self.m_t_weights_pose = gpuarray.zeros_like(self.grad_pose_weights)
        self.v_t_weights_pose = gpuarray.zeros_like(self.grad_pose_weights)
        self.mhat_t_weights_pose = \
            gpuarray.zeros_like(self.grad_pose_weights)
        self.vhat_t_weights_pose = \
            gpuarray.zeros_like(self.grad_pose_weights)
        self.dev_weights_pose = gpuarray.zeros_like(self.grad_pose_weights)
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        for _ in range(N_epoch):
            for key in dict_training_data.keys():
                frame_data = dict_training_data[key]['images']
                pose_data = dict_training_data[key]['position']
                n_frames, *junk = frame_data.shape
                self.reset_state()

                self.forward(frame_data[0, ...], pose_data[0, ...])
                for i in range(1, n_frames - 1):
                    n += 1
                    single_frame = frame_data[i, ...]
                    pan_tilt_pose = pose_data[i, ...]
                    next_frame = frame_data[i + 1, ...]
                    next_input = gpuarray.to_gpu(next_frame)

                    self.forward(single_frame, pan_tilt_pose)
                    next_input_pre_map = gpuarray.to_gpu(
                        next_frame.ravel()) / self.norm
                    next_input = gpuarray.zeros_like(next_input_pre_map)
                    output_pred_shuffling.prepared_call(
                        self.grid_input, self.threads,
                        next_input.gpudata,
                        next_input_pre_map.gpudata,
                        self.input_frame_shuffle.gpudata,
                        self.L_input
                    )
                    base_pvm_gpu_backprop(self, next_input)
                    dense_kron.prepared_call(self.pose_grid,
                                             self.pose_threads,
                                             self.pose.gpudata,
                                             self.delta_i2h.gpudata,
                                             self.grad_pose_weights.gpudata,
                                             self.dof, self.L_hidden)

                    reg_pose_weights_grad = (
                        self.grad_pose_weights *
                        (1. + 2 * L2_norm_reg * self.pose_weights)
                    )
                    axpby.prepared_call(self.grid_update_pose_weights,
                                        self.threads,
                                        self.m_t_weights_pose.gpudata,
                                        reg_pose_weights_grad.gpudata,
                                        beta1, 1. - beta1,
                                        self.pose_weights_size)
                    axpbyy.prepared_call(self.grid_update_pose_weights,
                                         self.threads,
                                         self.v_t_weights_pose.gpudata,
                                         reg_pose_weights_grad.gpudata,
                                         beta2, 1. - beta2,
                                         self.pose_weights_size)
                    axpby.prepared_call(self.grid_update_pose_weights
                                        , self.threads,
                                        self.mhat_t_weights_pose.gpudata,
                                        self.m_t_weights_pose.gpudata,
                                        0, 1. / (1. - beta1 ** n),
                                        self.pose_weights_size)
                    axpby.prepared_call(self.grid_update_pose_weights,
                                        self.threads,
                                        self.vhat_t_weights_pose.gpudata,
                                        self.v_t_weights_pose.gpudata,
                                        0, 1. / (1. - beta2 ** n),
                                        self.pose_weights_size)
                    gradmod.prepared_call(self.grid_update_pose_weights,
                                          self.threads,
                                          self.mhat_t_weights_pose.gpudata,
                                          self.vhat_t_weights_pose.gpudata,
                                          self.dev_weights_pose.gpudata,
                                          eps, self.pose_weights_size)
                    adam_update_parameters(self, n,
                                           alpha=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           eps=eps,
                                           reg_factor=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred)**2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) /\
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse, interval), 'b')
                            plt.plot(movingaverage(self.img_mse, interval), 'r')
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()

    def quick_animate(self, dict_testing_data, wait_time=17, scale=1):
        print("-" * 80 + "\n" + "Animating testing data" + "\n" + "-" * 80)
        print("Use a Keyboard interruption to exit early.")
        print('-' * 80)
        datum = dict_testing_data[list(dict_testing_data.keys())[0]]

        frame_datum = datum['images']
        # pose_datum =datum['position']
        _, height, width, n_colors = frame_datum.shape
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(height, width, n_colors)

        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)

        try:
            for key, data in dict_testing_data.items():
                frame_data = dict_testing_data[key]['images']
                n_frames, height, width, n_colors = frame_data.shape
                pose_data = dict_testing_data[key]['position']
                self.reset_state()

                for i in range(n_frames):
                    single_frame = frame_data[i, ...]
                    single_pose = pose_data[i, ...]
                    self.forward(single_frame, single_pose)
                    cv2.imshow(
                        "Input Frame",
                        rescale(
                            np.array(
                                single_frame
                            )[..., ::-1]
                        )
                    )
                    cv2.imshow(
                        "Prediction",
                        rescale(
                            np.array(
                                self.pred[:self.L_input].get()
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.imshow(
                        "Error previous frame",
                        rescale(
                            np.array(
                                abs(
                                    self.err[:self.L_input].get() - 0.5
                                )
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()


class TransformMotionIntegrationPVM(OnTheFlyPVM):
    def __init__(self, connect_dict, flat_map, width, norm=1., fov=75,
                 act_type='sigmoid', maxthreads=1024, max_grid_size=65535,
                 zero_bias=False):
        super(TransformMotionIntegrationPVM, self).__init__(connect_dict,
                                                            flat_map,
                                                            norm,
                                                            act_type,
                                                            maxthreads,
                                                            max_grid_size,
                                                            zero_bias)

        self.f = width / (2 * np.tan(fov * np.pi / 2 / 180))

    def forward(self, single_frame, pose, prev_pose):
        height, width, n_color = single_frame.shape
        theta1, phi1 = (prev_pose[:2] - 512. / 1023.) * 300. * np.pi / 180.
        theta2, phi2 = (pose[:2] - 512. / 1023.) * 300. * np.pi / 180.

        cosphi1 = np.cos(phi1)
        sinphi1 = np.sin(phi1)

        costheta = np.cos(theta2 - theta1)
        sintheta = np.sin(theta2 - theta1)

        cosphi2 = np.cos(phi2)
        sinphi2 = np.sin(phi2)

        M = np.array(
            [[costheta,
              sinphi1 * sintheta,
              -cosphi1 * sintheta],
             [-sinphi2 * sintheta,
              sinphi2 * sinphi1 * costheta + cosphi2 * cosphi1,
              -sinphi2 * cosphi1 * costheta + cosphi2 * sinphi1],
             [cosphi2 * sintheta,
              -sinphi1 * cosphi2 * costheta + sinphi2 * cosphi1,
              cosphi1 * cosphi2 * costheta + sinphi2 * sinphi1]
             ]
        )

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        xx = (xx - width / 2) / self.f
        yy = (yy - height / 2) / self.f

        XX = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
        YY = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]

        X = (XX * self.f + width / 2).astype(np.int32)
        Y = (YY * self.f + height / 2).astype(np.int32)
        X[np.less(X, 0)] = 0
        X[np.greater_equal(X, width)] = width - 1
        Y[np.less(Y, 0)] = 0
        Y[np.greater_equal(Y, height)] = height - 1

        transform_frame = single_frame[Y, X, :]

        super(TransformMotionIntegrationPVM, self).forward(transform_frame)

    def backward(self, single_frame, pose, prev_pose, next_frame,
                 learning_rate, L2_norm_reg=0):
        self.forward(single_frame, pose, prev_pose)

        next_input_pre_map = gpuarray.to_gpu(
            next_frame.ravel()) / self.norm

        next_input = gpuarray.zeros_like(next_input_pre_map)

        output_pred_shuffling.prepared_call(
            self.grid_input, self.threads,
            next_input.gpudata,
            next_input_pre_map.gpudata,
            self.input_frame_shuffle.gpudata,
            self.L_input
        )

        base_pvm_gpu_backprop(self, next_input)

    def train(self, dict_training_data, learning_rate_list, L2_norm_reg=0,
              print_every=100000, save_every_print=False,
              filename='default', interval=100):
        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        N_frames = len(learning_rate_list)
        while n < N_frames:
            for key in dict_training_data.keys():
                frame_data = dict_training_data[key]['images']
                pose_data = dict_training_data[key]['position']
                n_frames, rows, cols, n_colors = frame_data.shape
                self.reset_state()

                for i in range(n_frames - 1):
                    learning_rate = learning_rate_list[n]

                    n += 1
                    pan_tilt_pose = pose_data[i + 1, ...]
                    prev_pose = pose_data[i, ...]
                    single_frame = frame_data[i, ...]
                    next_frame = frame_data[i + 1, ...]
                    self.backward(single_frame, pan_tilt_pose, prev_pose,
                                  next_frame,
                                  learning_rate,
                                  L2_norm_reg=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred) ** 2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
                    if n % print_every == 0:
                        mse_avg = sum(self.mse[-print_every:]) / \
                                  print_every
                        print("{:>10} frames: {:>10}".format(n, mse_avg))
                        if save_every_print:
                            self.save_parameters(filename)
                            plt.plot(movingaverage(self.mse, interval), 'b')
                            plt.plot(movingaverage(self.img_mse, interval), 'r')
                            plt.savefig(filename +
                                        'training_moving_avg' +
                                        '_MSE_vs_frames.pdf',
                                        transparent=True)
                            plt.close()
                    if n == N_frames:
                        break
                if n == N_frames:
                    break

    def adam_train(self, dict_training_data, N_epoch, L2_norm_reg=0,
              print_every_epoch=True, save_every_print=False,
              filename='default', interval=100,
              alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-8):
        # attribute outside of __init__ but nobody cares
        self.m_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.v_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.mhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.vhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.dev_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)

        self.m_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.v_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.mhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.vhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.dev_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)

        self.m_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.v_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.mhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.vhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.dev_weights_h2op =gpuarray.zeros_like(self.grad_weight_h2op)

        self.m_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.v_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.mhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.vhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.dev_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)

        print("-" * 80)
        print(" " * 19 + 'AVG MSE over last {} frames'.format(print_every))
        n = 0
        for _ in range(N_epoch):
            for key in dict_training_data.keys():
                frame_data = dict_training_data[key]['images']
                pose_data = dict_training_data[key]['position']
                n_frames, *junk = frame_data.shape
                self.reset_state()

                #self.forward(frame_data[0, ...], pose_data[0, ...])
                for i in range(n_frames - 1):
                    n += 1
                    single_frame = frame_data[i, ...]
                    prev_pose = pose_data[i, ...]
                    pan_tilt_pose = pose_data[i + 1, ...]
                    next_frame = frame_data[i + 1, ...]
                    # next_input = gpuarray.to_gpu(next_frame)

                    self.forward(single_frame, pan_tilt_pose, prev_pose)
                    next_input_pre_map = gpuarray.to_gpu(
                        next_frame.ravel()) / self.norm
                    next_input = gpuarray.zeros_like(next_input_pre_map)
                    output_pred_shuffling.prepared_call(
                        self.grid_input, self.threads,
                        next_input.gpudata,
                        next_input_pre_map.gpudata,
                        self.input_frame_shuffle.gpudata,
                        self.L_input
                    )
                    base_pvm_gpu_backprop(self, next_input)
                    adam_update_parameters(self, n,
                                           alpha=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           eps=eps,
                                           reg_factor=L2_norm_reg)
                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred)**2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
            print(" " * 19 + 'AVG MSE over last {} frames'.format(n))
            if print_every_epoch:
                mse_avg = sum(self.mse[-n:]) / n
                print("{:>10} Epoch: {:>10}".format(epoch, mse_avg))
                if save_every_print:
                    self.save_parameters(filename)
                    plt.plot(movingaverage(self.mse, interval), 'b')
                    plt.plot(movingaverage(self.img_mse, interval), 'r')
                    plt.legend(['Training error',
                                'Training error (image only)'])
                    plt.savefig(filename +
                                'training_moving_avg' +
                                '_MSE_vs_frames.pdf',
                                transparent=True)
                    plt.close()

    def quick_animate(self, dict_testing_data, wait_time=17, scale=1):
        print("-" * 80 + "\n" + "Animating testing data" + "\n" + "-" * 80)
        print("Use a Keyboard interruption to exit early.")
        print('-' * 80)
        datum = dict_testing_data[list(dict_testing_data.keys())[0]]

        frame_datum = datum['images']
        # pose_datum =datum['position']
        _, height, width, n_colors = frame_datum.shape
        rev_map = np.argsort(
            self.input_frame_shuffle.get()
        ).reshape(height, width, n_colors)

        def rescale(img):
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)

        try:
            for key, data in dict_testing_data.items():
                frame_data = dict_testing_data[key]['images']
                n_frames, height, width, n_colors = frame_data.shape
                pose_data = dict_testing_data[key]['position']
                self.reset_state()

                for i in range(1, n_frames):
                    single_frame = frame_data[i, ...]
                    single_pose = pose_data[i, ...]
                    prev_pose = pose_data[i - 1, ...]
                    self.forward(single_frame, single_pose, prev_pose)
                    cv2.imshow(
                        "Input Frame",
                        rescale(
                            np.array(
                                single_frame
                            )[..., ::-1]
                        )
                    )
                    cv2.imshow(
                        "Prediction",
                        rescale(
                            np.array(
                                self.pred[:self.L_input].get()
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.imshow(
                        "Error previous frame",
                        rescale(
                            np.array(
                                abs(
                                    self.err[:self.L_input].get() - 0.5
                                )
                            )[rev_map[..., ::-1]]
                        )
                    )
                    cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()

    def test(self, dict_testing_data):
        tot_frames_in_epoch = 0
        img_test_error = gpuarray.to_gpu(np.zeros(self.L_input, dtype=np.float64))
        test_error = gpuarray.to_gpu(np.zeros(self.L_pred, dtype=np.float64))
        for key, data in dict_testing_data.items():
            frames = data['images']
            n_frame, row, column, n_colors = frames.shape
            position = data['position']

            for i in range(n_frame - 1):
                self.forward(frames[i, ...], position[i + 1, ...], position[i, ...])
                next_frame = frames[i + 1, ...]
                next_input_pre_map = gpuarray.to_gpu(next_frame) / self.norm
                next_input = gpuarray.zeros_like(next_input_pre_map)
                output_pred_shuffling.prepared_call(self.grid_input, self.threads,
                                                    next_input.gpudata,
                                                    next_input_pre_map.gpudata,
                                                    self.input_frame_shuffle.gpudata,
                                                    self.L_input)
                # ignore appending the hidden values fed to upper layers to the input
                try:
                    append_hid.prepared_call(self.grid_int_der_err, self.threads,
                                             next_input.gpudata, self.hidden.gpudata,
                                             self.ideal_pred.gpudata,
                                             self.hid_append_map.gpudata,
                                             self.L_input,
                                             self.L_pred)
                except:
                    self.ideal_pred = next_input[:]

                # calculating the gradients in error using MSE
                sub.prepared_async_call(self.grid_int_der_err, self.threads,
                                        self.stream1,
                                        self.pred.gpudata, self.ideal_pred.gpudata,
                                        self.delta_pred.gpudata, self.L_pred)
                mask = self.pred_mask.get()
                test_error += (self.delta_pred) ** 2 / sum(mask)
                img_test_error += (self.delta_pred[:self.L_input]) ** 2 \
                                  / sum(mask[:self.L_input])
                tot_frames_in_epoch += 1
        avg_err = sum(test_error) / tot_frames_in_epoch
        img_avg_err = sum(img_test_error) / tot_frames_in_epoch
        return (tot_frames_in_epoch, np.asscalar(avg_err.get()),
                np.asscalar(img_avg_err.get()))

    def adam_train_and_validate(self, dict_training_data, dict_valid_data,
                                N_epoch, L2_norm_reg=0,
                                print_every_epoch=True,
                                save_every_print=False, filename='default',
                                interval=100, alpha=0.0001, beta1=0.9,
                                beta2=0.999, eps=1e-8):
        # nobody cares
        self.m_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.v_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.mhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.vhat_t_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)
        self.dev_bias_h2op = gpuarray.zeros_like(self.grad_bias_h2op)

        self.m_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.v_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.mhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.vhat_t_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)
        self.dev_bias_i2h = gpuarray.zeros_like(self.grad_bias_i2h)

        self.m_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.v_t_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)
        self.mhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.vhat_t_weights_h2op = \
            gpuarray.zeros_like(self.grad_weight_h2op)
        self.dev_weights_h2op = gpuarray.zeros_like(self.grad_weight_h2op)

        self.m_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.v_t_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        self.mhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.vhat_t_weights_i2h = \
            gpuarray.zeros_like(self.grad_weight_i2h)
        self.dev_weights_i2h = gpuarray.zeros_like(self.grad_weight_i2h)
        print("-" * 80)
        n = 0
        valid_err_list = []
        img_valid_err_list = []
        for epoch in range(N_epoch):
            for key, data in dict_training_data.items():
                frame_data = dict_training_data[key]['images']
                pose_data = dict_training_data[key]['position']
                n_frames, *junk = frame_data.shape
                self.reset_state()

                for i in range(n_frames - 1):
                    if epoch == 0:
                        n += 1
                    single_frame = frame_data[i, ...]
                    prev_pose = pose_data[i, ...]
                    pan_tilt_pose = pose_data[i + 1, ...]
                    next_frame = frame_data[i + 1, ...]

                    self.forward(single_frame, pan_tilt_pose, prev_pose)
                    next_input_pre_map = gpuarray.to_gpu(next_frame) \
                                         / self.norm
                    next_input = gpuarray.zeros_like(next_input_pre_map)
                    output_pred_shuffling.prepared_call(
                        self.grid_input,
                        self.threads,
                        next_input.gpudata,
                        next_input_pre_map.gpudata,
                        self.input_frame_shuffle.gpudata,
                        self.L_input)
                    base_pvm_gpu_backprop(self, next_input)
                    adam_update_parameters(self, n,
                                           alpha=alpha,
                                           beta1=beta1,
                                           beta2=beta2,
                                           eps=eps,
                                           reg_factor=L2_norm_reg)

                    mse_sum = sum((self.pred_mask *
                                   (self.pred - self.ideal_pred) ** 2).get()
                                  )
                    img_mse_sum = sum((self.pred_mask *
                                       (self.pred -
                                        self.ideal_pred)
                                       ** 2)[:self.L_input].get()
                                      )
                    mse_sum /= sum(self.pred_mask.get())
                    img_mse_sum /= sum(self.pred_mask[:self.L_input].get())
                    self.mse.append(mse_sum)
                    self.img_mse.append(img_mse_sum)
            # print(" " * 19 + 'AVG MSE over last {} frames'.format(n))
            if print_every_epoch:
                mse_avg = sum(self.mse[-n:]) / n
                print("{:>10} Epoch Training Error: {:>10}".format(epoch,
                                                                   mse_avg))
                valid_tot_frames, val_err, img_val_err = \
                    self.test(dict_valid_data)
                valid_err_list.append(val_err)
                img_valid_err_list.append(img_val_err)
                print("{:>10} Epoch Validation Error: {:>10}".format(epoch,
                                                                     val_err))
                if save_every_print:
                    self.save_parameters(filename)
                    plt.plot(movingaverage(self.mse, interval), 'b')
                    plt.plot(movingaverage(self.img_mse, interval), 'r')
                    plt.plot(n * (np.arange(1, epoch + 2) - 0.5), valid_err_list,
                             '.--b', markersize=10)
                    plt.plot(n * (np.arange(1, epoch + 2) - 0.5), img_valid_err_list,
                             '.--r', markersize=10)
                    plt.legend(['Training error',
                                'Training error (image only)',
                                'Validation error',
                                'Validation error (image only)'])
                    plt.savefig(filename +
                                'training_moving_avg' +
                                '_MSE_vs_frames.pdf',
                                transparent=True)
                    plt.close()

