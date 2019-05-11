import numpy as np
from pycuda import gpuarray, compiler
from collections import OrderedDict
import os
import h5py
import sys

# pick your device the default is 0 if not specified if the next line is not commented
os.environ['CUDA_DEVICE'] = '1'

# autoinit automatically initializes a CUDA context
import pycuda.autoinit

from PVM_PyCUDA import OnTheFlyPVM
from FormattingFiles import flatten_image  # , unflatten_image
from RectangularGridConstructor import make_connections  # , break_stuff


def main(epochs):
    # The parameters for the PVM they are different from the original paper
    n_color = 3
    input_edge_x, input_edge_y = 1, 1
    input_size = input_edge_x * input_edge_y * n_color
    hidden_size = 5
    inner_hidden_size = 5
    output_sizes = [0] * 9  # 8#
    inner_output_size = 0
    structure = [(128, 96), (64, 48), (32, 24), (16, 12), (8, 6), (4, 3), (3, 2), (2, 1), 1]

    edge_n_pixels_x, edge_n_pixels_y = input_edge_x * structure[0][0], input_edge_y * structure[0][1]

    # initialize any instance of a PVM you need to specify how it's connected
    # this can be as general as you want in principle as connectivity is
    # defined in dictionary. The function make_connections is a way to
    # construct a layered hierarchy of rectangular grids with nearest neighbor lateral connections
    # was done in the paper
    connect_dict = make_connections(structure, input_size, hidden_size,
                                    output_sizes, context_from_top_0_0=True)

    dim = (edge_n_pixels_y, edge_n_pixels_x, 3)
    input_shape = (input_edge_y, input_edge_x)
    basic_index = np.arange(np.prod(dim)).reshape(dim)
    flat_map = flatten_image(basic_index, input_shape)

    path = '/media/sdb/'
    train_filename = path + 'PVM_movement_integration_test_set_96h_by_128w_no_position.hdf5'
    test_filename = path + 'PVM_movement_integration_train_set_96h_by_128w_no_position.hdf5'

    train_data = h5py.File(train_filename, 'r')
    test_data =  h5py.File(test_filename, 'r')

    tot_frames_in_epoch = 0
    for key, data in train_data.items():
        n_frame, row, column, n_colors = data.shape

        for _ in range(n_frame - 1):
            tot_frames_in_epoch += 1

    pvm = OnTheFlyPVM(connect_dict, flat_map, norm=255.,
                      act_type='sigmoid', maxthreads=1024,
                      max_grid_size=65535, zero_bias=False)

    pvm.adam_train_and_validate(train_data,
                                test_data,
                                epochs,
                                print_every_epoch=True,
                                save_every_print=True,
                                filename=path +
                                    '/MotionIntegration/' +
                                    'swap_motion_integration_data_with_no_servo_values_adam_1_pixel_per_unit_hidden_5_epochs_' +
                                    str(epochs),
                                interval=tot_frames_in_epoch
                                )


if __name__ == "__main__":
    main(int(sys.argv[1]))