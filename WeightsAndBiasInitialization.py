from __future__ import absolute_import, print_function, division
from builtins import *
import numpy as np
import math
from RectangularGridConstructor import shape_check


def weight_initialize(connect_dict):
    """
    Glorot initializing the weight matrix of each PVM unit based on the
    output of the function make_connections or any other ordered dictionary
    with keys of the format '_{layer}_' followed by whatever you want with
    values (unit_count, input_size, hidden_size, output_size, fedfromlist,
    latsuplist)

    :param connect_dict: An ordered dictionary that
    maps a string representing a PVM unit to a tuple with

    (unit_count, input_size, hidden_size, output_size, fedfrom_list,
     latsup_list)

    unit_count is a numerical value associated with the key
    input_size is an integer representing the size of the input to that
    PVM unit in the input layer
    hidden_size is the size of the hidden state variable of each PVM unit
    fedfrom_list is a list of the keys for the PVM units feeding into the unit
    latsup_list is a list of the keys that send their context to the PVM unit

    :param hidden_size: An integer representing the size of the hidden of
    each PVM unit.

    :return : A tuple containing the input to hidden weights and biases.
    The zeroth element being the pointer array for a CSR format of the
    weights
    The first element being the indices of the columns in an array (CSR
    format)
    The second element being the values of the respective weights (CSR format)
    The third element being the biases

    :return : A tuple containing the hidden to output and prediction weights
    and biases.
    The zeroth element being the pointer array for a CSR format of the
    weights
    The first element being the indices of the columns in an array (CSR
    format)
    The second element being the values of the respective weights (CSR format)
    The third element being the biases

    :return : A tuple containing the mapping arrays.
    The zeroth element being the pointer array for a CSR format of the
    weights
    The first element being the indices of the columns in an array (CSR
    format)
    The second element being the values of the respective weights (CSR format)
    The third element being the biases
    :return input_new_unit: list containing the index of first element of input
    for each unit (it's is length N + 1, N being the number of PVM units ends 
    with the total length of the input)
    :return op_new_unit: list containing the index of first element of the 
    output and prediction for each unit (it's is length N + 1, N being the 
    number of PVM units ends with the total length of the output & prediction)
    """

    # lists with mappings for memory shuffling
    input_map = []  # full_input[input_map[i]] = input[i]
    der_map = []  # full_input[der_map[i]] = der[i]
    int_map = []  # full_input[int_map[i]] = integral[i]
    err_map = []  # full_input[err_map[i]] = error[i]

    hid_map = []  # full_input[j] = hidden[hid_map[j]] if hid_map[j] != -1
    hid_append_map = []
    # input_app[len(input)+i] = hidden[hid_append_map[i]]

    out_map = []  # output[i] = op[out_map[i]]
    pred_map = []  # pred[i] = op[pred_map[i]]

    # inputs (+ context) to hidden weights and biases
    i2h_bias = []
    # weights will be a sparse array in CSR format
    i2h_weights = []  # the values of the weights
    i2h_pointers = [0]  # sparse array pointers
    i2h_indices = []  # column indices

    # contains the index of each PVM unit in the raw input
    input_new_unit = [0]

    # hidden to output & prediction weights and biases
    h2op_bias = []
    # weights will be a sparse array in CSR format
    h2op_weights = []
    h2op_pointers = [0]
    h2op_indices = []

    # contains the index of each PVM unit of the
    # output and predictions will be of length N_pmv_units + 1
    op_new_unit = [0]
    
    # contains the index of each PVM unit of the hidden
    hid_new_unit = [0]

    for key, val in connect_dict.items():
        unit_count, input_size, hidden_size, \
        output_size, fedfromlist, latsuplist = val

        # I need a way to keep track of all the hidden variables
        # I have no choice but to iterate over the connection
        # dictionary twice
        hid_new_unit.append(hid_new_unit[-1] + hidden_size)


    i2h_col_start = 0
    h2op_col_start = 0
    for key, val in connect_dict.items():
        unit_count, input_size, hidden_size,\
        output_size, fedfromlist, latsuplist = val

        # counting the number of units feeding into the array
        N_units_feeding_into = len(fedfromlist)

        # the mappings for inputs, derivatives and integral
        if N_units_feeding_into == 0:
            raw_fed_size = input_size
            # raw_fed_size is the same as input_size in this case
            input_map += list(range(i2h_col_start,
                                    i2h_col_start + input_size))
        else:
            if input_size != 0:
                raise ValueError('Units fed hidden values cannot have '
                                 'direct inputs. Please use a separate '
                                 'unit.')
            raw_fed_size = 0
            for feeding_unit in fedfromlist:
                # adding the unique hidden lengths to raw_fed_size
                raw_fed_size += connect_dict[feeding_unit][2]
        der_map += list(range(i2h_col_start + raw_fed_size,
                              i2h_col_start + 2 * raw_fed_size))
        int_map += list(range(i2h_col_start + 2 * raw_fed_size,
                              i2h_col_start + 3 * raw_fed_size))
        err_map += list(range(i2h_col_start + 3 * raw_fed_size,
                              i2h_col_start + 4 * raw_fed_size))

        # input, derivative, integral, error, hidden and
        # lateral/superior context are all contributing to
        # the input size to the sigmoid layer
        fed_size = 4 * raw_fed_size + hidden_size
        for latsup_unit in latsuplist:
            # adding the unique hidden lengths to fed_size
            fed_size += connect_dict[latsup_unit][2]

        # prediction of the input and output heatmap
        full_output_size = raw_fed_size + output_size

        i2h_col_end = i2h_col_start + fed_size
        # assigning the weights and bias for hidden calculation
        for row in range(hidden_size):
            # random Gaussian variables with 1/fed_size variance
            i2h_weights.append(np.random.randn(fed_size)
                               / math.sqrt(fed_size))
            i2h_pointers.append(i2h_pointers[-1] + fed_size)
            i2h_indices.append(np.arange(i2h_col_start, i2h_col_end))

        i2h_bias.append(np.random.randn(fed_size))

        # initialize all hid_map in the relevant range to -1
        hid_map += [-1] * fed_size


        hid_start = i2h_col_start
        hid_end = hid_start
        for unit in fedfromlist:
            uc = connect_dict[unit][0]  # unit count of fed units
            u_hid_size = connect_dict[unit][2]
            hid_end += u_hid_size
            hid_map[hid_start:hid_end] = list(range(hid_new_unit[uc],
                                                    hid_new_unit[uc + 1]))
            hid_append_map += hid_map[hid_start:hid_end]
            hid_start += u_hid_size

        hid_start = i2h_col_start + 4 * raw_fed_size
        hid_end = hid_start + hidden_size

        hid_map[hid_start:hid_end] = list(range(hid_new_unit[unit_count],
                                                hid_new_unit[unit_count + 1]
                                                )
                                          )

        for unit in latsuplist:
            u_hid_size = connect_dict[unit][2]
            hid_start += u_hid_size
            hid_end += u_hid_size
            uc = connect_dict[unit][0]  # unit count of lat and superior
            hid_map[hid_start:hid_end] = list(range(hid_new_unit[uc],
                                                    hid_new_unit[uc + 1]))

        i2h_col_start = i2h_col_end
        input_new_unit.append(i2h_col_start)

        h2op_col_end = h2op_col_start + hidden_size
        # assigning the weights and biases of output + prediction
        for row in range(full_output_size):
            # random Gaussian variables with 1/hidden_size variance
            h2op_weights.append(np.random.randn(hidden_size)
                                / math.sqrt(hidden_size))
            h2op_pointers.append(h2op_pointers[-1] + hidden_size)
            h2op_indices.append(np.arange(h2op_col_start, h2op_col_end))

        h2op_bias.append(np.random.randn(full_output_size))

        new_op_start = op_new_unit[-1]
        new_op_out_ends = new_op_start + output_size
        new_op_end = op_new_unit[-1] + full_output_size
        out_map += list(range(new_op_start, new_op_out_ends))
        pred_map += list(range(new_op_out_ends, new_op_end))

        h2op_col_start = h2op_col_end
        op_new_unit.append(new_op_end)

    i2h_bias = np.concatenate(i2h_bias, axis=0)
    i2h_indices = np.concatenate(i2h_indices, axis=0).astype(np.int32)
    i2h_weights = np.concatenate(i2h_weights, axis=0)
    i2h_pointers = np.array(i2h_pointers, dtype=np.int32)

    h2op_bias = np.concatenate(h2op_bias, axis=0)
    h2op_indices = np.concatenate(h2op_indices, axis=0).astype(np.int32)
    h2op_weights = np.concatenate(h2op_weights, axis=0)
    h2op_pointers = np.array(h2op_pointers, dtype=np.int32)

    return (i2h_pointers, i2h_indices, i2h_weights, i2h_bias), \
        (h2op_pointers, h2op_indices, h2op_weights, h2op_bias), \
        (input_map, der_map, int_map, err_map, hid_map,
         hid_append_map, out_map, pred_map), \
        input_new_unit, op_new_unit, hid_new_unit


def tracker_weight_initialize(structure,
                              output_sizes, heatmap_size):
    """
    A function that creates arrays necessary for initializing the weights
    and biases of the tracker. Each layer will have a MLP which
    calculates a heat map of the size specified in heatmap_size
    :param structure: A list of layer sizes of the PVM, where the shape of the
    first layer is element 0, the second layer in element 1, etc.
    The elements of the list can be tuples, lists (of length 1 or 2) or
    integers. If an integer or a tuple or list with a single element it
    will specify that the layer is a square grid which edge size equal to the
    value. If the element is a tuple or list like (x, y) or [x, y] it will
    correspond to a grid of horizontal size x and vertical size y.

    :param output_sizes: A list of output sizes for each layer it must be the
    same length as structure.

    :param heatmap_size: The size of the heatmap for all layers.

    :return tracker_pointers: A numpy array of the pointers associated with
    the weights of the tracker (CSR matrix)

    :return tracker_indices: A numpy array of the column indices associated
    with the weights of the tracker (CSR matrix)

    :return tracker_weights: A numpy array of the values associated with the
    with the weights of the tracker (CSR matrix)

    :return tracker_bias: A numpy array of the biases associated with the
    biases of the tracker.
    """
    N_layers = len(structure)

    if N_layers != len(output_sizes):
        raise ValueError("must specify the output sizes such that they"
                         + " match the number of layers")

    tracker_bias = []
    tracker_weights = []
    tracker_pointers = [0]
    tracker_indices = []

    tracker_col_start = 0
    tracker_new_layer = [0]
    for layer in range(N_layers):
        L_x, L_y = shape_check(structure[layer])

        output_size = output_sizes[layer]

        tracker_input_size = L_x * L_y * output_size

        tracker_col_end = tracker_col_start + tracker_input_size
        for row in range(heatmap_size):
            # random Gaussian variables with
            # 1/tracker_input_size variance
            tracker_weights.append(np.random.randn(tracker_input_size)
                                   / math.sqrt(tracker_input_size))
            tracker_pointers.append(tracker_pointers[-1]
                                    + tracker_input_size)
            tracker_indices.append(np.arange(tracker_col_start,
                                             tracker_col_end))

        tracker_bias.append(np.random.randn(heatmap_size))

        tracker_col_start = tracker_col_end
        tracker_new_layer.append(tracker_col_start)

    tracker_bias = np.concatenate(tracker_bias, axis=0)
    tracker_indices = np.concatenate(tracker_indices, axis=0).astype(np.int32)
    tracker_weights = np.concatenate(tracker_weights, axis=0)
    tracker_pointers = np.array(tracker_pointers, dtype=np.int32)

    return tracker_pointers, tracker_indices, tracker_weights, tracker_bias,\
        tracker_new_layer


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


if __name__ == "__main__":
    from RectangularGridConstructor import make_connections
    test_connect_dict = make_connections([(16, 16), [8, 8], (4), 3, [2], 1],
                                         6 * 6 * 3, 49,
                                         [1, 2 * 2, 4 * 4, 8 * 8, 6 * 6, 16 * 16])
    outs = weight_initialize(test_connect_dict)


    def test_no_overlap_in_mapping(outs, i):
        for idx in outs[2][i]:
            if outs[2][4][idx] != -1:
                print('We have a problem: ', idx)
        print('Done.')


    for i in range(4):
        test_no_overlap_in_mapping(outs, i)

    print(ptr_to_row(np.array([0, 0, 2, 3, 4])))
