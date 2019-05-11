from __future__ import absolute_import, print_function, division
from builtins import *
import numpy as np
import pandas as pd
import h5py
from itertools import product
from PIL import Image
import os
from RectangularGridConstructor import shape_check


def flatten_image(img_arr, input_shape):
    """
    A helper function for flattening rectangular images for the input layer
    of the PVM
    :param img_arr: An array representation of the image to flatten so
    that we get input_shape sized rectangles going to the PVM units in the
    first layer.
    :param input_shape: The shape of the input (in pixels) as a tuple going to
    a single PVM_unit in the vertical and horizontal direction.
    :return flat_arr: The flattened representation of the original array
    in img_arr.
    """
    # don't change the data type
    datatype = img_arr.dtype
    L_y, L_x, n_color = img_arr.shape
    l_y, l_x = input_shape
    block_size = l_x * l_y * n_color

    x_div = L_x / l_x
    y_div = L_y / l_y

    # initialize flat array without changing the data type
    flat_arr = np.zeros(L_x * L_y * n_color, dtype=datatype)
    if x_div % 1 != 0:
        raise ValueError('Horizontal input shape does not cleanly divide'
                         + ' horizontal edge. Change the first element '
                         + 'of input_shape.')
    if y_div % 1 != 0:
        raise ValueError('Vertical input shape does not cleanly divide'
                         + ' vertical edge. Change the second element of '
                         + 'of input_shape.')

    x_div = int(x_div)
    y_div = int(y_div)

    # for y in range(y_div):
    #     for x in range(x_div):
    #         flat_arr[block_size * (x + y * x_div):
    #                  block_size * (x + y * x_div + 1)] = \
    #             img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
    #                     :].flatten()
    # for y, x in product(range(y_div), range(x_div)):
    for x, y in product(range(x_div), range(y_div)):
        flat_arr[block_size * (x + y * x_div):
                 block_size * (x + y * x_div + 1)] = \
            img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
                    :].flatten()
    return flat_arr


def unflatten_image(flat_arr, image_shape, input_shape):
    """
    A helper function that reverses the action of flatten_image.
    :param flat_arr: The flattened representation of the original array
    in img_arr.
    :param image_shape: The shape of the original image as a tuple in the form
    (vertical number of pixels, horizontal number of pixels, number of colors)
    this is the same as using ndarray.shape
    :param input_shape: A tuple, the shape of the input going to a single
    PVM_unit in the vertical and horizontal direction.
    :return img_arr: An array representation of the image to reverse the
    flattening done in flatten_image.
    """
    # don't change the data type
    datatype = flat_arr.dtype
    L_y, L_x, n_color = image_shape
    l_y, l_x = input_shape
    block_size = l_x * l_y * n_color

    x_div = L_x / l_x
    y_div = L_y / l_y

    # initialize flat array with changing the data type
    img_arr = np.zeros((L_y, L_x, n_color), dtype=datatype)
    if x_div % 1 != 0:
        raise ValueError('Horizontal input shape does not cleanly divide'
                         + ' horizontal edge. Change the first element '
                         + 'of input_shape.')
    if y_div % 1 != 0:
        raise ValueError('Vertical input shape does not cleanly divide'
                         + 'vertical edge. Change the second element of '
                         + 'of input_shape.')

    x_div = int(x_div)
    y_div = int(y_div)

    # for y in range(y_div):
    #     for x in range(x_div):
    #         img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
    #             flat_arr[block_size * (x + y * x_div):
    #                      block_size * (x + y * x_div + 1)].reshape((l_y,
    #                                                                 l_x,
    #                                                                 n_color))

    for y, x in product(range(y_div), range(x_div)):
        img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
            flat_arr[block_size * (x + y * x_div):
                     block_size * (x + y * x_div + 1)].reshape((l_y,
                                                                l_x,
                                                                n_color))
    return img_arr


def reformat_tracker_data(datapath, new_size, input_shape, img_dir='img',
                          groundtruth_rect='groundtruth_rect.0.txt', n_color=3):
    """
    Reads data in the with the directory structure:
    datapath
        |
        |---name_of_dataset_directory1
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |---groundtruct_rect
        |
        |---name_of_dataset_directory2
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |---groundtruct_rect
        .
        .
        .

    in each folder with the image data needs to be ordered in time with the
    earliest at the top and latest at the bottom

    :param datapath: A string with the name of a directory containing the
    data.
    :param new_size: A tuple with new sizes of the vertical and horizontal
    dimensions of the image.
    :param input_shape: A tuple, the shape of the input going to a single
    PVM_unit in the vertical and horizontal direction.
    :param img_dir: It defaults to 'img', a string with the common name of the
    directory.
    :param groundtruth_rect: It defaults to 'groundtruth_rect.0.txt' a string
    with the name of a text file with x (horizontal) pixel position and y
    (vertical) pixel position of the upper left corner of the bounding box
    in the first two columns and the next two columns and the respective
    widths of the bounding boxes in the next two columns.
    :param n_color: The number of colors used in the format of the image it
    defaults to 3 for RGB
    :return datadict: A dictionary with the data set directory's names as keys
    and a numpy array with each frame of the video along a row.
    :return groundtruthdict: A dictionary with pandas dataframes containing
    the pixel position of the upper left hand corner of the ground truth
    bounding box and the bounding box widths just like in the file with the
    name in groundtruth_rect.
    """
    rectlin = lambda z: z if z > 0 else 0
    dirlist = os.listdir(datapath)
    new_size_y, new_size_x = new_size

    dim = (new_size_y, new_size_x, n_color)
    tot_size = new_size_x * new_size_y * n_color
    flat_idx = flatten_image(np.arange(tot_size).reshape(dim),
                             input_shape)

    datadict = {}
    groundtruthdict = {}
    for dir_ in dirlist:

        df = pd.read_csv('/'.join([datapath, dir_, groundtruth_rect]),
                         header=None)

        imgpath = datapath + '/'.join([dir_, img_dir + '/'])
        imglist = sorted(os.listdir(imgpath))
        flat_img_list = []
        pos_width_list = []
        for idx, img in enumerate(imglist):
            img_open = Image.open(datapath + '/'.join([dir_, img_dir, img]))
            L_x, L_y = img_open.size
            x, x_w = (new_size_x / L_x) * df[[0, 2]].iloc[idx].apply(rectlin)
            y, y_w = (new_size_y / L_y) * df[[1, 3]].iloc[idx].apply(rectlin)
            img_resize = img_open.resize((new_size_x, new_size_y),  # (width, height)
                                         Image.ANTIALIAS)
            img_arr = np.array(img_resize)
            flat_img = img_arr.ravel()[flat_idx]
            flat_img_list.append(flat_img)
            pos_width_list.append([x, y, x_w, y_w])
        # each row of the array is an image and since it is
        # sorted I should have an easy time with the training
        groundtruthdf = pd.DataFrame(pos_width_list, columns=['x', 'y',
                                                              'x_w', 'y_w'])
        datadict[dir_] = np.array(flat_img_list)
        groundtruthdict[dir_] = groundtruthdf

    return datadict, groundtruthdict


def reformat_raw_data(datapath, new_size, img_dir='img',
                      groundtruth_rect='groundtruth_rect.0.txt'):
    """
    Reads data in the with the directory structure:
    datapath
        |
        |---name_of_dataset_directory1
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |---groundtruct_rect
        |
        |---name_of_dataset_directory2
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |---groundtruct_rect
        .
        .
        .

    in each folder with the image data needs to be ordered in time with the
    earliest at the top and latest at the bottom

    :param datapath: A string with the name of a directory containing the
    data.
    :param new_size: A tuple with new sizes of the vertical and horizontal
    dimensions of the image.
    :param img_dir: It defaults to 'img', a string with the common name of the
    directory.
    :param groundtruth_rect: It defaults to 'groundtruth_rect.0.txt' a string
    with the name of a text file with x (horizontal) pixel position and y
    (vertical) pixel position of the upper left corner of the bounding box
    in the first two columns and the next two columns and the respective
    widths of the bounding boxes in the next two columns.
    :param n_color: The number of colors used in the format of the image it
    defaults to 3 for RGB
    :return datadict: A dictionary with the data set directory's names as keys
    and a numpy array with each unaltered frame of the video along a row.
    Use datadict['name_of_key'][row, ...] to get a frame.
    :return groundtruthdict: A dictionary with pandas dataframes containing
    the pixel position of the upper left hand corner of the ground truth
    bounding box and the bounding box widths just like in the file with the
    name in groundtruth_rect.
    """
    rectlin = lambda z: z if z > 0 else 0
    dirlist = os.listdir(datapath)
    new_size_y, new_size_x = new_size

    datadict = {}
    groundtruthdict = {}
    for dir_ in dirlist:

        df = pd.read_csv('/'.join([datapath, dir_, groundtruth_rect]),
                         header=None)

        imgpath = datapath + '/'.join([dir_, img_dir + '/'])
        imglist = sorted(os.listdir(imgpath))
        img_list = []
        pos_width_list = []
        for idx, img in enumerate(imglist):
            img_open = Image.open(datapath + '/'.join([dir_, img_dir, img]))
            L_x, L_y = img_open.size
            x, x_w = (new_size_x / L_x) * df[[0, 2]].iloc[idx].apply(rectlin)
            y, y_w = (new_size_y / L_y) * df[[1, 3]].iloc[idx].apply(rectlin)
            img_resize = img_open.resize((new_size_x, new_size_y),  # (width, height)
                                         Image.ANTIALIAS)
            img_arr = np.array(img_resize)
            img_list.append(img_arr)
            pos_width_list.append([x, y, x_w, y_w])
        # each row of the array is an image and since it is
        # sorted I should have an easy time with the training
        groundtruthdf = pd.DataFrame(pos_width_list, columns=['x', 'y',
                                                              'x_w', 'y_w'])
        datadict[dir_] = np.array(img_list)
        groundtruthdict[dir_] = groundtruthdf

    return datadict, groundtruthdict


def save_tracker_data_to(save_dir, datadict, groundtruthdict):
    """
    This will take the output of reformat_tracker_data as save the values
    as numpy arrays and a csv to the directory specified. The directory will
    be created if it does not exist.
    :param save_dir: A string with the path name of the directory example:
    '/./data/green_ball_formatted_data' or '/./green_ball_formatted_data/'.
    The directory does not need to exist, it will be created.
    :param datadict: A dictionary with the data set directory's names as keys
    and a numpy array with each frame of the video along a row. (See the
    documentation for reformat_tracker_data)
    :param groundtruthdict: A pandas dataframe containing the pixel position
    of the upper left hand corner of the ground truth bounding box and the
    bounding box widths just like in the file with the name in
    groundtruth_rect. (See the documentation for reformat_tracker_data)
    """
    if len(datadict) != len(groundtruthdict):
        raise ValueError('Lengths of the two dictionaries do not match')

    if save_dir[-1] != '/':
        outpath, root = save_dir.rsplit('/', 1)
    else:
        save_dir = save_dir[:-1]
        outpath, root = save_dir.rsplit('/', 1)
    if root not in os.listdir(outpath):
        os.mkdir(save_dir)

    for key, array in datadict.items():
        filename = save_dir + '/' + key
        np.save(filename, array)
        groundtruthdict[key].to_csv(filename + '.csv', index=False)


def norm_and_heatmap(data_arr, df, heat_map_edges,
                     input_layer_dim, norm=255.):
    """
    A helper function that rescales the inputs by a normalization factor
    and creates a flattened heat map for training from a single element of
    the respective dictionaries output from the function
    reformat_tracker_data.
    :param data_arr: A numpy array contain the data of each frame in a row.
    It is should be an element of the dictionary produced by the function
    reformat_tracker_data.
    :param df: A dataframe with column labels 'x', 'y', 'x_w' and 'y_w'
    each row is for a frame. The values are the x and y position of the
    upper left corner of the bounding box and x_w and y_w are the respective
    widths.
    :param heat_map_edges: A tuple, list (of length 2 or 1) or an integer
    which specifies the dimension of the heat map (x then y) e.g.: (16, 16),
    or equivalently 16
    :param input_layer_dim: Shape of the input layer in pixels with the x
    dimension first then the y. It is of the same form as heat_map_edges.
    :param norm_factor: Default is 255. This should be chosen depending on
    your data so that the inputs to the PVM are between 0 and 1. If RBG 255
    does the trick.
    :return rescale_arr: Normalized value of the input data_arr it is
    data_arr / norm
    :return ground_truth_heat_map_list: a list whose elements are numpy arrays
    that are flattened representations of heat maps they are zero everywhere
    but the pixels of the interior of the bounding box where the values are
    one.
    """
    n_frames = len(df)
    if n_frames != data_arr.shape[0]:
        raise ValueError('Number of rows do not match')
    heat_map_edge_x, heat_map_edge_y = shape_check(heat_map_edges)
    edge_n_pixels_x, edge_n_pixels_y = shape_check(input_layer_dim)
    heat_map_size = heat_map_edge_x * heat_map_edge_y

    rescale_arr = data_arr / norm
    ground_truth_heat_map_list = []
    for idx in range(n_frames):
        rescale_factor_x = heat_map_edge_x / edge_n_pixels_x
        rescale_factor_y = heat_map_edge_y / edge_n_pixels_y
        x, x_w = rescale_factor_x * df.iloc[idx][['x', 'x_w']]
        y, y_w = rescale_factor_y * df.iloc[idx][['y', 'y_w']]
        x = int(x)
        y = int(y)
        x_w = int(x_w)
        y_w = int(y_w)
        ground_truth_heat_map = np.zeros((heat_map_edge_y, heat_map_edge_x))

        # changed for consistency and to be more easily understood
        # x is horizontal corresponding to columns and y is vertical
        # corresponding to rows
        ground_truth_heat_map[y:y + y_w, x:x + x_w] = 1
        ground_truth_heat_map = ground_truth_heat_map.reshape(heat_map_size)
        ground_truth_heat_map_list.append(ground_truth_heat_map)

    return rescale_arr, ground_truth_heat_map_list


def fractal_flatten_image(img_arr, first_rec=True, corner=""):
    """
    A helper function for flattening rectangular images for the input layer
    of the PVM
    :param img_arr: An array representation of the image to flatten so
    that we get input_shape sized rectangles going to the PVM units in the
    first layer.
    :param input_shape: The shape of the input (in pixels) as a tuple going to
    a single PVM_unit in the horizontal and vertical direction.
    :return flat_arr: The flattened representation of the original array
    in img_arr.
    """
    # don't change the data type
    datatype = img_arr.dtype
    L_y, L_x, n_color = img_arr.shape
    if L_y != L_x:
        raise ValueError('Not a square matrix')
    elif np.log2(L_x) % 1 != 0:
        raise ValueError('Image size is not a power of 2 along horizontal edge.')

    if first_rec:
        x_div, y_div = 4, 4
    else:
        x_div, y_div = 2, 2

    l_x, l_y = L_x // x_div, L_y // y_div

    block_size = l_x * l_y * n_color

    # initialize flat array without changing the data type
    flat_arr = np.zeros(L_x * L_y * n_color, dtype=datatype)

    if first_rec and (L_x > 4):
        for y, x in product(range(y_div), range(x_div)):
            if x == 0 or y == 0 or x == 3 or y == 3:
                flat_arr[block_size * (x + y * x_div):
                         block_size * (x + y * x_div + 1)] = \
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
                    :].flatten()
            elif x == 1 and y == 1:
                flat_arr[block_size * (x + y * x_div):
                         block_size * (x + y * x_div + 1)] = \
                    fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                          x * l_x:(x + 1) * l_x,
                                          :], first_rec=False, corner="SE")
            elif x == 1 and y == 2:
                flat_arr[block_size * (x + y * x_div):
                         block_size * (x + y * x_div + 1)] = \
                    fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                          x * l_x:(x + 1) * l_x,
                                          :], first_rec=False, corner="NE")
            elif x == 2 and y == 1:
                flat_arr[block_size * (x + y * x_div):
                         block_size * (x + y * x_div + 1)] = \
                    fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                          x * l_x:(x + 1) * l_x,
                                          :], first_rec=False, corner="SW")
            elif x == 2 and y == 2:
                flat_arr[block_size * (x + y * x_div):
                         block_size * (x + y * x_div + 1)] = \
                    fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                          x * l_x:(x + 1) * l_x,
                                          :], first_rec=False, corner="NW")
    elif not first_rec and (L_x > 2):
        for y, x in product(range(y_div), range(x_div)):
            if corner == "SE":
                if x == 0 or (x == 1 and y == 0):
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
                        :].flatten()
                else:
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                              x * l_x:(x + 1) * l_x,
                                              :], first_rec=False, corner="SE")
            elif corner == "NE":
                if x == 0 or (x == 1 and y == 1):
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
                        :].flatten()
                else:
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                              x * l_x:(x + 1) * l_x,
                                              :], first_rec=False, corner="NE")
            elif corner == "SW":
                if x == 1 or (x == 0 and y == 0):
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
                        :].flatten()
                else:
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                              x * l_x:(x + 1) * l_x,
                                              :], first_rec=False, corner="SW")
            elif corner == "NW":
                if x == 1 or (x == 0 and y == 1):
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x,
                        :].flatten()
                else:
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)] = \
                        fractal_flatten_image(img_arr[y * l_y:(y + 1) * l_y,
                                              x * l_x:(x + 1) * l_x,
                                              :], first_rec=False, corner="NW")
    else:
        flat_arr = img_arr.flatten()
    return flat_arr


def fractal_unflatten_image(flat_arr, image_shape, first_rec=True, corner=""):
    """
    A helper function for flattening rectangular images for the input layer
    of the PVM
    :param img_arr: An array representation of the image to flatten so
    that we get input_shape sized rectangles going to the PVM units in the
    first layer.
    :param input_shape: The shape of the input (in pixels) as a tuple going to
    a single PVM_unit in the horizontal and vertical direction.
    :return flat_arr: The flattened representation of the original array
    in img_arr.
    """
    # don't change the data type
    datatype = flat_arr.dtype
    L_y, L_x, n_color = image_shape

    img_arr = np.zeros(image_shape, dtype=datatype)
    if L_y != L_x:
        raise ValueError('Not a square matrix')
    elif np.log2(L_x) % 1 != 0:
        raise ValueError('Image size is not a power of 2 along horizontal edge.')

    if first_rec:
        x_div, y_div = 4, 4
    else:
        x_div, y_div = 2, 2

    l_x, l_y = L_x // x_div, L_y // y_div

    block_size = l_x * l_y * n_color

    # initialize flat array without changing the data type
    img_arr = np.zeros((L_x, L_y, n_color), dtype=datatype)

    if first_rec and (L_x > 4):
        for y, x in product(range(y_div), range(x_div)):
            if x == 0 or y == 0 or x == 3 or y == 3:
                img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                    flat_arr[block_size * (x + y * x_div):
                             block_size * (x + y * x_div + 1)].reshape((l_y,
                                                                        l_x,
                                                                        n_color))
            elif x == 1 and y == 1:
                img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                    fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                     block_size * (x + y * x_div + 1)],
                                            (l_y, l_x, n_color),
                                            first_rec=False, corner="SE")
            elif x == 1 and y == 2:
                img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                    fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                     block_size * (x + y * x_div + 1)],
                                            (l_y, l_x, n_color),
                                            first_rec=False, corner="NE")
            elif x == 2 and y == 1:
                img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                    fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                     block_size * (x + y * x_div + 1)],
                                            (l_y, l_x, n_color),
                                            first_rec=False, corner="SW")
            elif x == 2 and y == 2:
                img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                    fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                     block_size * (x + y * x_div + 1)],
                                            (l_y, l_x, n_color),
                                            first_rec=False, corner="NW")
    elif not first_rec and (L_x > 2):
        for y, x in product(range(y_div), range(x_div)):
            if corner == "SE":
                if x == 0 or (x == 1 and y == 0):
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        flat_arr[block_size * (x + y * x_div):
                                 block_size * (x + y * x_div + 1)].reshape((l_y,
                                                                            l_x,
                                                                            n_color))
                else:
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                         block_size * (x + y * x_div + 1)],
                                                (l_y, l_x, n_color),
                                                first_rec=False, corner="SE")
            elif corner == "NE":
                if x == 0 or (x == 1 and y == 1):
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        flat_arr[block_size * (x + y * x_div):
                                 block_size * (x + y * x_div + 1)].reshape((l_y,
                                                                            l_x,
                                                                            n_color))
                else:
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                         block_size * (x + y * x_div + 1)],
                                                (l_y, l_x, n_color),
                                                first_rec=False, corner="NE")
            elif corner == "SW":
                if x == 1 or (x == 0 and y == 0):
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        flat_arr[block_size * (x + y * x_div):
                                 block_size * (x + y * x_div + 1)].reshape((l_y,
                                                                            l_x,
                                                                            n_color))
                else:
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                         block_size * (x + y * x_div + 1)],
                                                (l_y, l_x, n_color),
                                                first_rec=False, corner="SW")
            elif corner == "NW":
                if x == 1 or (x == 0 and y == 1):
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        flat_arr[block_size * (x + y * x_div):
                                 block_size * (x + y * x_div + 1)].reshape((l_y,
                                                                            l_x,
                                                                            n_color))
                else:
                    img_arr[y * l_y:(y + 1) * l_y, x * l_x:(x + 1) * l_x, :] = \
                        fractal_unflatten_image(flat_arr[block_size * (x + y * x_div):
                                                         block_size * (x + y * x_div + 1)],
                                                (l_y, l_x, n_color),
                                                first_rec=False, corner="NW")
    else:
        img_arr = flat_arr.reshape((L_y, L_x, n_color))
    return img_arr


def hdf5_raw_data(datapath, filename, new_size, img_dir='img'):
    """
    Reads data in the with the directory structure:
    datapath
        |
        |---name_of_dataset_directory1
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |
        |
        |---name_of_dataset_directory2
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |
        .
        .
        .

    in each folder with the image data needs to be ordered in time with the
    earliest at the top and latest at the bottom

    :param datapath: A string with the name of a directory containing the
    data.
    :param filename: A string containing the path and filename of the hdf5 file
    that is being written the .hdf5 extension is not needed
    :param new_size: A tuple with new sizes of the vertical and horizontal
    dimensions of the image.
    :param img_dir: It defaults to 'img', a string with the common name of the
    directory.
    :param n_color: The number of colors used in the format of the image it
    defaults to 3 for RGB
    """
    dirlist = os.listdir(datapath)
    new_size_y, new_size_x = new_size

    with h5py.File(filename, 'w') as f:
        for dir_ in dirlist:

            imgpath = datapath + '/'.join([dir_, img_dir + '/'])
            imglist = sorted(os.listdir(imgpath))
            img_list = []
            for img in imglist:
                img_open = Image.open(datapath + '/'.join([dir_, img_dir, img]))
                img_resize = img_open.resize((new_size_x, new_size_y),  # (width, height)
                                             Image.ANTIALIAS)
                img_arr = np.array(img_resize)
                img_list.append(img_arr)
            # each row of the array is an image and since it is
            # sorted I should have an easy time with the training
            f[dir_] = np.array(img_list)


def hdf5_append_position_data(datapath, filename, new_size, img_dir='img'):
    """
    Reads data in the with the directory structure:
    datapath
        |
        |---name_of_dataset_directory1
        |       |---output.txt
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |
        |
        |---name_of_dataset_directory2
        |       |---output.txt
        |       |---img_dir
        |       |       |---img0001
        |       |       |---img0002
        |       |       |---img0003
        |       |       |---img0004
        |       |       .
        |       |       .
        |       |       .
        |       |
        |       |
        .
        .
        .

    in each folder with the image data needs to be ordered in time with the
    earliest at the top and latest at the bottom

    :param datapath: A string with the name of a directory containing the
    data.
    :param filename: A string containing the path and filename of the hdf5 file
    that is being written the .hdf5 extension is not needed
    :param new_size: A tuple with new sizes of the vertical and horizontal
    dimensions of the image.
    :param img_dir: It defaults to 'img', a string with the common name of the
    directory.
    :param n_color: The number of colors used in the format of the image it
    defaults to 3 for RGB
    """
    dirlist = os.listdir(datapath)
    new_size_y, new_size_x = new_size

    with h5py.File(filename, 'w') as f:
        for dir_ in dirlist:
            df = pd.read_csv(datapath + '/'.join([dir_, 'output.txt']), header=None,
                             names=['time', 'pan', 'tilt']
                             )
            df['time'] = pd.to_datetime(df['time'])

            df['pan'] = df['pan'] / 1023.
            df['tilt'] = df['tilt'] / 1023.
            df['dt'] = df['time'].diff()
            df['dpan'] = (df['pan'].diff() + 1.) / 2.
            df['dtilt'] = (df['tilt'].diff() + 1.) / 2.

            # df = df.drop(0)

            imgpath = datapath + '/'.join([dir_, img_dir + '/'])
            imglist = sorted(os.listdir(imgpath))
            img_list = []
            name_list = []
            for img in imglist[1:]:
                time = img.split('.png')[0]
                img_open = Image.open(datapath + '/'.join([dir_, img_dir, img]))
                img_resize = img_open.resize((new_size_x, new_size_y),  # (width, height)
                                             Image.ANTIALIAS)
                img_arr = np.array(img_resize)
                img_list.append(img_arr)
                name_list.append(time)

            df['dpan/dt (1/ms)'] = df['dpan'] * np.timedelta64(1, 'ms') / df['dt']
            df['dtilt/dt (1/ms)'] = df['dtilt'] * np.timedelta64(1, 'ms') / df['dt']


            # each row of the array is an image and since it is
            # sorted I should have an easy time with the training
            grp = f.create_group(dir_)
            grp.create_dataset('images', data=np.array(img_list))
            grp.create_dataset('position', data=df[['pan',
                                                    'tilt',
                                                    'dpan',
                                                    'dtilt']])


if __name__=="__main__":
    import hypothesis
    import hypothesis.extra.numpy as exnp

    ints = hypothesis.strategies.integers


    @hypothesis.given(exnp.arrays(np.uint8, (10, 30, 3),
                                  elements=ints(0, 32)))
    def test_flatten_inverts(img):
        assert (unflatten_image(flatten_image(img, (6, 5)),
                                img.shape, (6, 5)) == img).all()


    test_flatten_inverts()