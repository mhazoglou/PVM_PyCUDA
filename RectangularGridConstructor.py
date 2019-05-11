from __future__ import absolute_import, print_function, division
from builtins import *
from collections import OrderedDict


def shape_check(shape):
    """
    As a structure list can be a list of integers
    or tuples of size two or a mixture of the two types
    this will return the two dimensions depending on the
    input format
    """
    if isinstance(shape, int):
        L_x = shape
        L_y = shape
    elif len(shape) == 2:
        L_x, L_y = shape
    elif len(shape) == 1:
        L_x = shape[0]
        L_y = shape[0]
    else:
        raise ValueError('Not a valid entry must be a single integer\n' +
                         'or a tuple of length one or two.')
    return L_x, L_y


def feeds2(structure, layer, x, y):
    """
    Specifies which PVM-units the location layer, x, y
    feeds to in the next layer assuming that the structure is
    rectangular and doesn't differ along an edge by more than 1 unit
    or by double the dimension of units
    :param structure: A list whose elements specify the dimensions of a
    rectangular grid either as tuples/lists of length 1 or 2 or integers
    for square grids. The horizontal dimension comes first
    e.g.: [16, 8, 4, 3, 2, 1] or equivalently
    [(16, 16), (8, 8), (4), 2, 1]
    :param layer: An integer specifying the layer that the unit of
    interest belongs to. Counting starts from 0.
    :param x: Horizontal position of the PVM unit. Counting starts from 0.
    Data type is an integer.
    :param y: Vertical position of the PVM unit. Counting starts from 0.
    Data type is an integer.
    """
    assert layer < len(structure) - 1
    
    def find_edge_pos_upper_layer(pos, L, L_n):
        if (L - L_n) == 1:
            if pos == 0:
                p_list = [0]
            elif pos == L-1:
                p_list = [L_n-1]
            else:
                p_list = [pos-1, pos]
        elif L/L_n % 2 == 0:
            pos_ov2 = pos / 2
            p_list = [int(pos_ov2 - pos_ov2 % 1)]
        elif L == L_n:
            p_list = [pos]
        else:
            raise ValueError(
                'Underlying assumption about layer structure violated.\n' +
                'Edge sizes must differ by at most 1 or a multiple of 2')
        return p_list
    
    shape = structure[layer]
    L_x, L_y = shape_check(shape)

    # shape of the layer above the current
    shape_next = structure[layer+1]
    L_x_n, L_y_n = shape_check(shape_next)
    
    p_x_list = find_edge_pos_upper_layer(x, L_x, L_x_n)
    p_y_list = find_edge_pos_upper_layer(y, L_y, L_y_n)
    
    feed2List = []
    for p_x in p_x_list:
        feed2List += [(p_x, p_y) for p_y in p_y_list]
        
    return feed2List


def fedfrom(structure, layer, x, y):
    """
    Specifies which PVM-units the location layer, x, y is fed
    from in the layer directly below assuming that the structure is
    rectangular and doesn't differ along an edge by more than 1 unit
    or by double the dimension of units
    :param structure: A list whose elements specify the dimensions of a
    rectangular grid either as tuples/lists of length 1 or 2 or integers
    for square grids. The horizontal dimension comes first
    e.g.: [16, 8, 4, 3, 2, 1] or equivalently
    [(16, 16), (8, 8), (4), 2, 1]
    :param layer: An integer specifying the layer that the unit of
    interest belongs to. Counting starts from 0.
    :param x: Horizontal position of the PVM unit. Counting starts from 0.
    Data type is an integer.
    :param y: Vertical position of the PVM unit. Counting starts from 0.
    Data type is an integer.
    """
    assert layer > 0
    
    def find_edge_pos_lower_layer(pos, L, L_b):
        if (L_b - L) == 1:
            p_list = [pos, pos + 1]
        elif L_b/L % 2 == 0:
            pos_doub = 2*pos
            p_list = [pos_doub, pos_doub+1]
        elif L == L_b:
            p_list = [pos]
        else:
            raise ValueError(
                'Underlying assumption about layer structure violated.\n' +
                'Edge sizes must differ by at most 1 or a multiple of 2')
        return p_list
    
    shape = structure[layer]
    L_x, L_y = shape_check(shape)

    shape_b = structure[layer-1]
    L_x_b, L_y_b = shape_check(shape_b)
    
    p_x_list = find_edge_pos_lower_layer(x, L_x, L_x_b)
    p_y_list = find_edge_pos_lower_layer(y, L_y, L_y_b)
    
    fedfromList = []
    for p_x in p_x_list:
        fedfromList += [(p_x, p_y) for p_y in p_y_list]
        
    return fedfromList


def make_connections(structure, input_size, hidden_size, output_sizes,
                     context_from_top_0_0=True):
    """
    A helper function to make the dictionaries for characterizing
    the connections of a PVM hierarchy made of rectangular grid layers
    with lateral and superior context. Context from the top most
    layer of the x=0 and y=0 unit is also passed down to every unit
    if the context_from_top_0_0 is left on by default.

    :param structure: A list whose elements specify the dimensions of a
    rectangular grid either as tuples/lists of length 1 or 2 or integers
    for square grids from the bottom of the hierarchy to the top.
    The horizontal dimension comes first e.g.: [16, 8, 4, 3, 2, 1] or
    equivalently [(16, 16), (8, 8), (4), 2, 1]

    :param input_size: An integer representing the size of the input to that
    PVM unit in the input layer (layer 0, units with the label _0)

    :param hidden_size: An integer specifying the hidden size of all PVM
    units

    :param output_sizes: A list containing the integer sizes of the outputs
    of each unit in that layer. They are all the same in one layer.

    :param context_from_top_0_0: An optional boolean describing whether
    or not to include context from the top layer

    :return connect_dict: An ordered dictionary that
    maps a string representing a PVM unit to a tuple with

    (unit_count, input_size, hidden_size, output_size, fedfrom_list,
     latsup_list)

    unit_count is a numerical value associated with the key
    input_size is an integer representing the size of the input to that
    PVM unit in the input layer
    hidden_size is the same as the parameter hidden_size it is the hidden_size
    of each PVM unit
    output_size is an integer specifying the size of the output from that unit
    fedfrom_list is a list of the keys for the PVM units feeding into the unit
    latsup_list is a list of the keys that send their context to the PVM unit
    """
    N_layers = len(structure)
    
    if N_layers != len(output_sizes):
        raise ValueError("Must specify the output sizes such that" +
                         " they match the number of layers.")
    
    ret_name = lambda ls: ['_{0}_{1}_{2}'.format(l, xx_, yy_)
                           for l, xx_, yy_ in ls]
    
    connect_dict = OrderedDict()  # {}
    unit_count = 0
    for layer, shape in enumerate(structure):
        output_size = output_sizes[layer]
        L_x, L_y = shape_check(shape)
        if layer == 0:
            raw_input_size = input_size
        else:
            raw_input_size = 0
        for y in range(L_y):
            for x in range(L_x):
                name = '_{0}_{1}_{2}'.format(layer, x, y)
                
                nn_list = []
                if x == 0 and L_x > 1:
                    nn_list.append((layer, 1, y))
                elif x == L_x-1 and L_x > 1:
                    nn_list.append((layer, x-1, y))
                elif L_x == 1:
                    pass
                else:
                    nn_list += [(layer, x-1, y), (layer, x+1, y)]
                if y == 0 and L_y > 1:
                    nn_list.append((layer, x, 1))
                elif y == L_y-1 and L_y > 1:
                    nn_list.append((layer, x, y-1))
                elif L_y == 1:
                    pass
                else:
                    nn_list += [(layer, x, y-1), (layer, x, y+1)]
                
                try:
                    feeds2list = feeds2(structure, layer, x, y)
                except AssertionError:
                    feeds2list = []
                try:
                    fedfromlist = fedfrom(structure, layer, x, y)
                    fedfromlist = [(layer-1, x_, y_) 
                                   for x_, y_ in fedfromlist]
                except AssertionError:
                    fedfromlist = []
                
                if layer < N_layers-2:
                    if context_from_top_0_0:
                        latsuplist = nn_list + [(layer+1, x_, y_)
                                                for x_, y_ in feeds2list]\
                                                + [(N_layers-1, 0, 0)]
                    else:
                        latsuplist = nn_list + [(layer+1, x_, y_)
                                                for x_, y_ in feeds2list]
                elif layer == N_layers-2:
                    if context_from_top_0_0:
                        latsuplist = nn_list + [(N_layers-1, 0, 0)]
                    else:
                        latsuplist = nn_list
                else:
                    latsuplist = []  # None
                
                try:
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          ret_name(fedfromlist), 
                                          ret_name(latsuplist))
                except ValueError:
                    try:
                        connect_dict[name] = (unit_count,
                                              raw_input_size,
                                              hidden_size,
                                              output_size,
                                              fedfromlist, 
                                              ret_name(latsuplist))
                    except TypeError:
                        connect_dict[name] = (unit_count,
                                              raw_input_size,
                                              hidden_size,
                                              output_size,
                                              fedfromlist, 
                                              latsuplist)
                except TypeError:
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          ret_name(fedfromlist), 
                                          latsuplist)
                unit_count += 1
    return connect_dict


def break_stuff(connect_dict, break_unit_list, input_shape, new_hidden_size,
                new_output_size, div_shape=(1, 1), n_color=3):
    """
    For anything besides a default div_shape of (1, 1) you will need to
    provide your own mappings
    :param connect_dict:
    :param break_unit_list:
    :param input_shape:
    :param new_hidden_size:
    :param new_output_size:
    :param div_shape:
    :param n_color:
    :return:
    """
    output_connect_dict = OrderedDict()
    h, w = input_shape
    div_ver, div_hor = div_shape
    if (h % div_ver != 0) or (w % div_hor != 0):
        raise ValueError('Either the vertical or horizontal value of '
                         'div_shape is not a factor of the respective '
                         'value in input shape')
    n_div_hor = w // div_hor
    n_div_ver = h // div_ver
    unit_count = 0
    for unit in break_unit_list:
        _, should_be_zero, x_loc, y_loc = unit.split('_')
        if int(should_be_zero) != 0:
            raise ValueError('This function is only implemented for '
                             'the first layer. You cannot break higher '
                             'layers')
    for key, vals in connect_dict.items():
        count, input_size, hidden_size, output_size, fedfrom_list, \
            latsup_list = vals
        _, layer, key_X, key_Y = key.split('_')
        if key in break_unit_list:
            for x in range(n_div_hor):
                for y in range(n_div_ver):
                    if x == 0:
                        new_latsup_list = [key + '_1_{}'.format(y)]
                    elif x == n_div_hor - 1:
                        new_latsup_list = [key + '_{}_{}'.format(x - 1, y)]
                    else:
                        new_latsup_list = [key + '_{}_{}'.format(x - 1, y),
                                           key + '_{}_{}'.format(x + 1, y)]
                    if y == 0:
                        new_latsup_list.append(key + '_{}_1'.format(x))
                    elif y == n_div_ver - 1:
                        new_latsup_list.append(key +
                                               '_{}_{}'.format(x, y - 1))
                    else:
                        new_latsup_list += [key + '_{}_{}'.format(x, y - 1),
                                            key + '_{}_{}'.format(x, y + 1)]
                    for name in latsup_list:
                        _, name_layer, name_X, name_Y = name.split('_')
                        if name in break_unit_list:
                            if (int(key_X) < int(name_X)) and \
                                    x == n_div_hor - 1:
                                new_latsup_list.append(name +
                                                       '_0_{}'.format(y))
                            elif (int(key_X) > int(name_X)) and x == 0:
                                new_latsup_list.append(name +
                                                       '_{}_{}'.format(
                                                           n_div_hor - 1, y))

                            if (int(key_Y) < int(name_Y)) and \
                                    y == n_div_ver - 1:
                                new_latsup_list.append(name +
                                                       '_{}_0'.format(x))
                            elif (int(key_Y) > int(name_Y)) and y == 0:
                                new_latsup_list.append(name +
                                                       '_{}_{}'.format(
                                                           x, n_div_ver - 1))
                        else:
                            new_latsup_list.append(name)

                    new_fedfrom_list = []
                    for name in fedfrom_list:
                        if name in break_unit_list:
                            for x_ in range(n_div_hor):
                                for y_ in range(n_div_ver):
                                    new_fedfrom_list.append(name +
                                                            '_{}_{}'.format(
                                                                x_, y_))
                        else:
                            new_fedfrom_list.append(name)

                    new_key = key + '_{}_{}'.format(x, y)
                    output_connect_dict[new_key] = (unit_count,
                                                    div_ver * div_hor
                                                    * n_color,
                                                    new_hidden_size,
                                                    new_output_size,
                                                    new_fedfrom_list,
                                                    new_latsup_list)
                    unit_count += 1
        else:
            new_latsup_list = []
            for name in latsup_list:
                _, name_layer, name_X, name_Y = name.split('_')
                if name in break_unit_list:
                    if int(key_X) < int(name_X):
                        for y in range(n_div_ver):
                            new_latsup_list.append(name +
                                                   '_0_{}'.format(y))
                    elif int(key_X) > int(name_X):
                        for y in range(n_div_ver):
                            new_latsup_list.append(name +
                                                   '_{}_{}'.format(
                                                       n_div_hor - 1, y))

                    if int(key_Y) < int(name_Y):
                        for x in range(n_div_hor):
                            new_latsup_list.append(name +
                                                   '_{}_{}'.format(x, 0))
                    elif int(key_Y) > int(name_Y):
                        for x in range(n_div_hor):
                            new_latsup_list.append(name +
                                                   '_{}_{}'.format(
                                                       x, n_div_ver - 1))
                else:
                    new_latsup_list.append(name)

            new_fedfrom_list = []
            for name in fedfrom_list:
                if name in break_unit_list:
                    for x in range(n_div_hor):
                        for y in range(n_div_ver):
                            new_fedfrom_list.append(name +
                                                    '_{}_{}'.format(x, y))
                else:
                    new_fedfrom_list.append(name)

            output_connect_dict[key] = (unit_count,
                                        input_size,
                                        hidden_size,
                                        output_size,
                                        new_fedfrom_list,
                                        new_latsup_list)
            unit_count += 1
    return output_connect_dict


if __name__ == "__main__":
    test_connect_dict = make_connections([(16, 16), [8, 8], (4,), 3, [2], 1],
                                         108, 49,
                                         [1, 2*2, 4*4, 8*8, 6*6, 16*16])
    print(test_connect_dict)
