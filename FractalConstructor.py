from __future__ import absolute_import, print_function, division
from builtins import *
from collections import OrderedDict
import numpy as np
from itertools import product


def layer_fractal_connections(power, name):
    """_layer_ring_cardinality"""
    _, layer, ring, card_dir = name.split('_')

    layer = int(layer)
    ring = int(ring)

    if ring <= 0:
        raise ValueError('Incorrect value for ring. Must be greater than zero.')
    if power <= 0 or not isinstance(power, int):
        raise ValueError('Incorrect value for power. Must be an integer greater than zero.')
    if layer < 0:
        raise ValueError('Not a valid layer number. It must greater than or equal to zero.')

    if ring > 2 and ring < power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'WNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WNW'])]
        elif card_dir == 'NNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NNW'])]
        elif card_dir == 'NNE':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NNE'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ENE'])]
        elif card_dir == 'ENE':
            return ['_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ENE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE']),
                    '_'.join(['', str(layer), str(ring - 1), 'ENE'])]
        elif card_dir == 'ESE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ESE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE']),
                    '_'.join(['', str(layer), str(ring - 1), 'ESE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring), 'SSE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ESE']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSE'])]
        elif card_dir == 'SSE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'SSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SSE'])]
        elif card_dir == 'SSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'SSE']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SSW'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring), 'SSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WSW'])]
        elif card_dir == 'WSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'WNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW']),
                    '_'.join(['', str(layer), str(ring - 1), 'WSW'])]
        elif card_dir == 'WNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WNW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW']),
                    '_'.join(['', str(layer), str(ring - 1), 'WNW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring > 2 and ring == power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'WNW'])]
        elif card_dir == 'NNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'NNE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NNW'])]
        elif card_dir == 'NNE':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NNE'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring), 'NNE'])]
        elif card_dir == 'ENE':
            return ['_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE']),
                    '_'.join(['', str(layer), str(ring - 1), 'ENE'])]
        elif card_dir == 'ESE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE']),
                    '_'.join(['', str(layer), str(ring - 1), 'ESE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring), 'SSE'])]
        elif card_dir == 'SSE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'SSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SSE'])]
        elif card_dir == 'SSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'SSE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SSW'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring), 'SSW'])]
        elif card_dir == 'WSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'WNW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW']),
                    '_'.join(['', str(layer), str(ring - 1), 'WSW'])]
        elif card_dir == 'WNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW']),
                    '_'.join(['', str(layer), str(ring - 1), 'WNW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 2 and ring < power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'WNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WNW'])]
        elif card_dir == 'NNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW'])]
        elif card_dir == 'NNE':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ENE'])]
        elif card_dir == 'ENE':
            return ['_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ENE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE'])]
        elif card_dir == 'ESE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ESE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring), 'SSE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ESE']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSE'])]
        elif card_dir == 'SSE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'SSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE'])]
        elif card_dir == 'SSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'SSE']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring), 'SSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WSW]'])]
        elif card_dir == 'WSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'WNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW'])]
        elif card_dir == 'WNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WNW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 2 and ring == power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'WNW'])]
        elif card_dir == 'NNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'NNE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW'])]
        elif card_dir == 'NNE':
            return ['_'.join(['', str(layer), str(ring), 'NNW']),
                    '_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring), 'NNE'])]
        elif card_dir == 'ENE':
            return ['_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring - 1), 'NE'])]
        elif card_dir == 'ESE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'ENE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer), str(ring), 'ESE']),
                    '_'.join(['', str(layer), str(ring), 'SSE'])]
        elif card_dir == 'SSE':
            return ['_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring), 'SSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SE'])]
        elif card_dir == 'SSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'SSE']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring), 'SSW'])]
        elif card_dir == 'WSW':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'WNW']),
                    '_'.join(['', str(layer), str(ring - 1), 'SW'])]
        elif card_dir == 'WNW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'WSW']),
                    '_'.join(['', str(layer), str(ring - 1), 'NW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 1 and power > 1:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNW']),
                    '_'.join(['', str(layer), str(ring + 1), 'WNW'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring + 1), 'NNE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ENE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSE']),
                    '_'.join(['', str(layer), str(ring + 1), 'ESE'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'SE']),
                    '_'.join(['', str(layer), str(ring + 1), 'WSW']),
                    '_'.join(['', str(layer), str(ring + 1), 'SSW'])]
        else:
            raise ValueError('Cardinality is not a valid value only NW, NE, SE, SW available.')
    elif ring == 1 and power == 1:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer), str(ring), 'NE']),
                    '_'.join(['', str(layer), str(ring), 'SW'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'SE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer), str(ring), 'SW']),
                    '_'.join(['', str(layer), str(ring), 'NE'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer), str(ring), 'NW']),
                    '_'.join(['', str(layer), str(ring), 'SE'])]
        else:
            raise ValueError('Cardinality is not a valid value only NW, NE, SE, SW available.')
    else:
        raise ValueError("I can't believe you've done this. " +
                         "Something went wrong with string in 'name'.")


def fedfrom_fractal_connections(power, name):
    """_layer_ring_cardinality"""
    _, layer, ring, card_dir = name.split('_')

    layer = int(layer)
    ring = int(ring)

    if ring <= 0:
        raise ValueError('Incorrect value for ring. Must be greater than zero.')
    if power <= 0 or not isinstance(power, int):
        raise ValueError('Incorrect value for power. Must be an integer greater than zero.')
    if layer < 0:
        raise ValueError('Not a valid layer number. It must greater than zero.')
    elif layer == 0:
        return []

    card_dir_list = ['NW', 'NNW', 'NNE', 'NE', 'ENE', 'ESE',
                     'SE', 'SSE', 'SSW', 'SW', 'WSW', 'WNW']

    if ring > 1 and ring <= power:
        if card_dir in card_dir_list:
            return ['_'.join(['', str(layer - 1), str(ring + 1), card_dir]),
                    '_'.join(['', str(layer - 1), str(ring), card_dir])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 1 and ring <= power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer - 1), str(ring), 'NW']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'NW']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'NNW']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'WNW'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer - 1), str(ring), 'NE']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'NE']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'NNE']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'ENE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer - 1), str(ring), 'SE']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'SE']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'SSE']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'ESE'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer - 1), str(ring), 'SW']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'SW']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'WSW']),
                    '_'.join(['', str(layer - 1), str(ring + 1), 'SSW'])]
        else:
            raise ValueError('Cardinality is not a valid value only NW, NE, SE, SW available.')
    else:
        raise ValueError("I can't believe you've done this. " +
                         "Something went wrong with string in 'name'.")


def feeds2_fractal_connections(power, name):
    _, layer, ring, card_dir = name.split('_')

    layer = int(layer)
    ring = int(ring)

    if ring <= 0:
        raise ValueError('Incorrect value for ring. Must be greater than zero.')
    if power <= 0 or not isinstance(power, int):
        raise ValueError('Incorrect value for power. Must be an integer greater than zero.')
    if layer < 0:
        raise ValueError('Not a valid layer number. It must greater than zero.')

    card_dir_list = ['NW', 'NNW', 'NNE', 'NE', 'ENE', 'ESE',
                     'SE', 'SSE', 'SSW', 'SW', 'WSW', 'WNW']

    if ring > 2 and ring < power:
        if card_dir in card_dir_list:
            return ['_'.join(['', str(layer + 1), str(ring), card_dir]),
                    '_'.join(['', str(layer + 1), str(ring - 1), card_dir])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring > 2 and ring == power:
        if card_dir in card_dir_list:
            return ['_'.join(['', str(layer + 1), str(ring - 1), card_dir])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 2 and ring < power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer + 1), str(ring), 'NW']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'NW'])]
        elif card_dir == 'NNW':
            return ['_'.join(['', str(layer + 1), str(ring), 'NNW']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'NW'])]
        elif card_dir == 'NNE':
            return ['_'.join(['', str(layer + 1), str(ring), 'NNE']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'NE'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer + 1), str(ring), 'NE']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'NE'])]
        elif card_dir == 'ENE':
            return ['_'.join(['', str(layer + 1), str(ring), 'ENE']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'NE'])]
        elif card_dir == 'ESE':
            return ['_'.join(['', str(layer + 1), str(ring), 'ESE']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'SE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer + 1), str(ring), 'SSE']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'SE'])]
        elif card_dir == 'SSE':
            return ['_'.join(['', str(layer + 1), str(ring), 'SSE']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'SE'])]
        elif card_dir == 'SSW':
            return ['_'.join(['', str(layer + 1), str(ring), 'SSW']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'SW'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer + 1), str(ring), 'SW']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'SW'])]
        elif card_dir == 'WSW':
            return ['_'.join(['', str(layer + 1), str(ring), 'WSW']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'SW'])]
        elif card_dir == 'WNW':
            return ['_'.join(['', str(layer + 1), str(ring), 'WNW']),
                    '_'.join(['', str(layer + 1), str(ring - 1), 'NW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 2 and power == 2:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'NW'])]
        elif card_dir == 'NNW':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'NW'])]
        elif card_dir == 'NNE':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'NE'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'NE'])]
        elif card_dir == 'ENE':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'NE'])]
        elif card_dir == 'ESE':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'SE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'SE'])]
        elif card_dir == 'SSE':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'SE'])]
        elif card_dir == 'SSW':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'SW'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'SW'])]
        elif card_dir == 'WSW':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'SW'])]
        elif card_dir == 'WNW':
            return ['_'.join(['', str(layer + 1), str(ring - 1), 'NW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW, NNW, NNE, ENE, ESE, SSE, '
                             'SSW, WSW, WNW available.')
    elif ring == 1 and ring < power:
        if card_dir == 'NW':
            return ['_'.join(['', str(layer + 1), str(ring), 'NW'])]
        elif card_dir == 'NE':
            return ['_'.join(['', str(layer + 1), str(ring), 'NE'])]
        elif card_dir == 'SE':
            return ['_'.join(['', str(layer + 1), str(ring), 'SE'])]
        elif card_dir == 'SW':
            return ['_'.join(['', str(layer + 1), str(ring), 'SW'])]
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW available.')
    elif ring == 1 and power == 1:
        if card_dir in ['NW', 'NE', 'SE', 'SW']:
            return []
        else:
            raise ValueError('Cardinality is not a valid value only '
                             'NW, NE, SE, SW available.')
    else:
        raise ValueError("I can't believe you've done this. " +
                         "Something went wrong with string in 'name'.")


def fractal_connections(max_power, n_color, hidden_sizes, output_sizes,
                        context_from_top_0_0=True):
    """
    hidden_sizes is a list ordered from inner ring first to outer it contains
    the hidden size of a ring
    output_sizes is a list of lists ordered first by layer then ordered from
    inner ring first to outer it contains output size of a ring
    """
    if len(hidden_sizes) != max_power:
        raise ValueError('')
    elif len(output_sizes) != max_power:
        raise ValueError('')

    connect_dict = OrderedDict()
    unit_count = 0
    if context_from_top_0_0:
        top = ['_'.join(['', str(max_power), '0_0'])]
    else:
        top = []

    def ret_loc(x, y, corner=''):
        if corner == '':
            if x == 0 and y == 0:
                loc = '_NW'
            elif x == 0 and y == 3:
                loc = '_SW'
            elif x == 3 and y == 0:
                loc = '_NE'
            elif x == 3 and y == 3:
                loc = '_SE'
            elif x == 0 and y == 1:
                loc = '_WNW'
            elif x == 1 and y == 0:
                loc = '_NNW'
            elif x == 2 and y == 0:
                loc = '_NNE'
            elif x == 3 and y == 1:
                loc = '_ENE'
            elif x == 3 and y == 2:
                loc = '_ESE'
            elif x == 2 and y == 3:
                loc = '_SSE'
            elif x == 1 and y == 3:
                loc = '_SSW'
            elif x == 0 and y == 2:
                loc = '_WSW'
            else:
                raise ValueError('Something went horribly wrong')
        elif corner == 'SE':
            if x == 0 and y == 0:
                loc = '_NW'
            elif x == 1 and y == 0:
                loc = '_NNW'
            elif x == 0 and y == 1:
                loc = '_WNW'
        elif corner == 'SW':
            if x == 1 and y == 0:
                loc = '_NE'
            elif x == 0 and y == 0:
                loc = '_NNE'
            elif x == 1 and y == 1:
                loc = '_ENE'
        elif corner == 'NW':
            if x == 1 and y == 1:
                loc = '_SE'
            elif x == 1 and y == 0:
                loc = '_ESE'
            elif x == 0 and y == 1:
                loc = '_SSE'
        elif corner == 'NE':
            if x == 0 and y == 1:
                loc = '_SW'
            elif x == 1 and y == 1:
                loc = '_SSW'
            elif x == 0 and y == 0:
                loc = '_WSW'
        return loc

    def fractal_construct(power, layer, L, connect_dict, unit_count,
                          first_rec=True, corner=""):
        """
        :param power:
        :param layer:
        :param L:
        :param connect_dict:
        :param unit_count:
        :param first_rec:
        :param corner: (optional but please leave empty) Corner that
        recursion occurs in.
        :return:
        """
        # don't change the data type
        L_y, L_x = L, L

        if np.log2(L_x) % 1 != 0:
            raise ValueError('Image size is not a power of 2 along horizontal edge.')

        if first_rec:
            x_div, y_div = 4, 4
        else:
            x_div, y_div = 2, 2

        l_x, l_y = L_x // x_div, L_y // y_div

        if layer == 0:
            raw_input_size = l_x * l_y * n_color
        else:
            raw_input_size = 0

        if first_rec and (L_x > 4):
            for y, x in product(range(y_div), range(x_div)):
                if x == 0 or y == 0 or x == 3 or y == 3:
                    loc = ret_loc(x, y)
                    name = '_{0}_{1}'.format(layer, power) + loc
                    suplist = feeds2_fractal_connections(power, name)
                    latlist = layer_fractal_connections(power, name)
                    latsuplist = latlist + suplist + top
                    fedfromlist = fedfrom_fractal_connections(power, name)
                    hidden_size = hidden_sizes[power - 1]
                    output_size = output_sizes[power - 1]
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          fedfromlist,
                                          latsuplist)
                    unit_count += 1
                elif (x == 1 and y == 1):
                    sub_dict, unit_count = fractal_construct(power - 1,
                                                             layer,
                                                             l_x,
                                                             connect_dict,
                                                             unit_count,
                                                             first_rec=False,
                                                             corner="SE")
                    connect_dict.update(sub_dict)
                elif (x == 1 and y == 2):
                    sub_dict, unit_count = fractal_construct(power - 1,
                                                             layer,
                                                             l_x,
                                                             connect_dict,
                                                             unit_count,
                                                             first_rec=False,
                                                             corner="NE")
                    connect_dict.update(sub_dict)
                elif (x == 2 and y == 1):
                    sub_dict, unit_count = fractal_construct(power - 1,
                                                             layer,
                                                             l_x,
                                                             connect_dict,
                                                             unit_count,
                                                             first_rec=False,
                                                             corner="SW")
                    connect_dict.update(sub_dict)
                elif (x == 2 and y == 2):
                    sub_dict, unit_count = fractal_construct(power - 1,
                                                             layer,
                                                             l_x,
                                                             connect_dict,
                                                             unit_count,
                                                             first_rec=False,
                                                             corner="NW")
                    connect_dict.update(sub_dict)
        elif not first_rec and (L_x > 2):
            for y, x in product(range(y_div), range(x_div)):
                if corner == "SE":
                    if x == 0 or (x == 1 and y == 0):
                        loc = ret_loc(x, y, corner=corner)
                        name = '_{0}_{1}'.format(layer, power) + loc
                        suplist = feeds2_fractal_connections(power, name)
                        latlist = layer_fractal_connections(power, name)
                        latsuplist = latlist + suplist + top
                        fedfromlist = fedfrom_fractal_connections(power, name)
                        hidden_size = hidden_sizes[power - 1]
                        output_size = output_sizes[power - 1]
                        connect_dict[name] = (unit_count,
                                              raw_input_size,
                                              hidden_size,
                                              output_size,
                                              fedfromlist,
                                              latsuplist)
                        unit_count += 1
                    else:
                        sub_dict, unit_count = fractal_construct(power - 1,
                                                                 layer,
                                                                 l_x,
                                                                 connect_dict,
                                                                 unit_count,
                                                                 first_rec=False,
                                                                 corner="SE")
                        connect_dict.update(sub_dict)
                elif corner == "NE":
                    if x == 0 or (x == 1 and y == 1):
                        loc = ret_loc(x, y, corner=corner)
                        name = '_{0}_{1}'.format(layer, power) + loc
                        suplist = feeds2_fractal_connections(power, name)
                        latlist = layer_fractal_connections(power, name)
                        latsuplist = latlist + suplist + top
                        fedfromlist = fedfrom_fractal_connections(power, name)
                        hidden_size = hidden_sizes[power - 1]
                        output_size = output_sizes[power - 1]
                        connect_dict[name] = (unit_count,
                                              raw_input_size,
                                              hidden_size,
                                              output_size,
                                              fedfromlist,
                                              latsuplist)
                        unit_count += 1
                    else:
                        sub_dict, unit_count = fractal_construct(power - 1,
                                                                 layer,
                                                                 l_x,
                                                                 connect_dict,
                                                                 unit_count,
                                                                 first_rec=False,
                                                                 corner="NE")
                        connect_dict.update(sub_dict)
                elif corner == "SW":
                    if x == 1 or (x == 0 and y == 0):
                        loc = ret_loc(x, y, corner=corner)
                        name = '_{0}_{1}'.format(layer, power) + loc
                        suplist = feeds2_fractal_connections(power, name)
                        latlist = layer_fractal_connections(power, name)
                        latsuplist = latlist + suplist + top
                        fedfromlist = fedfrom_fractal_connections(power, name)
                        hidden_size = hidden_sizes[power - 1]
                        output_size = output_sizes[power - 1]
                        connect_dict[name] = (unit_count,
                                              raw_input_size,
                                              hidden_size,
                                              output_size,
                                              fedfromlist,
                                              latsuplist)
                        unit_count += 1
                    else:
                        sub_dict, unit_count = fractal_construct(power - 1,
                                                                 layer,
                                                                 l_x,
                                                                 connect_dict,
                                                                 unit_count,
                                                                 first_rec=False,
                                                                 corner="SW")
                        connect_dict.update(sub_dict)
                elif corner == "NW":
                    if x == 1 or (x == 0 and y == 1):
                        loc = ret_loc(x, y, corner=corner)
                        name = '_{0}_{1}'.format(layer, power) + loc
                        suplist = feeds2_fractal_connections(power, name)
                        latlist = layer_fractal_connections(power, name)
                        latsuplist = latlist + suplist + top
                        fedfromlist = fedfrom_fractal_connections(power, name)
                        hidden_size = hidden_sizes[power - 1]
                        output_size = output_sizes[power - 1]
                        connect_dict[name] = (unit_count,
                                              raw_input_size,
                                              hidden_size,
                                              output_size,
                                              fedfromlist,
                                              latsuplist)
                        unit_count += 1
                    else:
                        sub_dict, unit_count = fractal_construct(power - 1,
                                                                 layer,
                                                                 l_x,
                                                                 connect_dict,
                                                                 unit_count,
                                                                 first_rec=False,
                                                                 corner="NW")
                        connect_dict.update(sub_dict)
        elif L_x == 4 and first_rec:
            connect_dict = OrderedDict()
            names_list = ['_{0}_2_NW'.format(layer),
                          '_{0}_2_NNW'.format(layer),
                          '_{0}_2_NNE'.format(layer),
                          '_{0}_2_NE'.format(layer),
                          '_{0}_2_WNW'.format(layer),
                          '_{0}_1_NW'.format(layer),
                          '_{0}_1_NE'.format(layer),
                          '_{0}_2_ENE'.format(layer),
                          '_{0}_2_WSW'.format(layer),
                          '_{0}_1_SW'.format(layer),
                          '_{0}_1_SE'.format(layer),
                          '_{0}_2_ESE'.format(layer),
                          '_{0}_2_SW'.format(layer),
                          '_{0}_2_SSW'.format(layer),
                          '_{0}_2_SSE'.format(layer),
                          '_{0}_2_SE'.format(layer)]
            for name in names_list:
                power = int(name.split('_')[2])
                suplist = feeds2_fractal_connections(power, name)
                latlist = layer_fractal_connections(power, name)
                latsuplist = latlist + suplist + top
                fedfromlist = fedfrom_fractal_connections(power, name)
                hidden_size = hidden_sizes[power - 1]
                output_size = output_sizes[power - 1]
                connect_dict[name] = (unit_count,
                                      raw_input_size,
                                      hidden_size,
                                      output_size,
                                      fedfromlist,
                                      latsuplist)
                unit_count += 1
        elif L_x == 2 and first_rec:
            connect_dict = OrderedDict()
            names_list = ['_{0}_1_NW'.format(layer),
                          '_{0}_1_NE'.format(layer),
                          '_{0}_1_SW'.format(layer),
                          '_{0}_1_SE'.format(layer)]
            for name in names_list:
                suplist = feeds2_fractal_connections(power, name)
                latlist = layer_fractal_connections(power, name)
                latsuplist = latlist + suplist + top
                fedfromlist = fedfrom_fractal_connections(power, name)
                hidden_size = hidden_sizes[0]
                output_size = output_sizes[0]
                connect_dict[name] = (unit_count,
                                      raw_input_size,
                                      hidden_size,
                                      output_size,
                                      fedfromlist,
                                      latsuplist)
                unit_count += 1
        elif L_x == 2:
            # it goes in Z order from here (but not in general)
            connect_dict = OrderedDict()
            if corner == 'SE':
                names_list = ['_{0}_2_NW'.format(layer),
                              '_{0}_2_NNW'.format(layer),
                              '_{0}_2_WNW'.format(layer),
                              '_{0}_1_NW'.format(layer)]
                for name in names_list:
                    power = int(name.split('_')[2])
                    suplist = feeds2_fractal_connections(power, name)
                    latlist = layer_fractal_connections(power, name)
                    latsuplist = latlist + suplist + top
                    fedfromlist = fedfrom_fractal_connections(power, name)
                    hidden_size = hidden_sizes[power - 1]
                    output_size = output_sizes[power - 1]
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          fedfromlist,
                                          latsuplist)
                    unit_count += 1
            elif corner == 'SW':
                names_list = ['_{0}_2_NNE'.format(layer),
                              '_{0}_2_NE'.format(layer),
                              '_{0}_1_NE'.format(layer),
                              '_{0}_2_ENE'.format(layer)]
                for name in names_list:
                    power = int(name.split('_')[2])
                    suplist = feeds2_fractal_connections(power, name)
                    latlist = layer_fractal_connections(power, name)
                    latsuplist = latlist + suplist + top
                    fedfromlist = fedfrom_fractal_connections(power, name)
                    hidden_size = hidden_sizes[power - 1]
                    output_size = output_sizes[power - 1]
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          fedfromlist,
                                          latsuplist)
                    unit_count += 1
            elif corner == 'NE':
                names_list = ['_{0}_2_WSW'.format(layer),
                              '_{0}_1_SW'.format(layer),
                              '_{0}_2_SW'.format(layer),
                              '_{0}_2_SSW'.format(layer)]
                for name in names_list:
                    power = int(name.split('_')[2])
                    suplist = feeds2_fractal_connections(power, name)
                    latlist = layer_fractal_connections(power, name)
                    latsuplist = latlist + suplist + top
                    fedfromlist = fedfrom_fractal_connections(power, name)
                    hidden_size = hidden_sizes[power - 1]
                    output_size = output_sizes[power - 1]
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          fedfromlist,
                                          latsuplist)
                    unit_count += 1
            elif corner == 'NW':
                names_list = ['_{0}_1_SE'.format(layer),
                              '_{0}_2_ESE'.format(layer),
                              '_{0}_2_SSE'.format(layer),
                              '_{0}_2_SE'.format(layer)]
                for name in names_list:
                    power = int(name.split('_')[2])
                    suplist = feeds2_fractal_connections(power, name)
                    latlist = layer_fractal_connections(power, name)
                    latsuplist = latlist + suplist + top
                    fedfromlist = fedfrom_fractal_connections(power, name)
                    hidden_size = hidden_sizes[power - 1]
                    output_size = output_sizes[power - 1]
                    connect_dict[name] = (unit_count,
                                          raw_input_size,
                                          hidden_size,
                                          output_size,
                                          fedfromlist,
                                          latsuplist)
                    unit_count += 1
        return connect_dict, unit_count

    temp_dict = OrderedDict()
    original_max_power = max_power
    layer = 0
    for layer in range(max_power + 1):
        power = original_max_power - layer
        temp_dict, unit_count = fractal_construct(power, layer, 2 ** power,
                                                  temp_dict, unit_count)
        max_power -= 1
        connect_dict.update(temp_dict)
    fedfrom_list = ['_'.join(['', str(layer - 1), '1_NW']),
                    '_'.join(['', str(layer - 1), '1_NE']),
                    '_'.join(['', str(layer - 1), '1_SE']),
                    '_'.join(['', str(layer - 1), '1_SW'])]
    connect_dict['_'.join(['', str(layer), '0_0'])] = (unit_count,
                                                       0,
                                                       hidden_sizes[0],
                                                       output_sizes[0],
                                                       fedfrom_list,
                                                       [])
    return connect_dict
