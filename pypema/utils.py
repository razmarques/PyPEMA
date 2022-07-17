# -*- coding: utf-8 -*-
"""
Copyright (C) 2018 R. Marques

This file is part of PyPEMA.

PyPEMA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyPEMA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyPEMA.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
# from scipy.io import loadmat


def format_result(result_nparray):
    # Format titlebar
    tb1 = 'Explained variance (%)'
    tb2 = 'Elementary Modes'
    titlebar = tb1 + ' ' * 3 + tb2 + '\n'

    # Format result entries
    row_data = ''
    max_length = 0
    for irow in result_nparray:
        dec = 4  # number of decimal places to round the explained
        # variance value

        # get explained variance
        exp_var = np.round(irow[0], dec)

        # get elementary modes
        n_elmos = irow[1:]

        # get nonzero elementary modes
        non_zero_elmos = np.int64(n_elmos[np.nonzero(n_elmos)])

        # format elementary modes list
        elmo_str = ''
        for elmo in non_zero_elmos:
            elmo_str += str(elmo) + ', '

        row_entry = str(exp_var) + ' ' * (len(tb1) - dec) + \
                    elmo_str.rstrip(', ') + "\n"
        if len(row_entry) > max_length:
            max_length = len(row_entry)

        row_data += row_entry

    # Format title line
    title_line = '-' * max_length + '\n'

    # Assemble the full string
    formatted_result = titlebar + title_line + row_data

    return formatted_result


# def load_matfile(filepath):
#     data = loadmat(filepath)
#
#     fluxes = data['X']
#     elementary_modes = data['EM']
#
#     return fluxes, elementary_modes


def save_formated_result(filename, result_nparray):
    result = format_result(result_nparray)
    with open(filename, 'w') as file:
        file.write(result)
