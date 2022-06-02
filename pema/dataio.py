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
import scipy.io as spio


def format_result(result_nparray):

    # Format titlebar
    tb1 = 'Explained variance (%)'
    tb2 = 'Elementary Modes'
    titlebar = tb1 + ' '*3 + tb2 + '\n'

    # Format result entries
    rowdata = ''
    maxlen = 0
    for irow in result_nparray:
        dec = 4  # number of decimal places to round the explained
                 # variance value

        # get explained variance
        exp_var = np.round(irow[0], dec)

        # get elementary modes
        nems = irow[1:]

        # get nonzeros elementary modes
        nznems = np.int64(nems[np.nonzero(nems)])

        # format elementary modes list
        elmostr = ''
        for em in nznems:
            elmostr += str(em) + ', '

        rowentry = str(exp_var) + ' '*(len(tb1) - dec) + elmostr.rstrip(', ') + "\n"
        if len(rowentry) > maxlen:
            maxlen = len(rowentry)

        rowdata += rowentry

    # Format title line
    titleline = '-'*maxlen + '\n'

    # Assemble the full string
    formatted_result = titlebar + titleline + rowdata

    return formatted_result


def load_matfile(filepath):
    data = spio.loadmat(filepath)
    
    fluxes = data['X']
    elementary_modes = data['EM']
    
    return fluxes, elementary_modes


def save_formated_result(filename, result_nparray):
    fres = format_result(result_nparray)

    file = open(filename, 'w')
    file.write(fres)
    file.close()
