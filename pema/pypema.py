# -*- coding: utf-8 -*-
"""
A. Folch-Fortuny, R. Marques, I. Isidro, R. Oliveira, A. Ferrer
Copyright (C) 2015 A. Folch-Fortuny
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

import timeit
import time

from pema import dataloader
from pema import calcfuncs
from pema import branching

def run(X, EM, nrel, nbranch, maxPEMs):
    
    # Data pretreatment
    normX, normEM = calcfuncs.pretreatment(X, EM)
    
    tic = timeit.default_timer()
    
    # Run PEMA according to branch number
    print("Running PEMA for {0} relaxations and {1} branch points".format(nrel, nbranch))
    if nbranch == 1:
        result = branching.branch1(normX, normEM, nrel, maxPEMs)
    elif nbranch == 2:
        result = branching.branch2(normX, normEM, nrel, maxPEMs)
    elif nbranch == 3:
        result = branching.branch3(normX, normEM, nrel, maxPEMs)
    elif nbranch == 4:
        result = branching.branch4(normX, normEM, nrel, maxPEMs)
    elif nbranch == 5:
        result = branching.branch5(normX, normEM, nrel, maxPEMs)
    elif nbranch == 6:
        result = branching.branch6(normX, normEM, nrel, maxPEMs)
    elif nbranch == 7:
        result = branching.branch7(normX, normEM, nrel, maxPEMs)
    elif nbranch == 8:
        result = branching.branch8(normX, normEM, nrel, maxPEMs)
    elif nbranch == 9:
        result = branching.branch9(normX, normEM, nrel, maxPEMs)
    elif nbranch == 10:
        result = branching.branch10(normX, normEM, nrel, maxPEMs)
    else:
        raise ValueError('Only a maximum of 10 branches are allowed.')
    
    toc = timeit.default_timer()
    time.sleep(0.2)
    print("Elapsed time is {} seconds".format(toc - tic))
    
    return result
