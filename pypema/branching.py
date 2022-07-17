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

# TODO: reimplement this module to allow easier maintainability

import numpy as np

import progressbar as pb

from pypema import calcfuncs


def branch_1(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 1
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(
        fluxes, elementary_modes, np.array((), dtype='int64'), n_relax
    )
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0  # progress bar iterator
    
    for i in range(n_relax): # out_elmo_stack[0].size is too verbose
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        prev_exp_var = exp_var_stack[0][i]
        
        ipem = 0 # counter for principal elementary modes
        while ipem < max_pems-1 and prev_exp_var < 100:
            ipem += 1
            all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(
                fluxes, elementary_modes, extra_elmos, n_relax
            )
            
            if n_relax > 0 and exp_var_stack[1][0] > prev_exp_var:
                
                if exp_var_stack[1][0] > result[ipem,0]:
                    ne = all_elmos.shape[0]
                    
                    result[ipem][0] = exp_var_stack[1][0]
                    result[ipem][range(1, ne+1)] = all_elmos[:,0]
                    prev_exp_var = exp_var_stack[1][0]
                    
                extra_elmos = np.hstack((extra_elmos, out_elmo_stack[1][0]))
                
            else:
                ipem = max_pems - 1
        ibar += 1
        pbar.update(ibar)
    pbar.finish()
    return result


def branch_2(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 2
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = \
        calcfuncs.generic_high_elmos(
        fluxes, elementary_modes, np.array((), dtype='int64'), n_relax
    )
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = [
        '\r',
        'Progress: ',
        pb.Bar(marker='#', left='[', right=']'),
        ' ',
        pb.Percentage()
    ]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0  # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]]) # select an elementary_modes from the relaxed EMs
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = \
            calcfuncs.generic_high_elmos(
            fluxes, elementary_modes, extra_elmos, n_relax
        )
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack(
                (out_elmo_stack[0][i], out_elmo_stack[1][j])
            )
            prev_exp_var = exp_var_stack[1][j]
            
            ipem = 1
            while ipem < max_pems-1 and prev_exp_var < 100:
                ipem += 1
                all_elmos, out_elmo_stack[2], exp_var_stack[2] = \
                    calcfuncs.generic_high_elmos(
                        fluxes, elementary_modes, extra_elmos, n_relax
                    )
                
                if n_relax > 0 and exp_var_stack[2][0] > prev_exp_var:
                    
                    if exp_var_stack[2][0] > result[ipem,0]:
                        ne = all_elmos.shape[0]
                        
                        result[ipem][0] = exp_var_stack[2][0]
                        result[ipem][range(1, ne+1)] = all_elmos[:,0]
                    
                    extra_elmos = np.hstack(
                        (extra_elmos, out_elmo_stack[2][0])
                    )
                    prev_exp_var = exp_var_stack[2][0]
                else:
                    ipem = max_pems - 1
            ibar += 1
            pbar.update(ibar)
    pbar.finish()
    return result


def branch_3(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 3
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
            
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                prev_exp_var = exp_var_stack[2][k]
                
                ipem = 2
                while ipem < max_pems-1 and prev_exp_var < 100:
                    ipem += 1
                    all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    if n_relax > 0 and exp_var_stack[3][0] > prev_exp_var:
                        
                        if exp_var_stack[3][0] > result[ipem,0]:
                            ne = all_elmos.shape[0]
                            
                            result[ipem][0] = exp_var_stack[3][0]
                            result[ipem][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((extra_elmos, out_elmo_stack[3][0]))
                        prev_exp_var = exp_var_stack[3][0]
                    else:
                        ipem = max_pems - 1    
                ibar += 1
                pbar.update(ibar)
    pbar.finish()     
    return result


def branch_4(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 4
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = -1 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                          out_elmo_stack[3][o]))
                    prev_exp_var = exp_var_stack[3][0]
                    
                    ipem = 3
                    while ipem < max_pems-1 and prev_exp_var < 100:
                        ipem += 1
                        all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                        
                        if n_relax > 0 and exp_var_stack[4][0] > prev_exp_var:
                            
                            if exp_var_stack[4][0] > result[ipem,0]:
                                ne = all_elmos.shape[0]
                                
                                result[ipem][0] = exp_var_stack[4][0]
                                result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                
                            extra_elmos = np.hstack((extra_elmos, out_elmo_stack[4][0]))
                            prev_exp_var = exp_var_stack[4][0]
                        else:
                            ipem = max_pems - 1
                    ibar += 1
                    pbar.update(ibar)
    pbar.finish()
    return result


def branch_5(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 5
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k],
                                          out_elmo_stack[3][o]))
                    all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    for p in range(n_relax):
                        # Fifth branch point
                        if exp_var_stack[4][p] > result[4][0]:
                            ne = all_elmos.shape[0]
                            
                            result[4][0] =  exp_var_stack[4][p]
                            result[4][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                              out_elmo_stack[3][o], out_elmo_stack[4][p]))
                        prev_exp_var = exp_var_stack[4][0]
                        
                        ipem = 4
                        while ipem < max_pems-1 and prev_exp_var < 100:
                            ipem += 1
                            all_elmos, out_elmo_stack[5], exp_var_stack[5] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                            
                            if n_relax > 0 and exp_var_stack[5][0] > prev_exp_var:
                                
                                if exp_var_stack[5][0] > result[ipem,0]:
                                    ne = all_elmos.shape[0]
                                    
                                    result[ipem][0] = exp_var_stack[5][0]
                                    result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                    
                                extra_elmos = np.hstack((extra_elmos, out_elmo_stack[5][0]))
                                prev_exp_var = exp_var_stack[5][0]
                            else:
                                ipem = max_pems - 1
                        ibar += 1
                        pbar.update(ibar)
    pbar.finish()
    return result


def branch_6(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 6
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k],
                                          out_elmo_stack[3][o]))
                    all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    for p in range(n_relax):
                        # Fifth branch point
                        if exp_var_stack[4][p] > result[4][0]:
                            ne = all_elmos.shape[0]
                            
                            result[4][0] =  exp_var_stack[4][p]
                            result[4][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                              out_elmo_stack[3][o], out_elmo_stack[4][p]))
                        all_elmos, out_elmo_stack[5], exp_var_stack[5] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                        
                        for q in range(n_relax):
                            # Sixth branch point
                            if exp_var_stack[5][q] > result[5][0]:
                                ne = all_elmos.shape[0]
                                
                                result[5][0] = exp_var_stack[5][q]
                                result[5][range(1, ne+1)] = all_elmos[:,0]
                                
                            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                  out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q]))
                            prev_exp_var = exp_var_stack[5][0]
                            
                            ipem = 5
                            while ipem < max_pems-1 and prev_exp_var < 100:
                                ipem += 1
                                all_elmos, out_elmo_stack[6], exp_var_stack[6] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                
                                if n_relax > 0 and exp_var_stack[6][0] > prev_exp_var:
                                    
                                    if exp_var_stack[6][0] > result[ipem,0]:
                                        ne = all_elmos.shape[0]
                                        
                                        result[ipem][0] = exp_var_stack[6][0]
                                        result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                        
                                    extra_elmos = np.hstack((extra_elmos, out_elmo_stack[6][0]))
                                    prev_exp_var = exp_var_stack[6][0]
                                else:
                                    ipem = max_pems - 1
                            ibar += 1
                            pbar.update(ibar)
    pbar.finish()
    return result


def branch_7(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 7
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k],
                                          out_elmo_stack[3][o]))
                    all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    for p in range(n_relax):
                        # Fifth branch point
                        if exp_var_stack[4][p] > result[4][0]:
                            ne = all_elmos.shape[0]
                            
                            result[4][0] =  exp_var_stack[4][p]
                            result[4][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                              out_elmo_stack[3][o], out_elmo_stack[4][p]))
                        all_elmos, out_elmo_stack[5], exp_var_stack[5] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                        
                        for q in range(n_relax):
                            # Sixth branch point
                            if exp_var_stack[5][q] > result[5][0]:
                                ne = all_elmos.shape[0]
                                
                                result[5][0] = exp_var_stack[5][q]
                                result[5][range(1, ne+1)] = all_elmos[:,0]
                                
                            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                  out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q]))
                            all_elmos, out_elmo_stack[6], exp_var_stack[6] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                            
                            for r in range(n_relax):
                                # Seventh branch point
                                if exp_var_stack[6][r] > result[6][0]:
                                    ne = all_elmos.shape[0]
                                    
                                    result[6][0] = exp_var_stack[6][r]
                                    result[6][range(1, ne+1)] = all_elmos[:,0]
                                    
                                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                      out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                      out_elmo_stack[6][r]))
                                prev_exp_var = exp_var_stack[6][0]
                                
                                ipem = 6
                                while ipem < max_pems-1 and prev_exp_var < 100:
                                    ipem += 1
                                    all_elmos, out_elmo_stack[7], exp_var_stack[7] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                    
                                    if n_relax > 0 and exp_var_stack[7][0] > prev_exp_var:
                                        
                                        if exp_var_stack[7][0] > result[ipem][0]:
                                            ne = all_elmos.shape[0]
                                            
                                            result[ipem][0] = exp_var_stack[7][0]
                                            result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                            
                                        extra_elmos = np.hstack((extra_elmos, out_elmo_stack[7][0]))
                                        prev_exp_var = exp_var_stack[7][0]
                                    else:
                                        ipem = max_pems - 1
                                ibar += 1
                                pbar.update(ibar)
    pbar.finish()
    return result


def branch_8(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 8
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k],
                                          out_elmo_stack[3][o]))
                    all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    for p in range(n_relax):
                        # Fifth branch point
                        if exp_var_stack[4][p] > result[4][0]:
                            ne = all_elmos.shape[0]
                            
                            result[4][0] =  exp_var_stack[4][p]
                            result[4][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                              out_elmo_stack[3][o], out_elmo_stack[4][p]))
                        all_elmos, out_elmo_stack[5], exp_var_stack[5] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                        
                        for q in range(n_relax):
                            # Sixth branch point
                            if exp_var_stack[5][q] > result[5][0]:
                                ne = all_elmos.shape[0]
                                
                                result[5][0] = exp_var_stack[5][q]
                                result[5][range(1, ne+1)] = all_elmos[:,0]
                                
                            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                  out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q]))
                            all_elmos, out_elmo_stack[6], exp_var_stack[6] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                            
                            for r in range(n_relax):
                                # Seventh branch point
                                if exp_var_stack[6][r] > result[6][0]:
                                    ne = all_elmos.shape[0]
                                    
                                    result[6][0] = exp_var_stack[6][r]
                                    result[6][range(1, ne+1)] = all_elmos[:,0]
                                    
                                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                      out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                      out_elmo_stack[6][r]))
                                all_elmos, out_elmo_stack[7], exp_var_stack[7] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                
                                for s in range(n_relax):
                                    # Eighth branch point
                                    if exp_var_stack[7][s] > result[7][0]:
                                        ne = all_elmos.shape[0]
                                        
                                        result[7][0] = exp_var_stack[7][s]
                                        result[7][range(1, ne+1)] = all_elmos[:,0]
                                        
                                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                          out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                          out_elmo_stack[6][r], out_elmo_stack[7][s]))
                                    prev_exp_var = exp_var_stack[7][0]
                                    
                                    ipem = 7
                                    while ipem < max_pems-1 and prev_exp_var < 100:
                                        ipem += 1
                                        all_elmos, out_elmo_stack[8], exp_var_stack[8] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                        
                                        if n_relax > 0 and exp_var_stack[8][0] > prev_exp_var:
                                            
                                            if exp_var_stack[8][0] > result[ipem][0]:
                                                ne = all_elmos.shape[0]
                                                
                                                result[ipem][0] = exp_var_stack[8][0]
                                                result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                                
                                            extra_elmos = np.hstack((extra_elmos, out_elmo_stack[8][0]))
                                            prev_exp_var = exp_var_stack[8][0]
                                        else:
                                            ipem = max_pems - 1
                                    ibar += 1
                                    pbar.update(ibar)
    pbar.finish()
    return result


def branch_9(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 9
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k],
                                          out_elmo_stack[3][o]))
                    all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    for p in range(n_relax):
                        # Fifth branch point
                        if exp_var_stack[4][p] > result[4][0]:
                            ne = all_elmos.shape[0]
                            
                            result[4][0] =  exp_var_stack[4][p]
                            result[4][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                              out_elmo_stack[3][o], out_elmo_stack[4][p]))
                        all_elmos, out_elmo_stack[5], exp_var_stack[5] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                        
                        for q in range(n_relax):
                            # Sixth branch point
                            if exp_var_stack[5][q] > result[5][0]:
                                ne = all_elmos.shape[0]
                                
                                result[5][0] = exp_var_stack[5][q]
                                result[5][range(1, ne+1)] = all_elmos[:,0]
                                
                            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                  out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q]))
                            all_elmos, out_elmo_stack[6], exp_var_stack[6] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                            
                            for r in range(n_relax):
                                # Seventh branch point
                                if exp_var_stack[6][r] > result[6][0]:
                                    ne = all_elmos.shape[0]
                                    
                                    result[6][0] = exp_var_stack[6][r]
                                    result[6][range(1, ne+1)] = all_elmos[:,0]
                                    
                                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                      out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                      out_elmo_stack[6][r]))
                                all_elmos, out_elmo_stack[7], exp_var_stack[7] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                
                                for s in range(n_relax):
                                    # Eighth branch point
                                    if exp_var_stack[7][s] > result[7][0]:
                                        ne = all_elmos.shape[0]
                                        
                                        result[7][0] = exp_var_stack[7][r]
                                        result[7][range(1, ne+1)] = all_elmos[:,0]
                                        
                                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                          out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                          out_elmo_stack[6][r], out_elmo_stack[7][r]))
                                    all_elmos, out_elmo_stack[8], exp_var_stack[8] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                    
                                    for t in range(n_relax):
                                        # Ninth branch point
                                        if exp_var_stack[8][t] > result[8][0]:
                                            ne = all_elmos.shape[0]
                                            
                                            result[8][0] = exp_var_stack[8][t]
                                            result[8][range(1, ne+1)] = all_elmos[:,0]
                                            
                                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                              out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                              out_elmo_stack[6][r], out_elmo_stack[7][s], out_elmo_stack[8][t]))
                                        prev_exp_var = exp_var_stack[8][0]
                                        
                                        ipem = 8
                                        while ipem < max_pems-1 and prev_exp_var < 100:
                                            ipem += 1
                                            all_elmos, out_elmo_stack[9], exp_var_stack[9] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                            
                                            if n_relax > 0 and exp_var_stack[9][0] > prev_exp_var:
                                                
                                                if exp_var_stack[9][0] > result[ipem][0]:
                                                    ne = all_elmos.shape[0]
                                                    
                                                    result[ipem][0] = exp_var_stack[9][0]
                                                    result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                                    
                                                extra_elmos = np.hstack((extra_elmos, out_elmo_stack[9][0]))
                                                prev_exp_var = exp_var_stack[9][0]
                                            else:
                                                ipem = max_pems - 1
                                        ibar += 1
                                        pbar.update(ibar)
    pbar.finish()
    return result


def branch_10(fluxes, elementary_modes, n_relax, max_pems):
    n_branch = 10
    
    # Create working variables
    result = np.zeros((max_pems, max_pems + 1))
    out_elmo_stack = [None]*(n_branch + 1) # stores extracted EMs for each branch
    exp_var_stack = [None]*(n_branch + 1) # stores variance values for each branch
    
    all_elmos, out_elmo_stack[0], exp_var_stack[0] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, np.array((), dtype='int64'), n_relax)
    
    result[0][0] = exp_var_stack[0][0]
    result[0][1] = out_elmo_stack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=n_relax**n_branch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(n_relax):
        # First branch point
        extra_elmos = np.array([out_elmo_stack[0][i]])
        all_elmos, out_elmo_stack[1], exp_var_stack[1] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
        
        for j in range(n_relax):
            # Second branch point
            if exp_var_stack[1][j] > result[1][0]:
                ne = all_elmos.shape[0]
                
                result[1][0] = exp_var_stack[1][j]
                result[1][range(1, ne+1)] = all_elmos[:,0]
                
            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j]))
            all_elmos, out_elmo_stack[2], exp_var_stack[2] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
            
            for k in range(n_relax):
                # Third branch point
                if exp_var_stack[2][k] > result[2][0]:
                    ne = all_elmos.shape[0]
                    
                    result[2][0] = exp_var_stack[2][k]
                    result[2][range(1, ne+1)] = all_elmos[:,0]
                    
                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k]))
                all_elmos, out_elmo_stack[3], exp_var_stack[3] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                
                for o in range(n_relax):
                    # Fourth branch point
                    if exp_var_stack[3][o] > result[3][0]:
                        ne = all_elmos.shape[0]
                        
                        result[3][0] = exp_var_stack[3][o]
                        result[3][range(1, ne+1)] = all_elmos[:,0]
                        
                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k],
                                          out_elmo_stack[3][o]))
                    all_elmos, out_elmo_stack[4], exp_var_stack[4] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                    
                    for p in range(n_relax):
                        # Fifth branch point
                        if exp_var_stack[4][p] > result[4][0]:
                            ne = all_elmos.shape[0]
                            
                            result[4][0] =  exp_var_stack[4][p]
                            result[4][range(1, ne+1)] = all_elmos[:,0]
                            
                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                              out_elmo_stack[3][o], out_elmo_stack[4][p]))
                        all_elmos, out_elmo_stack[5], exp_var_stack[5] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                        
                        for q in range(n_relax):
                            # Sixth branch point
                            if exp_var_stack[5][q] > result[5][0]:
                                ne = all_elmos.shape[0]
                                
                                result[5][0] = exp_var_stack[5][q]
                                result[5][range(1, ne+1)] = all_elmos[:,0]
                                
                            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                  out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q]))
                            all_elmos, out_elmo_stack[6], exp_var_stack[6] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                            
                            for r in range(n_relax):
                                # Seventh branch point
                                if exp_var_stack[6][r] > result[6][0]:
                                    ne = all_elmos.shape[0]
                                    
                                    result[6][0] = exp_var_stack[6][r]
                                    result[6][range(1, ne+1)] = all_elmos[:,0]
                                    
                                extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                      out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                      out_elmo_stack[6][r]))
                                all_elmos, out_elmo_stack[7], exp_var_stack[7] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                
                                for s in range(n_relax):
                                    # Eighth branch point
                                    if exp_var_stack[7][s] > result[7][0]:
                                        ne = all_elmos.shape[0]
                                        
                                        result[7][0] = exp_var_stack[7][r]
                                        result[7][range(1, ne+1)] = all_elmos[:,0]
                                        
                                    extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                          out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                          out_elmo_stack[6][r], out_elmo_stack[7][r]))
                                    all_elmos, out_elmo_stack[8], exp_var_stack[8] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                    
                                    for t in range(n_relax):
                                        # Ninth branch point
                                        if exp_var_stack[8][t] > result[7][0]:
                                            ne = all_elmos.shape[0]
                                            
                                            result[8][0] = exp_var_stack[8][t]
                                            result[8][range(1, ne+1)] = all_elmos[:,0]
                                            
                                        extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                              out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                              out_elmo_stack[6][r], out_elmo_stack[7][s], out_elmo_stack[8][t],))
                                        all_elmos, out_elmo_stack[9], exp_var_stack[9] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                        
                                        for u in range(n_relax):
                                            # Tenth branch point
                                            if exp_var_stack[9][u] > result[9][0]:
                                                ne = all_elmos.shape[0]
                                                
                                                result[9][0] = exp_var_stack[9][u]
                                                result[9][range(1, ne+1)] = all_elmos[:,0]
                                                
                                            extra_elmos = np.hstack((out_elmo_stack[0][i], out_elmo_stack[1][j], out_elmo_stack[2][k], 
                                                                  out_elmo_stack[3][o], out_elmo_stack[4][p], out_elmo_stack[5][q],
                                                                  out_elmo_stack[6][r], out_elmo_stack[7][s], out_elmo_stack[8][t],
                                                                  out_elmo_stack[9][u]))
                                            prev_exp_var = exp_var_stack[9][0]
                                            
                                            ipem = 9
                                            while ipem < max_pems-1 and prev_exp_var < 100:
                                                ipem += 1
                                                all_elmos, out_elmo_stack[10], exp_var_stack[10] = calcfuncs.generic_high_elmos(fluxes, elementary_modes, extra_elmos, n_relax)
                                                
                                                if n_relax > 0 and exp_var_stack[10][0] > prev_exp_var:
                                                    
                                                    if exp_var_stack[10][0] > result[ipem][0]:
                                                        ne = all_elmos.shape[0]
                                                        
                                                        result[ipem][0] = exp_var_stack[10][0]
                                                        result[ipem][range(1, ne+1)] = all_elmos[:,0]
                                                        
                                                    extra_elmos = np.hstack((extra_elmos, out_elmo_stack[10][0]))
                                                    prev_exp_var = exp_var_stack[10][0]
                                                else:
                                                    ipem = max_pems - 1
                                            ibar += 1
                                            pbar.update(ibar)
    pbar.finish()
    return result
