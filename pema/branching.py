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

import numpy as np

import progressbar as pb

from pema import calcfuncs

def branch1(X, EM, nrel, maxPEMs):
    nbranch = 1
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel): # outEMsStack[0].size is too verbose
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        prev_expVar = expVarStack[0][i]
        
        ipem = 0 # counter for principal elementary modes
        while ipem < maxPEMs-1 and prev_expVar < 100:
            ipem += 1
            allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            if nrel > 0 and expVarStack[1][0] > prev_expVar:
                
                if expVarStack[1][0] > result[ipem,0]:
                    ne = allEMs.shape[0]
                    
                    result[ipem][0] = expVarStack[1][0]
                    result[ipem][range(1, ne+1)] = allEMs[:,0]
                    prev_expVar = expVarStack[1][0]
                    
                extraEMs = np.hstack((extraEMs, outEMsStack[1][0]))
                
            else:
                ipem = maxPEMs - 1
        ibar += 1
        pbar.update(ibar)
    pbar.finish()
    return result

def branch2(X, EM, nrel, maxPEMs):
    nbranch = 2
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]]) # select an EM from the relaxed EMs
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            prev_expVar = expVarStack[1][j]
            
            ipem = 1
            while ipem < maxPEMs-1 and prev_expVar < 100:
                ipem += 1
                allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                if nrel > 0 and expVarStack[2][0] > prev_expVar:
                    
                    if expVarStack[2][0] > result[ipem,0]:
                        ne = allEMs.shape[0]
                        
                        result[ipem][0] = expVarStack[2][0]
                        result[ipem][range(1, ne+1)] = allEMs[:,0]
                    
                    extraEMs = np.hstack((extraEMs, outEMsStack[2][0]))
                    prev_expVar = expVarStack[2][0]
                else:
                    ipem = maxPEMs - 1
            ibar += 1
            pbar.update(ibar)
    pbar.finish()
    return result

def branch3(X, EM, nrel, maxPEMs):
    nbranch = 3
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
            
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                prev_expVar = expVarStack[2][k]
                
                ipem = 2
                while ipem < maxPEMs-1 and prev_expVar < 100:
                    ipem += 1
                    allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    if nrel > 0 and expVarStack[3][0] > prev_expVar:
                        
                        if expVarStack[3][0] > result[ipem,0]:
                            ne = allEMs.shape[0]
                            
                            result[ipem][0] = expVarStack[3][0]
                            result[ipem][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((extraEMs, outEMsStack[3][0]))
                        prev_expVar = expVarStack[3][0]
                    else:
                        ipem = maxPEMs - 1    
                ibar += 1
                pbar.update(ibar)
    pbar.finish()     
    return result

def branch4(X, EM, nrel, maxPEMs):
    nbranch = 4
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = -1 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                          outEMsStack[3][o]))
                    prev_expVar = expVarStack[3][0]
                    
                    ipem = 3
                    while ipem < maxPEMs-1 and prev_expVar < 100:
                        ipem += 1
                        allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                        
                        if nrel > 0 and expVarStack[4][0] > prev_expVar:
                            
                            if expVarStack[4][0] > result[ipem,0]:
                                ne = allEMs.shape[0]
                                
                                result[ipem][0] = expVarStack[4][0]
                                result[ipem][range(1, ne+1)] = allEMs[:,0]
                                
                            extraEMs = np.hstack((extraEMs, outEMsStack[4][0]))
                            prev_expVar = expVarStack[4][0]
                        else:
                            ipem = maxPEMs - 1
                    ibar += 1
                    pbar.update(ibar)
    pbar.finish()
    return result

def branch5(X, EM, nrel, maxPEMs):
    nbranch = 5
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k],
                                          outEMsStack[3][o]))
                    allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    for p in range(nrel):
                        # Fifth branch point
                        if expVarStack[4][p] > result[4][0]:
                            ne = allEMs.shape[0]
                            
                            result[4][0] =  expVarStack[4][p]
                            result[4][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                              outEMsStack[3][o], outEMsStack[4][p]))
                        prev_expVar = expVarStack[4][0]
                        
                        ipem = 4
                        while ipem < maxPEMs-1 and prev_expVar < 100:
                            ipem += 1
                            allEMs, outEMsStack[5], expVarStack[5] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                            
                            if nrel > 0 and expVarStack[5][0] > prev_expVar:
                                
                                if expVarStack[5][0] > result[ipem,0]:
                                    ne = allEMs.shape[0]
                                    
                                    result[ipem][0] = expVarStack[5][0]
                                    result[ipem][range(1, ne+1)] = allEMs[:,0]
                                    
                                extraEMs = np.hstack((extraEMs, outEMsStack[5][0]))
                                prev_expVar = expVarStack[5][0]
                            else:
                                ipem = maxPEMs - 1
                        ibar += 1
                        pbar.update(ibar)
    pbar.finish()
    return result

def branch6(X, EM, nrel, maxPEMs):
    nbranch = 6
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k],
                                          outEMsStack[3][o]))
                    allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    for p in range(nrel):
                        # Fifth branch point
                        if expVarStack[4][p] > result[4][0]:
                            ne = allEMs.shape[0]
                            
                            result[4][0] =  expVarStack[4][p]
                            result[4][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                              outEMsStack[3][o], outEMsStack[4][p]))
                        allEMs, outEMsStack[5], expVarStack[5] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                        
                        for q in range(nrel):
                            # Sixth branch point
                            if expVarStack[5][q] > result[5][0]:
                                ne = allEMs.shape[0]
                                
                                result[5][0] = expVarStack[5][q]
                                result[5][range(1, ne+1)] = allEMs[:,0]
                                
                            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                  outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q]))
                            prev_expVar = expVarStack[5][0]
                            
                            ipem = 5
                            while ipem < maxPEMs-1 and prev_expVar < 100:
                                ipem += 1
                                allEMs, outEMsStack[6], expVarStack[6] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                
                                if nrel > 0 and expVarStack[6][0] > prev_expVar:
                                    
                                    if expVarStack[6][0] > result[ipem,0]:
                                        ne = allEMs.shape[0]
                                        
                                        result[ipem][0] = expVarStack[6][0]
                                        result[ipem][range(1, ne+1)] = allEMs[:,0]
                                        
                                    extraEMs = np.hstack((extraEMs, outEMsStack[6][0]))
                                    prev_expVar = expVarStack[6][0]
                                else:
                                    ipem = maxPEMs - 1
                            ibar += 1
                            pbar.update(ibar)
    pbar.finish()
    return result

def branch7(X, EM, nrel, maxPEMs):
    nbranch = 7
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k],
                                          outEMsStack[3][o]))
                    allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    for p in range(nrel):
                        # Fifth branch point
                        if expVarStack[4][p] > result[4][0]:
                            ne = allEMs.shape[0]
                            
                            result[4][0] =  expVarStack[4][p]
                            result[4][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                              outEMsStack[3][o], outEMsStack[4][p]))
                        allEMs, outEMsStack[5], expVarStack[5] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                        
                        for q in range(nrel):
                            # Sixth branch point
                            if expVarStack[5][q] > result[5][0]:
                                ne = allEMs.shape[0]
                                
                                result[5][0] = expVarStack[5][q]
                                result[5][range(1, ne+1)] = allEMs[:,0]
                                
                            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                  outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q]))
                            allEMs, outEMsStack[6], expVarStack[6] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                            
                            for r in range(nrel):
                                # Seventh branch point
                                if expVarStack[6][r] > result[6][0]:
                                    ne = allEMs.shape[0]
                                    
                                    result[6][0] = expVarStack[6][r]
                                    result[6][range(1, ne+1)] = allEMs[:,0]
                                    
                                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                      outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                      outEMsStack[6][r]))
                                prev_expVar = expVarStack[6][0]
                                
                                ipem = 6
                                while ipem < maxPEMs-1 and prev_expVar < 100:
                                    ipem += 1
                                    allEMs, outEMsStack[7], expVarStack[7] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                    
                                    if nrel > 0 and expVarStack[7][0] > prev_expVar:
                                        
                                        if expVarStack[7][0] > result[ipem][0]:
                                            ne = allEMs.shape[0]
                                            
                                            result[ipem][0] = expVarStack[7][0]
                                            result[ipem][range(1, ne+1)] = allEMs[:,0]
                                            
                                        extraEMs = np.hstack((extraEMs, outEMsStack[7][0]))
                                        prev_expVar = expVarStack[7][0]
                                    else:
                                        ipem = maxPEMs - 1
                                ibar += 1
                                pbar.update(ibar)
    pbar.finish()
    return result

def branch8(X, EM, nrel, maxPEMs):
    nbranch = 8
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k],
                                          outEMsStack[3][o]))
                    allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    for p in range(nrel):
                        # Fifth branch point
                        if expVarStack[4][p] > result[4][0]:
                            ne = allEMs.shape[0]
                            
                            result[4][0] =  expVarStack[4][p]
                            result[4][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                              outEMsStack[3][o], outEMsStack[4][p]))
                        allEMs, outEMsStack[5], expVarStack[5] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                        
                        for q in range(nrel):
                            # Sixth branch point
                            if expVarStack[5][q] > result[5][0]:
                                ne = allEMs.shape[0]
                                
                                result[5][0] = expVarStack[5][q]
                                result[5][range(1, ne+1)] = allEMs[:,0]
                                
                            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                  outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q]))
                            allEMs, outEMsStack[6], expVarStack[6] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                            
                            for r in range(nrel):
                                # Seventh branch point
                                if expVarStack[6][r] > result[6][0]:
                                    ne = allEMs.shape[0]
                                    
                                    result[6][0] = expVarStack[6][r]
                                    result[6][range(1, ne+1)] = allEMs[:,0]
                                    
                                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                      outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                      outEMsStack[6][r]))
                                allEMs, outEMsStack[7], expVarStack[7] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                
                                for s in range(nrel):
                                    # Eighth branch point
                                    if expVarStack[7][s] > result[7][0]:
                                        ne = allEMs.shape[0]
                                        
                                        result[7][0] = expVarStack[7][s]
                                        result[7][range(1, ne+1)] = allEMs[:,0]
                                        
                                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                          outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                          outEMsStack[6][r], outEMsStack[7][s]))
                                    prev_expVar = expVarStack[7][0]
                                    
                                    ipem = 7
                                    while ipem < maxPEMs-1 and prev_expVar < 100:
                                        ipem += 1
                                        allEMs, outEMsStack[8], expVarStack[8] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                        
                                        if nrel > 0 and expVarStack[8][0] > prev_expVar:
                                            
                                            if expVarStack[8][0] > result[ipem][0]:
                                                ne = allEMs.shape[0]
                                                
                                                result[ipem][0] = expVarStack[8][0]
                                                result[ipem][range(1, ne+1)] = allEMs[:,0]
                                                
                                            extraEMs = np.hstack((extraEMs, outEMsStack[8][0]))
                                            prev_expVar = expVarStack[8][0]
                                        else:
                                            ipem = maxPEMs - 1
                                    ibar += 1
                                    pbar.update(ibar)
    pbar.finish()
    return result

def branch9(X, EM, nrel, maxPEMs):
    nbranch = 9
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k],
                                          outEMsStack[3][o]))
                    allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    for p in range(nrel):
                        # Fifth branch point
                        if expVarStack[4][p] > result[4][0]:
                            ne = allEMs.shape[0]
                            
                            result[4][0] =  expVarStack[4][p]
                            result[4][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                              outEMsStack[3][o], outEMsStack[4][p]))
                        allEMs, outEMsStack[5], expVarStack[5] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                        
                        for q in range(nrel):
                            # Sixth branch point
                            if expVarStack[5][q] > result[5][0]:
                                ne = allEMs.shape[0]
                                
                                result[5][0] = expVarStack[5][q]
                                result[5][range(1, ne+1)] = allEMs[:,0]
                                
                            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                  outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q]))
                            allEMs, outEMsStack[6], expVarStack[6] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                            
                            for r in range(nrel):
                                # Seventh branch point
                                if expVarStack[6][r] > result[6][0]:
                                    ne = allEMs.shape[0]
                                    
                                    result[6][0] = expVarStack[6][r]
                                    result[6][range(1, ne+1)] = allEMs[:,0]
                                    
                                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                      outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                      outEMsStack[6][r]))
                                allEMs, outEMsStack[7], expVarStack[7] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                
                                for s in range(nrel):
                                    # Eighth branch point
                                    if expVarStack[7][s] > result[7][0]:
                                        ne = allEMs.shape[0]
                                        
                                        result[7][0] = expVarStack[7][r]
                                        result[7][range(1, ne+1)] = allEMs[:,0]
                                        
                                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                          outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                          outEMsStack[6][r], outEMsStack[7][r]))
                                    allEMs, outEMsStack[8], expVarStack[8] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                    
                                    for t in range(nrel):
                                        # Ninth branch point
                                        if expVarStack[8][t] > result[8][0]:
                                            ne = allEMs.shape[0]
                                            
                                            result[8][0] = expVarStack[8][t]
                                            result[8][range(1, ne+1)] = allEMs[:,0]
                                            
                                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                              outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                              outEMsStack[6][r], outEMsStack[7][s], outEMsStack[8][t]))
                                        prev_expVar = expVarStack[8][0]
                                        
                                        ipem = 8
                                        while ipem < maxPEMs-1 and prev_expVar < 100:
                                            ipem += 1
                                            allEMs, outEMsStack[9], expVarStack[9] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                            
                                            if nrel > 0 and expVarStack[9][0] > prev_expVar:
                                                
                                                if expVarStack[9][0] > result[ipem][0]:
                                                    ne = allEMs.shape[0]
                                                    
                                                    result[ipem][0] = expVarStack[9][0]
                                                    result[ipem][range(1, ne+1)] = allEMs[:,0]
                                                    
                                                extraEMs = np.hstack((extraEMs, outEMsStack[9][0]))
                                                prev_expVar = expVarStack[9][0]
                                            else:
                                                ipem = maxPEMs - 1
                                        ibar += 1
                                        pbar.update(ibar)
    pbar.finish()
    return result

def branch10(X, EM, nrel, maxPEMs):
    nbranch = 10
    
    # Create working variables
    result = np.zeros((maxPEMs, maxPEMs + 1))
    outEMsStack = [None]*(nbranch + 1) # stores extracted EMs for each branch
    expVarStack = [None]*(nbranch + 1) # stores variance values for each branch
    
    allEMs, outEMsStack[0], expVarStack[0] = calcfuncs.generic_high_EMs(X, EM, np.array((), dtype='int64'), nrel)
    
    result[0][0] = expVarStack[0][0]
    result[0][1] = outEMsStack[0][0]
    
    # Setup progress bar
    options = ['\r', 'Progress: ', pb.Bar(marker='#',left='[',right=']'), ' ', pb.Percentage()]
    pbar = pb.ProgressBar(widgets=options, maxval=nrel**nbranch)
    pbar.start()
    ibar = 0 # progress bar iterator
    
    for i in range(nrel):
        # First branch point
        extraEMs = np.array([outEMsStack[0][i]])
        allEMs, outEMsStack[1], expVarStack[1] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
        
        for j in range(nrel):
            # Second branch point
            if expVarStack[1][j] > result[1][0]:
                ne = allEMs.shape[0]
                
                result[1][0] = expVarStack[1][j]
                result[1][range(1, ne+1)] = allEMs[:,0]
                
            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j]))
            allEMs, outEMsStack[2], expVarStack[2] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
            
            for k in range(nrel):
                # Third branch point
                if expVarStack[2][k] > result[2][0]:
                    ne = allEMs.shape[0]
                    
                    result[2][0] = expVarStack[2][k]
                    result[2][range(1, ne+1)] = allEMs[:,0]
                    
                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k]))
                allEMs, outEMsStack[3], expVarStack[3] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                
                for o in range(nrel):
                    # Fourth branch point
                    if expVarStack[3][o] > result[3][0]:
                        ne = allEMs.shape[0]
                        
                        result[3][0] = expVarStack[3][o]
                        result[3][range(1, ne+1)] = allEMs[:,0]
                        
                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k],
                                          outEMsStack[3][o]))
                    allEMs, outEMsStack[4], expVarStack[4] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                    
                    for p in range(nrel):
                        # Fifth branch point
                        if expVarStack[4][p] > result[4][0]:
                            ne = allEMs.shape[0]
                            
                            result[4][0] =  expVarStack[4][p]
                            result[4][range(1, ne+1)] = allEMs[:,0]
                            
                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                              outEMsStack[3][o], outEMsStack[4][p]))
                        allEMs, outEMsStack[5], expVarStack[5] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                        
                        for q in range(nrel):
                            # Sixth branch point
                            if expVarStack[5][q] > result[5][0]:
                                ne = allEMs.shape[0]
                                
                                result[5][0] = expVarStack[5][q]
                                result[5][range(1, ne+1)] = allEMs[:,0]
                                
                            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                  outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q]))
                            allEMs, outEMsStack[6], expVarStack[6] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                            
                            for r in range(nrel):
                                # Seventh branch point
                                if expVarStack[6][r] > result[6][0]:
                                    ne = allEMs.shape[0]
                                    
                                    result[6][0] = expVarStack[6][r]
                                    result[6][range(1, ne+1)] = allEMs[:,0]
                                    
                                extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                      outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                      outEMsStack[6][r]))
                                allEMs, outEMsStack[7], expVarStack[7] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                
                                for s in range(nrel):
                                    # Eighth branch point
                                    if expVarStack[7][s] > result[7][0]:
                                        ne = allEMs.shape[0]
                                        
                                        result[7][0] = expVarStack[7][r]
                                        result[7][range(1, ne+1)] = allEMs[:,0]
                                        
                                    extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                          outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                          outEMsStack[6][r], outEMsStack[7][r]))
                                    allEMs, outEMsStack[8], expVarStack[8] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                    
                                    for t in range(nrel):
                                        # Ninth branch point
                                        if expVarStack[8][t] > result[7][0]:
                                            ne = allEMs.shape[0]
                                            
                                            result[8][0] = expVarStack[8][t]
                                            result[8][range(1, ne+1)] = allEMs[:,0]
                                            
                                        extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                              outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                              outEMsStack[6][r], outEMsStack[7][s], outEMsStack[8][t],))
                                        allEMs, outEMsStack[9], expVarStack[9] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                        
                                        for u in range(nrel):
                                            # Tenth branch point
                                            if expVarStack[9][u] > result[9][0]:
                                                ne = allEMs.shape[0]
                                                
                                                result[9][0] = expVarStack[9][u]
                                                result[9][range(1, ne+1)] = allEMs[:,0]
                                                
                                            extraEMs = np.hstack((outEMsStack[0][i], outEMsStack[1][j], outEMsStack[2][k], 
                                                                  outEMsStack[3][o], outEMsStack[4][p], outEMsStack[5][q],
                                                                  outEMsStack[6][r], outEMsStack[7][s], outEMsStack[8][t],
                                                                  outEMsStack[9][u]))
                                            prev_expVar = expVarStack[9][0]
                                            
                                            ipem = 9
                                            while ipem < maxPEMs-1 and prev_expVar < 100:
                                                ipem += 1
                                                allEMs, outEMsStack[10], expVarStack[10] = calcfuncs.generic_high_EMs(X, EM, extraEMs, nrel)
                                                
                                                if nrel > 0 and expVarStack[10][0] > prev_expVar:
                                                    
                                                    if expVarStack[10][0] > result[ipem][0]:
                                                        ne = allEMs.shape[0]
                                                        
                                                        result[ipem][0] = expVarStack[10][0]
                                                        result[ipem][range(1, ne+1)] = allEMs[:,0]
                                                        
                                                    extraEMs = np.hstack((extraEMs, outEMsStack[10][0]))
                                                    prev_expVar = expVarStack[10][0]
                                                else:
                                                    ipem = maxPEMs - 1
                                            ibar += 1
                                            pbar.update(ibar)
    pbar.finish()
    return result
