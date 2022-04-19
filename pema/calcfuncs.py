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


def pretreatment(X, EM):
    """
    This function executes the normalization of the flux and Elementary mode
    data. The procedure computes the division of all matrix columnms by the
    respective standard deviation.
    """
    
    if X.shape[0]:
        S = np.std(X, axis=0, ddof=1).reshape(1,X.shape[1])
        S[S == 0] = 1
        normX = X/S
        normEM = EM/S.T
        
    return normX, normEM


def compute_fluxes_estimate(X, EM, positive_scores):
    
    # Computation of the scores matrix (T)
    T = np.dot(np.dot(X, EM), 
               np.linalg.pinv(
                       np.dot(EM.T, EM)))
    
    # NOTE: verify if this should be a matter of choice!!!
    if positive_scores:
        T[T<0] = 0
    
    Xrec = np.dot(T, EM.T)
    return Xrec, T


def explained_variance(X, EM):
    
    Xrec, _ = compute_fluxes_estimate(X, EM, True)
    Err = X - Xrec
    expVar = 100 * (1 - ( np.sum(np.square(Err)) / np.sum(np.square(X)) ) )
    
    return expVar


def generic_high_EMs(X, EM, EMlist, nrel):
    
    nEMs = EM.shape[1]
    expVar = np.empty(nEMs)
    
    for i in range(nEMs):
        if np.sum(EMlist == i) == 0:
            # Select the subset of elementary modes from EMlist
            EMlist_sel = np.append(EMlist, i)
            EM_sel = EM[:, EMlist_sel]
            
            # Compute the explained variance for the EM subset
            expVar[i] = explained_variance(X, EM_sel)
        else:
            expVar[i] = 0
    
    # Sort elementary modes by descending vlaues of explained variance
    sorted_expVar = np.sort(expVar)[::-1]
    indEMs = np.argsort(expVar)[::-1]
    
    # Compute the output values
    outExpVar = sorted_expVar[:nrel]
    outEMs = indEMs[:nrel]
    
    stacked_EMlist = EMlist.reshape(EMlist.size,1) * np.ones([EMlist.size, outEMs.size])
    allEMs = np.vstack((stacked_EMlist, outEMs))
    
    return allEMs, outEMs, outExpVar
