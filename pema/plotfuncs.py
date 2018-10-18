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
import matplotlib as mpl
import matplotlib.pyplot as plt

from pema import calcfuncs

def create_rgb_map(red, green, blue):
    npos = red.size
    pos = np.linspace(0, 1, npos)
    cdict = {'red':[None]*npos, 'green':[None]*npos, 'blue':[None]*npos}
    for i in range(npos):
        cdict['red'][i] = (pos[i], red[i], red[i])
        cdict['green'][i] = (pos[i], green[i], green[i])
        cdict['blue'][i] = (pos[i], blue[i], blue[i])
    
    cmap = mpl.colors.LinearSegmentedColormap('new_cmap', cdict)
    return cmap

def obs_vs_pred(X, EM, EMlist):
    
    EMsel = EM[:, EMlist]
    Xrec, _ = calcfuncs.compute_fluxes_estimate(X, EMsel, True)
    
    X_flat = X.flatten('F')
    Xrec_flat = Xrec.flatten('F')
    
    plt.figure()
    plt.plot(X_flat, Xrec_flat, '.', markersize=8)
    plt.plot([0, np.max(X_flat)], [0, np.max(X_flat)], color='r')
    plt.title("Observed vs. predicted", fontsize=14)
    plt.xlabel("Observed values", fontsize=12)
    plt.ylabel("Predicted values", fontsize=12)

def pem_plot(X, EM, EMlist):
    
    # Select a subset of Elementary Modes and get extreme values
    EMsel = EM[:, EMlist]
    minEM = np.min(EMsel)
    maxEM = np.max(EMsel)
    
    # Create a color map scaled for the value range in EMsel
    plt.figure()
    if minEM < 0:
        
        if -minEM < maxEM:
            y1 = int(round(-minEM*100/maxEM))
            y2 = y1/100
            
            col1 = np.hstack((np.ones(y1),
                              np.linspace(0.99, 0, 100) ))
            col2 = np.hstack((np.arange(1-y2, 1, 0.01),
                              np.linspace(0.99, 0, 100) ))
            col3 = np.hstack((np.arange(1-y2, 1, 0.01),
                              np.ones(100) ))
            
            rb_cmap = create_rgb_map(col1, col2, col3)
            plt.pcolor(EMsel, cmap=rb_cmap)
            plt.colorbar()
        else:
            y1 = int(round(-maxEM*100/minEM))
            y2 = y1/100
            
            col1 = np.hstack((np.ones(100),
                              np.arange(0.99, 1-y2, -0.01),
                              np.array((1-y2)) ))
            col2 = np.hstack((np.linspace(0, 1, 100),
                              np.arange(0.99, 1-y2, -0.01),
                              np.array((1-y2)) ))
            col3 = np.hstack((np.linspace(0, 1, 100),
                              np.ones(y1) ))
            
            rb_cmap = create_rgb_map(col1, col2, col3)
            plt.pcolor(EMsel, cmap=rb_cmap)
            plt.colorbar()
    else:
        col1 = np.linspace(0,1,101)
        col2 = np.linspace(0,1,101)
        col3 = np.ones(101)
        
        rb_cmap = create_rgb_map(col1, col2, col3)
        plt.pcolor(EMsel, cmap=rb_cmap)
        plt.colorbar()
    
    plt.title("Principal Elementary Modes plot", fontsize=14)
    plt.xlabel("PEMs", fontsize=12)
    plt.ylabel("Reactions", fontsize=12)
    
    # Create a binary PEMs plot
    plt.figure()
    EMbin = EMsel
    EMbin[EMsel > 0] = 1
    EMbin[EMsel < 0] = -1
    
    if np.min(EMbin) == 0:
        
        col1 = np.linspace(1,0,101)
        col2 = np.linspace(1,0,101)
        col3 = np.ones(101)
        
        rb_cmap = create_rgb_map(col1, col2, col3)
        plt.pcolor(EMbin, cmap=rb_cmap)
        
    else:
        
        col1 = np.hstack((np.ones(100),
                          np.linspace(1,0,101)))
        col2 = np.hstack((np.linspace(0,1,101),
                          np.linspace(0.99,0,100)))
        col3 = np.hstack((np.linspace(0,1,101),
                          np.ones(100)))
        
        rb_cmap = create_rgb_map(col1, col2, col3)
        plt.pcolor(EMbin, cmap=rb_cmap)
    
    plt.title("Principal Elementary Modes plot (binary)", fontsize=14)
    plt.xlabel("PEMs", fontsize=12)
    plt.ylabel("Reactions", fontsize=12)

def scree_plot(result):
    nnz = np.count_nonzero(result[:,0])
    
    plt.figure()
    plt.plot(range(1,nnz+1), result[range(nnz),0])
    plt.plot(range(1,nnz+1), result[range(nnz),0], '.', color='r', 
             markersize=12)
    
    plt.title("Cumulative scree plot", fontsize=14)
    plt.xlabel("Number of PEMs", fontsize=12)
    plt.ylabel("Explained variance (%) (scaled data)", fontsize=12)

def variance_obs(X, EM, EMlist):
    nxi = X.shape[0]
    expVarObs = [None]*nxi
    EMsel = EM[:, EMlist]
    
    Xrec, _ = calcfuncs.compute_fluxes_estimate(X, EMsel, False)
    Err = X - Xrec
    
    for k in range(nxi):
        expVarObs[k] = 100 * (1 - ( np.sum(np.square(Err[k,:])) / np.sum(np.square(X[k,:])) ) )
    
    plt.figure()
    plt.bar(np.array(range(nxi)), expVarObs)
    plt.title("Explained variance per observation", fontsize=14)
    plt.xlabel("Observation number", fontsize=12)
    plt.ylabel("Explained variance (%)", fontsize=12)

def weighting_plot(X, EM, EMlist):
    EMsel = EM[:, EMlist]
    _, T = calcfuncs.compute_fluxes_estimate(X, EMsel, True)
    
    # create a map color and heat map
    col1 = np.linspace(1,0,101)
    col2 = np.linspace(1,0,101)
    col3 = np.ones(101)
    
    rb_cmap = create_rgb_map(col1, col2, col3)
    plt.figure()
    plt.pcolor(T, cmap=rb_cmap)
    plt.colorbar()
    plt.title("Weighting plot", fontsize=14)
    plt.xlabel("PEMs", fontsize=12)
    plt.ylabel("Observations", fontsize=12)
    
    Tbin = T
    Tbin[Tbin > 0] = 1
    if np.sum(Tbin) == Tbin.size:
        print("Binary plot not shown. All observations use all PEMs")
    else:
        plt.figure()
        plt.pcolor(Tbin, cmap=rb_cmap)
        plt.title("Weighting plot (binary)", fontsize=14)
        plt.xlabel("PEMs", fontsize=12)
        plt.ylabel("Observations", fontsize=12)

def weights_vars(X, EM, EMlist):
    nEM = len(EMlist)
    EMsel = EM[:, EMlist]
    _, T = calcfuncs.compute_fluxes_estimate(X, EMsel, True)
    
    nt = T.shape[0]
    nem = EM.shape[0]
    varEM = np.empty(nEM)
    for i in range(nEM):
        Xrec = np.dot(T[:,i].reshape(nt,1), EMsel[:,i].reshape(1,nem))
        Err = X - Xrec
        varEM[i] = 100*(1 - ( np.sum(np.square(Err)) / np.sum(np.square(X)) ) )
    
    plt.figure()
    plt.plot(varEM)
    plt.plot(varEM, '.', color='r', markersize=12)
    plt.title("Variance explained by each PEM", fontsize=14)
    plt.xlabel("PEMs", fontsize=12)
    plt.ylabel("Explained variance (%)", fontsize=12)
    
    expVar = calcfuncs.explained_variance(X, EMsel)
    degree_ort = expVar/np.sum(varEM)
    print("Degree of orthogonality: {}".format(degree_ort))
