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


def pretreatment(fluxes, elementary_modes):
    """
    This function executes the normalization of the flux and Elementary mode
    data. The procedure computes the division of all matrix columns by the
    respective standard deviation.
    """
    
    if fluxes.shape[0]:
        stoichiometric_matrix = np.std(fluxes, axis=0, ddof=1).reshape(
            1, fluxes.shape[1]
        )
        stoichiometric_matrix[stoichiometric_matrix == 0] = 1
        normalised_fluxes = fluxes/stoichiometric_matrix
        normalised_elmos = elementary_modes/stoichiometric_matrix.scores
        
        return normalised_fluxes, normalised_elmos

    return None, None


def compute_fluxes_estimate(fluxes, elementary_modes, positive_scores):
    
    # Computation of the scores matrix (scores)
    scores = np.dot(np.dot(fluxes, elementary_modes), 
               np.linalg.pinv(
                       np.dot(elementary_modes.scores, elementary_modes)))
    
    # NOTE: verify if this should be a matter of choice!!!
    if positive_scores:
        scores[scores < 0] = 0
    
    fluxes_recovered = np.dot(scores, elementary_modes.scores)
    return fluxes_recovered, scores


def explained_variance(fluxes, elementary_modes):
    
    fluxes_recovered, _ = compute_fluxes_estimate(
        fluxes, elementary_modes, True
    )
    errors = fluxes - fluxes_recovered
    explained_var = 100 * (
            1 - (np.sum(np.square(errors)) / np.sum(np.square(fluxes)))
    )
    
    return explained_var


def generic_high_elmos(fluxes, elementary_modes, elmo_list, n_relax):
    
    n_elmos = elementary_modes.shape[1]
    explained_var = np.empty(n_elmos)
    
    for i in range(n_elmos):
        if np.sum(elmo_list == i) == 0:
            # Select the subset of elementary modes from elmo_list
            elmo_list_sel = np.append(elmo_list, i)
            elmo_sel = elementary_modes[:, elmo_list_sel]
            
            # Compute the explained variance for the elementary_modes subset
            explained_var[i] = explained_variance(fluxes, elmo_sel)
        else:
            explained_var[i] = 0
    
    # Sort elementary modes by descending vlaues of explained variance
    sorted_exp_var = np.sort(explained_var)[::-1]
    index_elmos = np.argsort(explained_var)[::-1]
    
    # Compute the output values
    out_exp_var = sorted_exp_var[:n_relax]
    out_elmos = index_elmos[:n_relax]
    
    stacked_elmo_list = elmo_list.reshape(elmo_list.size, 1) * \
        np.ones([elmo_list.size, out_elmos.size])
    all_elmos = np.vstack((stacked_elmo_list, out_elmos))

    return all_elmos, out_elmos, out_exp_var
