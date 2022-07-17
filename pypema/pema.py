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

import click

from pypema.branching import branch_1, branch_2, branch_3, branch_4, branch_5, \
    branch_6, branch_7, branch_8, branch_9, branch_10
from pypema.calcfuncs import pretreatment
from pypema.utils import save_formated_result


def run(fluxes,
        elementary_modes, 
        n_relax, 
        n_branch, 
        max_pems, 
        save_output=False):
    
    # Data pretreatment
    normalised_fluxes, normalised_elmos = pretreatment(
        fluxes, 
        elementary_modes
    )
    
    tic = timeit.default_timer()
    
    # Run PEMA according to branch number
    print("Running PEMA for {0} relaxations and {1} branch points".format(
        n_relax, n_branch)
    )
    if n_branch == 1:
        result = branch_1(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 2:
        result = branch_2(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 3:
        result = branch_3(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 4:
        result = branch_4(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 5:
        result = branch_5(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 6:
        result = branch_6(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 7:
        result = branch_7(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 8:
        result = branch_8(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 9:
        result = branch_9(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    elif n_branch == 10:
        result = branch_10(
            normalised_fluxes, normalised_elmos, n_relax, max_pems
        )
    else:
        raise ValueError('Only a maximum of 10 branches are allowed.')
    
    toc = timeit.default_timer()
    time.sleep(0.2)
    print("Elapsed time is {} seconds".format(toc - tic))

    # Save results to file
    if save_output:
        savefile = 'pems-{0}_rel-{1}_branch'.format(n_relax, n_branch)
        save_formated_result(savefile, result)

    return result


@click.command(
    help="Executes the Principal Elementary Mode Analysis for input "
         "fluxes and Elementary Modes files"
)
@click.argument(
    'flux_input',
    type=click.File('rb')
)
@click.argument(
    'elmo_input',
    type=click.File('rb')
)
@click.argument(
    'nrel',
    type=click.INT
)
@click.argument(
    'nbranch',
    type=click.INT
)
@click.argument(
    'max_pems',
    type=click.INT
)
def cli(flux_input, elmo_input, nrel, nbranch, max_pems):
    print("Running PyPEMA!!!")


if __name__ == "__main__":
    cli()
