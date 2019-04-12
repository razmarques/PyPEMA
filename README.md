# PyPEMA
PyPEMA is a Python implementation of the PEMA Toolbox originaly developed in MATLAB which implements the Principal Elementary Mode Analysis (PEMA) algorithm. This program was designed for Systems Biology applications with the purpose of determining the most significant metabolic pathways in a metabolic network in a fluxome dataset. PEMA operates by performing a data compression procedure similar to Principal Component Analysis (PCA) on the fluxome dataset but, instead of computing principal components, it uses the Elementary Modes (EM) as principal components to find which EM subset best capture the variance in the flux data. Further technical details may be found in the original PEMA paper published in the Molecular BioSystems Journal [1].

# Installation
PyPEMA is currently best used on a Python interactive console such as IPython. This way the PyPEMA project can be cloned or downloaded to a local machine and be used right away.

# User Guide
## Data
PyPEMA receives two data inputs, a matrix containing the flux data (X) and a matrix containing the elementary modes of the metabolic network. The former should be structured with flux distributions stacked row-wise, that is, the different fluxes organized along the columns and the particular observations of each flux along the rows. The elementary-modes matrix is organized with the fluxes along the rows and the elementary modes along the columns.

## Run PyPEMA
To run PyPEMA, import the pypema module from pema package and run as following:

```
from pema import pypema
result = pypema.run(X, EM, nrel, nbranch, maxPEMs)
```

Examples for the use of PyPEMA are provided in pypema_example.py

# References
[1] Abel Folch-Fortuny, Rodolfo Marques, Inês A. Isídro, Rui Oliveira, Alberto Ferrer (2016); "Principal Elementary Mode Analysis". Molecular Biosystems, 12: 737-746. DOI: 10.1039/c5mb00828j
