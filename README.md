# Sa-correlations
This repository provides Python code to compute damping-dependent correlations of response spectral ordinates. The following paper explains the method:

Poulos, A., & Miranda, E. (2022). Damping-dependent correlations between response spectral ordinates. Manuscript submitted for publication.

The computed correlations correspond to those between the residuals of two spectral ordinates of linear elastic single-degree-of-freedom oscillators with different periods and different damping ratios. The method is used on the NGA-West2 ground motion database to fit a regression model. The inputs of the model are the two periods and the two damping ratios of the oscillators. The output of the model is their correlation.

* `main.py` runs the complete analysis that fits the model and produces figures.
* `correlationModel.py` computes correlations using the fitted correlation model. Files rho5.csv, A.csv, B.csv, and C.csv must be in the same folder as this function.
* `example.py` is an example use of `correlationModel.py`, which reproduces part of Figure 8a of the paper.
