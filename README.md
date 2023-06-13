[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/orgSailingVLM/pySailingVLM/main) [![orgSailingVLM](https://circleci.com/gh/orgSailingVLM/pySailingVLM.svg?style=shield)](https://app.circleci.com/pipelines/github/orgSailingVLM/pySailingVLM)
# About package

This is open source Python package which implements Vortex Lattice Method for initial aerodynamic analysis of upwind sails. Thanks to its light weight requirements, the software can be immediately installed and execuded locally or accessed by cloud environment such as Google Collab. Package users can define own sail geometries and use pySailingVLM inside custom scripts which makes creating a set of dynamics very convenient.

## Jupyter & cli
pySailingVLm can be used as cli script or inside Jupyter Notebook. For more information see docs and interative Notebooks examples.
## See through
pySailingVLM calculates forces, coefficients, moments acting on sails. Results are visualized and plotted:

[<img src="https://github.com/orgSailingVLM/vlmbook/blob/main/figures/pysailingvlm_yacht.png" width="400"/>](pysailingvlm_yacht.png)

Users can also see colormap with their sails (flat example):

[<img src="https://github.com/orgSailingVLM/vlmbook/blob/main/figures/flat_cp.png" width="200"/>](flat_cp.png)

Apart from this, xlsx file is generated containing all results.
# For developers

Before running test do:
```
export NUMBA_DISABLE_JIT=1
```
