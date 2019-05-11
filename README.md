# PVM_PyCUDA

An CUDA based optimization of the Predictive Vision Model by Piekniewski et. al. in https://arxiv.org/abs/1607.06854 as well as some extensions that I personally used in my research.

## Getting Started

After installing the dependencies 

### Prerequisites

CUDA 8 (Future versions may support CUDA 9 and 10) 
PyCUDA
OpenCV
NumPy
MatPlotLib

### Installing

This module relies on PyCUDA which needs to be installed before anything else can function. All other dependencies are pip/conda installable. So far the CUDA code uses CUDA 8, for it to run on newer versions of CUDA you need to change the reductions in UsefulKernels.cu
