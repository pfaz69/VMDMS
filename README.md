# VMDMS
This example illustrates a toy application of VMDMS, detailed in the paper "Enhanced Forecasting of Shipboard Electrical Power Demand using Multivariate Input and Variational Mode Decomposition with Mode Selection" by Fazzini, La Tona, Diez, Di Piazza.
To run it, simply type "python main.py".

packages:
python 3.12.11 
tensorflow 2.18.0
matplotlib 3.10.3
scikit-learn 1.7.0
pydot-4.0.0

To create a conda environment:
conda create --name my_env --file requirements.txt

The command: 'pip install -r requirements.txt' should work with any Python virtual environment manager (venv, virtualenv, pipenv, Poetry) as well as conda, but the environment should be created first.

Program tested with Ubuntu 20.04.

If you use this code for your research, please consider citing the linked paper "Enhanced forecasting of shipboard electrical power demand using multivariate input and variational mode decomposition with mode selection". Here's a BibTeX reference:

@article{fazzini_2025, 
  year = {2025}, 
  title = {{Enhanced forecasting of shipboard electrical power demand using multivariate input and variational mode decomposition with mode selection}}, 
  author = {Fazzini, Paolo and La Tona, Giuseppe and Diez, Matteo and Di Piazza, Maria Carmela}, 
  journal = {Scientific Reports}, 
  doi = {10.1038/s41598-025-06153-z}, 
  pages = {23941}, 
  number = {1}, 
  volume = {15}
}





