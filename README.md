# L.lactis_metabolism
Python based L. lactis metabolism model for nisin-induced NOX overexperssion.
It contains all the functions to simulate L. lactis metabolites, to make
parameter estimation and to plot the metabolites curves.

-run_optimization.py allows to optimize with the differential evolution algorith
-run_simulation.py simulates and plots the model results.

both of them need either having the files
  - datos_finales_4rep.csv
  - parametros_input_v*.csv

or modifying the scripts to get other files (with the same format)
