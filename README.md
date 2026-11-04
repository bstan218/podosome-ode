This project contains work from Tim Elston's lab at UNC Chapel Hill: https://elstonlab.web.unc.edu/

Dependencies:
- python version >= 3.12
- matplotlib
- numpy
- scikit-learn
- pandas
- deap
- shap (for data analysis only)

The project is organized as follows:
* ode.py contains code for evaluation of systems of ordinary differential equations by the euler method.
* mass_action.py contains the implementation of a mass-action based ODE model to describe the lifecycle of podosome formation. It also defines data structures for storing and creating mass action model parameters.
  It can be called directly to plot the system of a given parameter set by running ```python mass_action.py -i parameters.json```, where parameters.json is a json file containing parameters readable by the MAPFactory class.
* fit.py contains an evolutionary algorithm built on the DEAP framework (https://deap.readthedocs.io/en/master/). It also contains the fitness function for oscillatory behavior.
* ea_analysis.py saves the results of consecutive evolutionary algorithm runs to iterations.csv.
* iterations.sl is a script for slurm job submission of fit.py
* analysis.ipynb is a jupyter notebook containing data analysis of the results of 150 evolutionary algorithm runs.
* p/ contains interesting parameter sets
