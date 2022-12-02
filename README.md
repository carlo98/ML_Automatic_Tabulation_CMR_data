# Machine Learning in Automatic Tabulation for Constraint Model Reformulation Data
This repository contains the problem classes, instances, dataset and machine learning (ML) models used during the thesis work on "Machine Learning in Automatic Tabulation for Constraint Model Reformulation".

# Table of contents
1. [Organization](#organization)

## Organization <a name="organization"></a>
The repository is organized as follows: 
- The instances of each problem class are grouped in a folder, along with problem's model.
- 'Generators' contains a few python and bash scripts used to generate the instances of some of the problem classes.
- 'Dataset' contains the results obtained by solving the instances with different solvers, i.e. minion, kissat, kissat-mdd and chuffed, as csv files. In the same folder are also available the instances' features (folder 'on-off-feature-instances') and the results obtained for the second task.
- 'ML_logs' contains the results obtained when training a few ML models on the dataset.
- 'Code' contains the code used by us to train the models and to study the results.
