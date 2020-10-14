# Decision trees as partitioning machines to characterize their generalization properties

<p>
<img src="https://github.com/jsleb333/paper-decision-trees-as-partitioning-machines/blob/assets/thumbnail.png" alt="">
</p>

## Preface
This package provides an implementation of the algorithms presented in the NeurIPS 2020 paper "Decision trees as partitioning machines to characterize their generalization properties" by Leboeuf, LeBland and Marchand.

## Requirements

To install requirements, run the following command
```
pip install -r requirements.txt
```
The code was written to run in Python 3.6 or in a more recent version.

## Running the experiments

To run the experiments, navigate from the command line to the directory root directory of the project and run the command
```
python "./experiments/main.py"
```
The datasets should be downloaded automatically and will be stored in the `./experiments/datasets/raw/` folder.
The results will be saved automatically in the files `./experiments/results/<datasetname>/<experiment_name>/<model_name>.csv`.
Running experiments on all 19 datasets took about 1 to 2 hours for each model on my Intel Core i5-750 CPU.

### Run a specific experiment

To run a specific experiment and not all of them, one can pass arguments to the `main.py` file.
For example, to run the experiment of our model on the Iris and Wine datasets, one would use the command
```
python "./experiments/main.py" --model_name=ours --datasets=[iris,wine] --exp_name=first_exp
```
and the experiments on these datasets would be saved under the name `first_exp`. Other acceptable arguments are: `--n_draws`, `--n_folds`, `--max_n_leaves` and `--error_prior_exponent`. Run the `--help` command for more details, or checkout directly the documention of the function `launch_experiment` in the file `./experiments/main.py`.

### Reproducing the tables

All tables of the paper are automatically generated from Python to LaTeX using the `python2latex` package.

To reproduce Table 1, run the command
```
python "./experiments/scripts/process_all_results.py" --exp_name=<exp_name_used_to_run_main>
```

To reproduce Tables 2 to 20 (at the same time), run the command
```
python "./experiments/scripts/process_results_by_datasets.py" --exp_name=<exp_name_used_to_run_main>
```

These scripts will generate the LaTeX code to produce the tables and will try to call `pdflatex` on the tex file created, if installed. Otherwise, they will simply output to the console the TeX string that generated the tables.

## Content

This package provides implementations of the algorithms that computes an upper and lower bound on the VC dimension of binary decision trees, as well as other useful tools to work with scikit-learn decision trees.
The package also provides an implementation in pure Python of a decision tree classifier based on the CART algorithm, and everything needed to prune using CART or our bound-based pruning algorithm. The `experiments` folder contains all the files necessary to reproduce experiments and results.

The directory `partitioning_machines` contains all files related to the implementation of the algorithms presented or used in the paper with unit tests in the directory `tests`, while the directory `experiments` contains the files to reproduce the results and the tables of the paper.

### Detailled content
#### `partitioning_machines` module
- Tree object built in a recursive manner in the file `tree.py`.
- PartitioningFunctionUpperBound object that implements an optimized version of the algorithm 1 of Appendix E in the file `partitioning_function_upper_bound.py`.
- VC dimension lower and upper bounds algorithms provided in the file `vcdim.py`.
- Implementation from scratch of the CART algorithm to greedily grow a binary decision tree with the Gini index or other impurity score in the file `decision_tree_classifier.py`
- Generalization bounds such as Shawe-Taylor's bound and Vapnik's bound in the file `generalization_bounds.py`.
- Label encoder object to handle the labels of the datasets and convert them to a one-hot encoding in the file `label_encoder.py`.
##### `utils` module
- Tree conversion from scikit-learn decision trees to new implementation in the file `convert_tree.py`.
- Tools to draw automatically binary trees in LaTeX in the file `draw_tree.py`.
#### `examples` folder
- Comparison of the tree generated by the scikit-learn implementation and our implementation of DecisionTreeClassifier in the file `comparison_with_sklearn.py`.
- Example on how to draw a binary tree in LaTeX in the file `draw_tree_in_tex.py`.
- Step by step pruning is depicted in the file `sequential_tree_pruning.py`.
- The bounds on the VC dimension of various tree structures is computed in the file `vcdim_computation.py`.
#### `experiments` folder
- A `main.py` file launches the experiments.
- A `pruning.py` files implements the two types of pruning (cross-validation or bound-based).
##### `datasets` folder
- A dataset manager is provided in the file `datasets.py`, which downloads and loads in memory automatically datasets from the UCI Machine Learning Repository. Datasets are stored in a folder named 'raw' once downloaded.
##### `scripts` folder
- A script to cross-validate the optimal value of 'r' in the distribution 'q_k' in the file `cross_validate_errors_prior.py`.
- A script to generate the Table 1 automatically from the results once the experiments are run in the file `process_all_results.py`
- A script to generate Table 2 to 20 automatically from the results once the experiments are run in the file `process_results_by_dataset.py`.
- A script to visualize the differences between the studied models in the file `tree_pruning_comparison.py`.
