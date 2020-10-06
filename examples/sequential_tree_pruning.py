"""
In this script, we prune sequentially a tree using Breiman's algorithm and we draw each pruned tree in LaTeX.
"""
from sklearn.datasets import load_iris
import numpy as np
import python2latex as p2l

from partitioning_machines import Tree, decision_tree_to_tikz, gini_impurity_criterion, breiman_alpha_pruning_objective
from partitioning_machines import DecisionTreeClassifier


dataset = load_iris()
X = dataset.data
y = [dataset.target_names[i] for i in dataset.target]
dtc = DecisionTreeClassifier(gini_impurity_criterion)
dtc.fit(X, y)

sequence_of_trees = [decision_tree_to_tikz(dtc, dtc.label_encoder.labels)]

pruning_coefs = dtc.compute_pruning_coefficients(breiman_alpha_pruning_objective)

for pruning_coef_threshold in pruning_coefs:
    n_nodes_removed = dtc.prune_tree(pruning_coef_threshold)
    if n_nodes_removed > 0:
        sequence_of_trees.append(decision_tree_to_tikz(dtc, dtc.label_encoder.labels))

doc = p2l.Document('sequential_tree_pruning', doc_type='standalone', border='1cm')
doc.add_package('tikz')
del doc.packages['geometry']
doc.add_to_preamble('\\usetikzlibrary{shapes}')
for tree in sequence_of_trees:
    doc += tree
    doc += r'\hspace{1cm}'
doc.build()

