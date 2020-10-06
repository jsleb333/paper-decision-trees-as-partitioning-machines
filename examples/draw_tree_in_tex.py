"""
In this script, we show a minimal working example on how to draw a tree in tex with python2latex.
"""
from sklearn.datasets import load_iris
import numpy as np

from partitioning_machines import Tree, draw_decision_tree, decision_tree_to_tikz, gini_impurity_criterion
from partitioning_machines import DecisionTreeClassifier

import python2latex as p2l

dataset = load_iris()
X = dataset.data
y = [dataset.target_names[i] for i in dataset.target]
dtc = DecisionTreeClassifier(gini_impurity_criterion)
dtc.fit(X, y)

tikzpicture_object = decision_tree_to_tikz(dtc, dtc.label_encoder.labels)

print(tikzpicture_object.build()) # Converts object to string usable in tex file

# Draw tree in LaTeX if pdflatex is available
draw_decision_tree(dtc)
