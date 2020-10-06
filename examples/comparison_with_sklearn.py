from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as DTCsklearn
import numpy as np

from partitioning_machines import Tree, draw_decision_tree, decision_tree_to_tikz, tree_from_sklearn_decision_tree
from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion

import python2latex as p2l

dataset = load_iris()
X = dataset.data
y = [dataset.target_names[i] for i in dataset.target]

# Our implementation
dtc_ours = DecisionTreeClassifier(gini_impurity_criterion)
dtc_ours.fit(X, y)

# Scikit-learn implementation
dtc_sklearn = DTCsklearn(random_state=42)
dtc_sklearn.fit(X, y)
dtc_sklearn_conv = tree_from_sklearn_decision_tree(dtc_sklearn)

dtc_ours_pic = decision_tree_to_tikz(dtc_ours,
                                     dtc_ours.label_encoder.labels,
                                     min_node_distance=1.8,
                                     level_distance=1.8,
                                     show_impurity=True,
                                     show_n_examples_by_label=True)
dtc_ours_pic.body.insert(0, r'\node at (0,1) {Our implementation};')

dtc_sklearn_pic = decision_tree_to_tikz(dtc_sklearn_conv,
                                        dtc_sklearn.classes_,
                                        min_node_distance=1.8,
                                        level_distance=1.8,
                                        show_impurity=True,
                                        show_n_examples_by_label=True)
dtc_sklearn_pic.body.insert(0, r'\node at (0,1) {Scikit-learn implementation};')

doc = p2l.Document('comparison_with_sklearn', doc_type='standalone', border='1cm')
doc.add_package('tikz')
del doc.packages['geometry']
doc.add_to_preamble('\\usetikzlibrary{shapes}')

doc += dtc_ours_pic
doc += '\\hspace{2cm}'
doc += dtc_sklearn_pic

doc.build()
