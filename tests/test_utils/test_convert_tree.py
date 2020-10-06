from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

from partitioning_machines import Tree
from partitioning_machines.utils import tree_from_sklearn_decision_tree


def test_tree_from_sklearn_decision_tree_with_actual_tree():
    X, y = load_iris(return_X_y=True)
    sklearn_tree = DecisionTreeClassifier()
    sklearn_tree = sklearn_tree.fit(X, y)
    
    tree_from_sklearn = tree_from_sklearn_decision_tree(sklearn_tree)

    