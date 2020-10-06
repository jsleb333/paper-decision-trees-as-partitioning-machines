from partitioning_machines.decision_tree_classifier import DecisionTreeClassifier, _DecisionTree
from partitioning_machines.decision_tree_classifier import _DecisionTree as Tree


def tree_from_sklearn_decision_tree(sklearn_tree):
    """
    Returns the Tree object corresponding to the scikit-learn DecisionTreeClassifier class.

    Args:
        sklearn_tree (DecisionTreeClassifier object): Learned tree needed to be converted.
    """
    dtc = DecisionTreeClassifier()
    dtc.tree = _build_tree_from_sklearn_tree(sklearn_tree.tree_)
    return dtc

def _build_tree_from_sklearn_tree(sklearn_tree, current_node=0):
    children_left, children_right = sklearn_tree.children_left, sklearn_tree.children_right
    impurity_score = sklearn_tree.impurity[current_node]
    n_examples_by_label = sklearn_tree.value[current_node].reshape(-1)
    
    if children_left[current_node] == -1 and children_right[current_node] == -1:
        subtree = Tree(impurity_score, n_examples_by_label)
    else:
        subtree = Tree(
            impurity_score,
            n_examples_by_label,
            rule_threshold=sklearn_tree.threshold[current_node],
            rule_feature=sklearn_tree.feature[current_node],
            left_subtree=_build_tree_from_sklearn_tree(sklearn_tree, children_left[current_node]),
            right_subtree=_build_tree_from_sklearn_tree(sklearn_tree, children_right[current_node])
        )
    return subtree
