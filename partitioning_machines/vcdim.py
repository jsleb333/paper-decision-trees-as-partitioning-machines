from partitioning_machines import PartitioningFunctionUpperBound


def vcdim_upper_bound(tree, n_features):
    """
    Computes an upper bound on the VC dimension of a tree knowing the number of available features. Implements Algorithm 2 of Appendix D of 'Decision trees as partitioning machines to characterize their generalization properties' by Leboeuf, LeBlanc and Marchand (2020).

    Args:
        tree (Tree object): Tree structure for which to compute the bound.
        n_features (int): Number of real-valued features. Corresponds to the variable '\ell' in the paper.
    """
    if tree.is_leaf():
        return 1

    m = tree.n_leaves + 1
    pfub = PartitioningFunctionUpperBound(tree, n_features)
    while pfub(m, 2) == 2**(m-1)-1:
        m += 1

    return m - 1


def vcdim_lower_bound(tree, n_features):
    """
    Computes a lower bound on the VC dimension of a tree knowing the number of available features. Implements the algorithm of Figure 7 of Yildiz (2015) with the base case replaced by our exact value for stumps. This is the algorithm used in Figure 3 in Appendix D of the paper.

    Args:
        tree (Tree object): Tree structure for which to compute the bound.
        n_features (int): Number of real-valued features. Corresponds to the variable '\ell' in the paper.
    """
    if tree.is_leaf():
        return 1
    if tree.is_stump():
        return vcdim_upper_bound(tree, n_features) # Upper bound is exact for stumps
    else:
        return vcdim_lower_bound(tree.left_subtree, n_features) + vcdim_lower_bound(tree.right_subtree, n_features)
