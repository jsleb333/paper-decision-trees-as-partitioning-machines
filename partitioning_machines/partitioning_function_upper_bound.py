from scipy.special import binom, factorial
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.combinatorial.factorials import ff
from copy import copy


class PartitioningFunctionUpperBound:
    """
    This class computes the partioning function upper bound of Theorem 9 of the paper 'Decision trees as partitioning machines to characterize their generalization properties' by Leboeuf, LeBlanc and Marchand (2020).

    It implements an optimized version of the algorithm 1 of Appendix D by avoiding to compute the same value for the same subtree structures inside the tree multiple times by storing already computed values.
    """
    def __init__(self, tree, n_features, pre_computed_tables=None, loose=False):
        """
        Args:
            tree (Tree object): Tree structure for which to compute the bound.

            n_features (int): Number of real-valued features. Corresponds to the variable '\ell' in the paper.

            pre_computed_tables (Union[dict, None]): If the upper bound has already been computed for another tree, the computed tables of the PartitioningFunctionUpperBound object can be transfered here to speed up the process for current tree. The transfered table will be updated with any new value computed. If None, a table will be created from scratch. One can get the computed table by accessing the 'pfub_table' attribute.

            loose (bool): If loose is True, a looser but *much more* computationally efficient version of the bound is computed. In that case, no table is needed. This is the bound used in the experiments, as explained in section 6.2.
        """
        self.tree = tree
        self.n_features = n_features
        self.pfub_table = {} if pre_computed_tables is None else pre_computed_tables
        self.loose = loose

    def _compute_upper_bound_tight(self, tree, n_parts, n_examples, n_features):
        """
        Optimized implementation of Algorithm 1 of Appendix D of the paper.
        """
        c, m, l = n_parts, n_examples, n_features

        if c > m or c > tree.n_leaves:
            return 0
        elif c == m or c == 1 or m == 1:
            return 1
        elif m <= tree.n_leaves:
            return stirling(m, c)
        # Modification 1: Check first in the table if value is already computed.
        if tree not in self.pfub_table:
            self.pfub_table[tree] = {}
        if (c, m, l) not in self.pfub_table[tree]:
            N = 0
            min_k = tree.left_subtree.n_leaves
            max_k = m - tree.right_subtree.n_leaves
            for k in range(min_k, max_k+1):
                # Modification 2: Since c = 2 is the most common use case, we give an optimized version, writing explicitely the sum over a and b.
                if c == 2:
                    N +=  min(2*l, binom(m, k)) * (1
                            + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, l)
                            + 2 * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, l)
                            + 2 * self._compute_upper_bound_tight(tree.left_subtree, 2, k, l) * self._compute_upper_bound_tight(tree.right_subtree, 2, m-k, l)
                            )
                else:
                    N += min(2*l, binom(m, k)) * sum(
                        sum(
                            binom(a, c - b) * binom(b, c - a) * factorial(a + b - c) *
                            self._compute_upper_bound_tight(tree.left_subtree, a, k, l) *
                            self._compute_upper_bound_tight(tree.right_subtree, b, m-k, l)
                            for b in range(max(1,c-a), c+1)
                        )
                        for a in range(1, c+1)
                    )

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            # Modification 3: Add value to look up table.
            self.pfub_table[tree][n_parts, n_examples, n_features] = min(N, stirling(n_examples, n_parts))

        return self.pfub_table[tree][n_parts, n_examples, n_features]

    def _compute_upper_bound_loose(self, tree, n_parts, n_examples, n_features):
        """
        Looser but faster implementation of Algorithm 1 of Appendix D of the paper. The corresponding equation can be found at the end of section 6.2.
        """
        c, m, l = n_parts, n_examples, n_features

        if c > m or c > tree.n_leaves:
            return 0
        elif c == m or c == 1 or m == 1:
            return 1
        elif m <= tree.n_leaves:
            return stirling(m, c)
        if tree not in self.pfub_table:
            self.pfub_table[tree] = {}
        if (c, m, l) not in self.pfub_table[tree]:
            N = 0
            k_left = m - tree.right_subtree.n_leaves
            k_right = m - tree.left_subtree.n_leaves
            N = 0
            if c == 2:
                N +=  2*l * (1
                        + 2 * self._compute_upper_bound_loose(tree.left_subtree, 2, k_left, l)
                        + 2 * self._compute_upper_bound_loose(tree.right_subtree, 2, k_right, l)
                        + 2 * self._compute_upper_bound_loose(tree.left_subtree, 2, k_left, l) * self._compute_upper_bound_loose(tree.right_subtree, 2, k_right, l)
                        )
            else:
                N += 2*l * sum(
                    sum(
                        binom(a, c - b) * binom(b, c - a) * factorial(a + b - c) *
                        self._compute_upper_bound_loose(tree.left_subtree, a, k_left, l) *
                        self._compute_upper_bound_loose(tree.right_subtree, b, k_right, l)
                        for b in range(max(1,c-a), c+1)
                    )
                    for a in range(1, c+1)
                )
            N *= m - tree.n_leaves

            if tree.left_subtree == tree.right_subtree:
                N /= 2

            self.pfub_table[tree][c, m, l] = min(N, stirling(n_examples, n_parts))
        return self.pfub_table[tree][c, m, l]

    def __call__(self, n_examples, n_parts=2):
        """
        Args:
            n_examples (int): Number of examples. Corresponds to the variable 'm' in the paper.
            n_parts (int): Number of parts. Corresponds to the variable 'c' in the paper.
        """
        if self.loose:
            return self._compute_upper_bound_loose(self.tree, n_parts, n_examples, self.n_features)
        else:
            return self._compute_upper_bound_tight(self.tree, n_parts, n_examples, self.n_features)



def partitioning_function_upper_bound(tree, n_parts, n_examples, n_features):
    """
    Args:
        tree (Tree object): Tree structure for which to compute the bound.
        n_parts (int): Number of parts in the partitions. Corresponds to the variable 'c' in the paper.
        n_examples (int): Number of examples. Corresponds to the variable 'm' in the paper.
        n_features (int): Number of real-valued features. Corresponds to the variable '\ell' in the paper.
    """
    pfub = PartitioningFunctionUpperBound(tree, n_features)
    return pfub(n_examples, n_parts)


def growth_function_upper_bound(tree, n_features, n_classes=2, pre_computed_tables=None, loose=False):
    pfub = PartitioningFunctionUpperBound(tree, n_features, pre_computed_tables, loose=loose)
    def upper_bound(n_examples):
        max_range = min(n_classes, tree.n_leaves, n_examples)
        return sum(ff(n_classes, n)*pfub(n_examples, n) for n in range(1, max_range+1))
    return upper_bound