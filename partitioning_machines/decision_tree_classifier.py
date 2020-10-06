import numpy as np

from partitioning_machines import Tree
from partitioning_machines import OneHotEncoder


class _DecisionTree(Tree):
    def __init__(self,
                 impurity_score,
                 n_examples_by_label,
                 rule_threshold=None,
                 rule_feature=None,
                 left_subtree=None,
                 right_subtree=None,
                 parent=None):
        super().__init__(left_subtree=left_subtree,
                         right_subtree=right_subtree,
                         parent=parent)

        self.impurity_score = impurity_score
        self.n_examples_by_label = n_examples_by_label
        self.label = np.zeros(n_examples_by_label.shape[0], dtype=int)
        self.label[np.argmax(n_examples_by_label)] = 1
        self.rule_threshold = rule_threshold
        self.rule_feature = rule_feature

    @property
    def n_examples(self):
        return int(self.n_examples_by_label.sum())

    @property
    def n_errors(self):
        """
        Returns the number of errors made by the subtree on the training dataset.
        """
        if self.is_leaf():
            return self.n_examples - np.max(self.n_examples_by_label)
        else:
            return self.left_subtree.n_errors + self.right_subtree.n_errors

    def predict(self, x):
        if self.is_leaf():
            return self.label
        else:
            if x[self.rule_feature] < self.rule_threshold:
                return self.left_subtree.predict(x)
            else:
                return self.right_subtree.predict(x)

    def predict_proba(self, x):
        if self.is_leaf():
            return self.n_examples_by_label / self.n_examples
        else:
            if x[self.rule_feature] < self.rule_threshold:
                return self.left_subtree.predict_proba(x)
            else:
                return self.right_subtree.predict_proba(x)

    def is_pure(self):
        return (self.n_examples_by_label == self.n_examples).any()


class DecisionTreeClassifier:
    def __init__(self,
                 impurity_criterion=None,
                 optimization_mode='min',
                 max_n_leaves=None,
                 max_depth=None,
                 min_examples_per_leaf=1):
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
        self.max_n_leaves = max_n_leaves if max_n_leaves is not None else np.infty
        self.max_depth = max_depth if max_depth is not None else np.infty
        self.min_examples_per_leaf = min_examples_per_leaf
        self.tree = None

    def fit(self, X, y, X_idx_sorted=None, verbose=False):
        self.n_examples, self.n_features = X.shape
        self.label_encoder = OneHotEncoder(y)
        encoded_y, _ = self.label_encoder.encode_labels(y)

        if X_idx_sorted is None:
            X_idx_sorted = np.argsort(X, 0)

        self._init_tree(encoded_y, self.n_examples)

        splitter = Splitter(X, encoded_y, self.impurity_criterion, self.optimization_mode, self.min_examples_per_leaf, verbose=verbose)

        possible_splits = [] # List of splits that can be produced.

        first_split = splitter.split(self.tree, X_idx_sorted)
        if first_split:
            possible_splits.append(first_split)

        while possible_splits and self.tree.n_leaves < self.max_n_leaves:
            best_split = possible_splits[0]
            for split in possible_splits:
                if self.optimization_mode == 'min':
                    if best_split.impurity_score > split.impurity_score:
                        best_split = split
                elif self.optimization_mode == 'max':
                    if best_split.impurity_score < split.impurity_score:
                        best_split = split

            if best_split.split_makes_gain():
                best_split.apply_split()
                if verbose:
                    print(f'New split with rule x_{best_split.rule_feature} < {best_split.rule_threshold}.')

                if self.tree.n_leaves < self.max_n_leaves and self.tree.height < self.max_depth:
                    X_idx_sorted_left, X_idx_sorted_right = best_split.compute_split_X_idx_sorted()

                    left_split = splitter.split(best_split.leaf.left_subtree, X_idx_sorted_left)
                    if left_split:
                        possible_splits.append(left_split)

                    right_split = splitter.split(best_split.leaf.right_subtree, X_idx_sorted_right)
                    if right_split:
                        possible_splits.append(right_split)

            possible_splits.remove(best_split)

        return self

    def _init_tree(self, encoded_y, n_examples):
        n_examples_by_label = np.sum(encoded_y, axis=0)
        self.tree = _DecisionTree(self.impurity_criterion(n_examples_by_label/n_examples),
                         n_examples_by_label)

    def predict(self, X):
        encoded_prediction = np.array([self.tree.predict(x) for x in X])
        return self.label_encoder.decode_labels(encoded_prediction)

    def predict_proba(self, X):
        return np.array([self.tree.predict_proba(x) for x in X])

    def compute_pruning_coefficients(self, prune_objective):
        """
        Computes and assigns a pruning coefficient to every internal node of the tree. The sorted list of coefficients is returned.

        Args:
            prune_objective (callable): Receives a Tree object and outputs a coefficient based on the performance of the tree. Will not be called with a leaf.

        Returns the list of pruning coefficients in increasing order.
        """
        pruning_coefs = []
        for subtree in self.tree:
            if not subtree.is_leaf():
                subtree.pruning_coef = prune_objective(subtree)
                pruning_coefs.append(subtree.pruning_coef)
        pruning_coefs.sort()

        return pruning_coefs

    def prune_tree(self, pruning_coef_threshold, pruning_objective=None):
        """
        Prunes the tree by replacing each subtree that have a pruning coefficient less than or equal to 'pruning_coef_threshold' by a leaf. Does so by inspecting each subtree to find the 'pruning_coef' attribute and comparing to the threshold. The 'pruning_coef' attribute is set beforehand when 'pruning_objective' is an appropriate callable, or by calling beforehand the method 'compute_pruning_coefficients'.

        Comparing to Algorithm 3 in Appendix E of the paper 'Decision trees as partitioning machines to characterize their generalization properties' by Leboeuf, LeBlanc and Marchand (2020), the current method only implements the 'for' loop inside the 'while' loop. The rest of the algorithm must be implemented by the user. This separation was needed so that the current method can also be used with other types of pruning algorithms.

        Args:
            pruning_coef_threshold (float): Threshold the pruning coefficient must satisfy.
            pruning_objective (callable): Will be used to compute the pruning coefficients if provided. Used by the 'compute_pruning_coefficients' method. If None, it assumes the 'compute_pruning_coefficients' method has already been called and subtrees possesses the 'pruning_coef' attribute.

        Returns: (int) the number of internal nodes pruned.
        """
        if pruning_objective is not None:
            self.compute_pruning_coefficients(pruning_objective)

        subtrees_to_remove = []
        n_nodes_before = self.tree.n_nodes

        for subtree in self.tree:
            if not subtree.is_leaf():
                if subtree.pruning_coef <= pruning_coef_threshold:
                    self._prune_subtree(subtree)

        return n_nodes_before - self.tree.n_nodes

    def _prune_subtree(self, subtree):
        subtree.left_subtree = None
        subtree.right_subtree = None
        self.tree.update_tree()


class Splitter:
    """
    Class that stores important data necessary to make an actual Split object. Its main purpose is to unclutter the 'fit' method of the 'DecisionTreeClassifier' class and to provide by the same token a "lazy evaluation" scheme for the splitting algorithm.
    """
    def __init__(self, X, y, impurity_criterion, optimization_mode, min_examples_per_leaf=1, verbose=False):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
        self.min_examples_per_leaf = min_examples_per_leaf
        self.verbose = verbose

    def split(self, leaf, X_idx_sorted):
        return Split(leaf, X_idx_sorted, self, self.verbose)

class Split:
    """
    Class that examines all possible split of a node and chooses the best one. Does not apply the split and does not preorder the data except if explicitely called by the appropriate methods. Also maintains useful information on the split, such as the number of examples sent to the left and to the right, as well as the gain made by the split.
    """
    def __init__(self, leaf, X_idx_sorted, splitter, verbose=False):
        self.leaf = leaf
        self.X_idx_sorted = X_idx_sorted
        self.splitter = splitter
        self.n_examples, self.n_features = X_idx_sorted.shape

        self.X = splitter.X
        self.y = splitter.y
        self.n_classes = self.y.shape[1]
        self.impurity_criterion = splitter.impurity_criterion
        self.optimization_mode = splitter.optimization_mode
        self.min_examples_per_leaf = splitter.min_examples_per_leaf
        self.verbose = verbose

        self.validity = self._find_best_split()

    def _find_best_split(self):
        """
        Finds the best split according to the impurity criterion and respecting the minimum number of examples per leaf. This is done by sorting the examples by the values of their features (in a matrix 'X_idx_sorted'), and by examining the gain made by moving the split exactly one examples at the time simultaneously for all features (using vectorized operations).
        """
        n_examples_by_label = self.leaf.n_examples_by_label

        if self.verbose:
            print(f'\nSplitting node with {self.n_examples} examples.')

        if self.leaf.is_pure():
            if self.verbose:
                print(f'No split because leaf is pure: {n_examples_by_label}')
            return False

        if self.n_examples < 2*self.min_examples_per_leaf:
            if self.verbose:
                print(f'No split because minimum number of examples per leaf not met.')
            return False

        n_examples_left = self.min_examples_per_leaf - 1
        n_examples_right = self.n_examples - n_examples_left

        n_examples_by_label_left = np.zeros((self.n_features, self.n_classes))
        for i in range(n_examples_left):
            n_examples_by_label_left += self.y[self.X_idx_sorted[i]]
        n_examples_by_label_right = n_examples_by_label - n_examples_by_label_left

        sign = 1 if self.optimization_mode == 'min' else -1
        self.impurity_score = sign * np.infty

        a_rule_has_been_found = False

        for x_idx in self.X_idx_sorted[n_examples_left:-self.min_examples_per_leaf]:
            n_examples_left += 1
            n_examples_right -= 1
            transfered_labels = self.y[x_idx]
            n_examples_by_label_left += transfered_labels
            n_examples_by_label_right -= transfered_labels
            tmp_impurity_score_by_feature = self._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right)

            x_idx_right = self.X_idx_sorted[n_examples_left]
            forbidden_features_mask = self._find_forbideen_features(x_idx, x_idx_right)
            tmp_impurity_score_by_feature[forbidden_features_mask] = sign * np.infty
            tmp_feature, tmp_impurity_score = self.argext(tmp_impurity_score_by_feature)

            if (sign*tmp_impurity_score < sign*self.impurity_score):
                rule_threshold_idx_left = x_idx[tmp_feature]
                rule_threshold_idx_right = x_idx_right[tmp_feature]
                self.rule_feature = tmp_feature
                self.rule_threshold = (self.X[rule_threshold_idx_left, tmp_feature] +
                                       self.X[rule_threshold_idx_right, tmp_feature])/2
                self.impurity_score = tmp_impurity_score
                self.n_examples_by_label_left = n_examples_by_label_left[tmp_feature].copy()
                self.n_examples_by_label_right = n_examples_by_label_right[tmp_feature].copy()
                a_rule_has_been_found = True

        if self.verbose:
            if a_rule_has_been_found:
                print(f'Rule found with feature {self.rule_feature} and threshold {self.rule_threshold}.')
            else:
                print('No rule has been found.')

        return a_rule_has_been_found

    def _find_forbideen_features(self, x_idx_left, x_idx_right):
        return np.isclose(np.array([x[idx] for x, idx in zip(self.X.T, x_idx_left)]),
                          np.array([x[idx] for x, idx in zip(self.X.T, x_idx_right)]))

    def __bool__(self):
        return self.validity

    @property
    def n_examples_left(self):
        return self.n_examples_by_label_left.sum(dtype=int)

    @property
    def n_examples_right(self):
        return self.n_examples_by_label_right.sum(dtype=int)

    def argext(self, arr):
        if self.optimization_mode == 'min':
            extremum = np.argmin
        elif self.optimization_mode == 'max':
            extremum = np.argmax
        extremum_idx = extremum(arr)
        return extremum_idx, arr[extremum_idx]

    def _split_impurity_criterion(self, n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right):
        return (self._weighted_impurity_criterion(n_examples_by_label_left, n_examples_left) +
                self._weighted_impurity_criterion(n_examples_by_label_right, n_examples_right)) / \
                (n_examples_left + n_examples_right)

    def _weighted_impurity_criterion(self, n_examples_by_label, n_examples):
        return self.impurity_criterion(n_examples_by_label/n_examples) * n_examples

    def split_makes_gain(self):
        if self.optimization_mode == 'min':
            return self.impurity_score < self.leaf.impurity_score
        elif self.optimization_mode == 'max':
            return self.impurity_score > self.leaf.impurity_score

    def apply_split(self):
        impurity_left = self.impurity_criterion(self.n_examples_by_label_left/self.n_examples_left)
        left_leaf = _DecisionTree(impurity_left,
                                  self.n_examples_by_label_left.copy(),
                                  parent=self.leaf)

        impurity_right = self.impurity_criterion(self.n_examples_by_label_right/self.n_examples_right)
        right_leaf = _DecisionTree(impurity_right,
                                  self.n_examples_by_label_right.copy(),
                                  parent=self.leaf)
        self.leaf.left_subtree = left_leaf
        self.leaf.right_subtree = right_leaf
        self.leaf.rule_threshold = self.rule_threshold
        self.leaf.rule_feature = self.rule_feature
        self.leaf.update_tree()

    def compute_split_X_idx_sorted(self):
        X_idx_sorted_left = np.zeros((self.n_examples_left, self.n_features), dtype=int)
        X_idx_sorted_right = np.zeros((self.n_examples_right, self.n_features), dtype=int)

        left_x_pos = np.zeros(self.n_features, dtype=int)
        right_x_pos = np.zeros(self.n_features, dtype=int)

        for x_idx in self.X_idx_sorted:
            for feat, idx in enumerate(x_idx):
                if self.X[idx, self.rule_feature] < self.rule_threshold:
                    X_idx_sorted_left[left_x_pos[feat], feat] = idx
                    left_x_pos[feat] += 1
                else:
                    X_idx_sorted_right[right_x_pos[feat], feat] = idx
                    right_x_pos[feat] += 1

        return X_idx_sorted_left, X_idx_sorted_right


def gini_impurity_criterion(frac_examples_by_label):
    axis = 1 if len(frac_examples_by_label.shape) > 1 else 0
    return np.sum(frac_examples_by_label * (1 - frac_examples_by_label), axis=axis)

def entropy_impurity_criterion(frac_examples_by_label):
    axis = 1 if len(frac_examples_by_label.shape) > 1 else 0
    return -np.sum(frac_examples_by_label * np.log(frac_examples_by_label), axis=axis)

def margin_impurity_criterion(frac_examples_by_label):
    axis = 1 if len(frac_examples_by_label.shape) > 1 else 0
    return 1 - np.max(frac_examples_by_label, axis=axis)

def breiman_alpha_pruning_objective(tree):
    """
    From Breiman (1984) Chapter 3. We want to solve for α in the following equality:
        R(t) + α = R(T_t) + α * n_leaves(T_t),
    where:
        t is the root node of the subtree 'tree'.
        R(t) is the number of errors made by the node t if the node was replaced with a leaf, divided by the total number of examples.
        T_t is the subtree 'tree' taking root at node t
        R(T_t) is the number of errors made by all leaves of the subtree T_t divided by the total number of examples.
        n_leaves(T_t) is the number of leaves of the subtree T_t.
    """
    node_n_errors = tree.n_examples - np.max(tree.n_examples_by_label)
    return (node_n_errors - tree.n_errors) / ( tree.root.n_examples * (tree.n_leaves - 1) )

def modified_breiman_pruning_objective_factory(n_features):
    def modified_breiman_pruning_objective(tree):
        d = lambda n_leaves: n_leaves * np.log(n_features * n_leaves)
        complexity = lambda n_leaves, n_examples: d(n_leaves) * np.log(n_examples / d(n_leaves))
        node_n_errors = tree.n_examples - np.max(tree.n_examples_by_label)
        m = tree.root.n_examples
        return (node_n_errors - tree.n_errors) / (complexity(tree.n_leaves, m) - complexity(1, m))
    return modified_breiman_pruning_objective
