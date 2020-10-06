import os, sys
import numpy as np
from pytest import fixture
from copy import deepcopy


from partitioning_machines import Tree, DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines.utils.draw_tree import *
from partitioning_machines.utils.draw_tree import _init_position, _find_extremal_position_by_layer, _find_largest_overlap, _deoverlap_position, _shift_tree


@fixture
def trees():
    trees = [Tree()]
    trees.append(Tree(deepcopy(trees[0]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[1]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[1]), deepcopy(trees[1])))
    trees.append(Tree(deepcopy(trees[2]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[2]), deepcopy(trees[1])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[0])))
    trees.append(Tree(deepcopy(trees[2]), deepcopy(trees[2])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[1])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[2])))
    trees.append(Tree(deepcopy(trees[3]), deepcopy(trees[3])))
    return trees


class TestComputeNodesPosition:
    def test_init_position(self, trees):
        _init_position(trees[1])
        assert trees[1].position == 0
        assert trees[1].left_subtree.position == -1
        assert trees[1].right_subtree.position == 1
        
    def test_find_extremal_position_by_layer_max_mode(self, trees):
        tree = trees[2]
        _init_position(tree)
        assert _find_extremal_position_by_layer(tree, 'max') == [0, 1, 0]
        assert _find_extremal_position_by_layer(tree.left_subtree, 'max') == [-1, 0]
        assert _find_extremal_position_by_layer(tree.left_subtree.left_subtree, 'max') == [-2]
        assert _find_extremal_position_by_layer(tree.left_subtree.right_subtree, 'max') == [0]
        assert _find_extremal_position_by_layer(tree.right_subtree, 'max') == [1]

        tree = trees[10]
        _init_position(tree)
        assert _find_extremal_position_by_layer(tree, 'max') == [0, 1, 2, 3]
        assert _find_extremal_position_by_layer(tree.left_subtree, 'max') == [-1, 0, 1]
        assert _find_extremal_position_by_layer(tree.left_subtree.left_subtree, 'max') == [-2, -1]
        assert _find_extremal_position_by_layer(tree.left_subtree.left_subtree.left_subtree, 'max') == [-3]

    def test_find_extremal_position_by_layer_min_mode(self, trees):
        tree = trees[2]
        _init_position(tree)
        assert _find_extremal_position_by_layer(tree, 'min') == [0, -1, -2]
        assert _find_extremal_position_by_layer(tree.left_subtree, 'min') == [-1, -2]
        assert _find_extremal_position_by_layer(tree.left_subtree.left_subtree, 'min') == [-2]
        assert _find_extremal_position_by_layer(tree.left_subtree.right_subtree, 'min') == [0]
        assert _find_extremal_position_by_layer(tree.right_subtree, 'min') == [1]

    def test_find_largest_overlap(self, trees):
        tree = trees[3]
        _init_position(tree)
        assert _find_largest_overlap(tree) == 0
        assert _find_largest_overlap(tree.left_subtree) == -2
        assert _find_largest_overlap(tree.right_subtree) == -2

        tree = trees[10]
        _init_position(tree)
        assert _find_largest_overlap(tree) == 2
        assert _find_largest_overlap(tree.left_subtree) == 0


    def test_shift_tree(self, trees):
        tree = trees[3]
        _init_position(tree)
        _shift_tree(tree.left_subtree, -1)
        assert [t.position for t in tree] == [0, -2, -3, -1, 1, 0, 2]

        tree = trees[10]
        _init_position(tree)
        _shift_tree(tree.left_subtree, -2)
        _shift_tree(tree.right_subtree, 2)
        assert [t.position for t in tree] == [0, -3, -4, -5, -3, -2, -3, -1, 3, 2, 1, 3, 4, 3, 5]

    def test_deoverlap_position(self, trees):
        tree = trees[3]
        _init_position(tree)
        _deoverlap_position(tree)
        assert [t.position for t in tree] == [0, -2, -3, -1, 2, 1, 3]

        tree = trees[7]
        _init_position(tree)
        _deoverlap_position(tree)
        assert [t.position for t in tree] == [0, -2, -3, -4, -2, -1, 2, 1, 0, 2, 3]

        tree = trees[10]
        _init_position(tree)
        _deoverlap_position(tree)
        assert [t.position for t in tree] == [0, -4, -6, -7, -5, -2, -3, -1, 4, 2, 1, 3, 6, 5, 7]


def test_tree_struct_to_tikz(trees):
    pic = tree_struct_to_tikz(trees[9])
    print(pic.build())

def test_decision_tree_to_tikz(trees):
    X = np.array([[1,2,3,4],
                  [3,4,7,3],
                  [6,7,3,2],
                  [5,5,2,6],
                  [9,1,9,5]
                  ])
    y = np.array([0,1,0,2,2])
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X, y)
    pic = decision_tree_to_tikz(dtc, dtc.label_encoder.labels)

def test_draw_tree_structure(trees):
    draw_tree_structure(trees[9], show_pdf=False)
    os.remove('./Tree_of_height_3.log')
    os.remove('./Tree_of_height_3.tex')
    os.remove('./Tree_of_height_3.aux')
    os.remove('./Tree_of_height_3.pdf')

def test_draw_decision_tree():
    X = np.array([[1,2,3,4],
                  [3,4,7,3],
                  [6,7,3,2],
                  [5,5,2,6],
                  [9,1,9,5]
                  ])
    y = ['Class 0', 'Class 1', 'Class 0', 'Class 2', 'Class 2']
    dtc = DecisionTreeClassifier(gini_impurity_criterion)
    dtc.fit(X, y)
    draw_decision_tree(dtc, show_pdf=False)
    os.remove('./Tree_of_height_2.log')
    os.remove('./Tree_of_height_2.tex')
    os.remove('./Tree_of_height_2.aux')
    os.remove('./Tree_of_height_2.pdf')
