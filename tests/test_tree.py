from pytest import fixture, raises
from copy import deepcopy, copy

from partitioning_machines import Tree


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

@fixture
def overlapping_trees(trees):
    for tree in trees:
        tree._init_position()

    return trees


class TestTree:
    def test_is_leaf(self, trees):
        assert trees[0].is_leaf()
        assert trees[1].left_subtree.is_leaf()
        assert trees[1].right_subtree.is_leaf()

    def test_is_stump(self, trees):
        assert trees[1].is_stump()
        assert trees[2].left_subtree.is_stump()

    def test_n_leaves(self, trees):
        assert trees[2].n_leaves == 3
        assert trees[2].left_subtree.n_leaves == 2

    def test_n_nodes(self, trees):
        assert trees[0].n_nodes == 0
        assert trees[2].n_nodes == 2
        assert trees[2].left_subtree.n_nodes == 1

    def test__eq__(self, trees):
        leaf = Tree()
        assert leaf == Tree()

        tree2 = Tree(Tree(leaf, leaf), leaf)
        tree2_mirror = Tree(leaf, Tree(leaf, leaf))
        assert tree2 == tree2_mirror

        assert trees[1] == trees[2].left_subtree

    def test__eq__wrong_type(self, trees):
        with raises(ValueError):
            trees[0] == 1

    def test_hash(self, trees):
        assert [hash(tree) for tree in trees[:7]] == [0,2,5,8,9,12,13]
        assert hash(trees[1].left_subtree) == 0
        assert hash(trees[2].left_subtree) == 2

    def test_height(self, trees):
        assert [tree.height for tree in trees[:7]] == [0,1,2,2,3,3,3]

    def test_repr(self, trees):
        assert [repr(tree) for tree in trees[:7]] == ['Tree()',
                                                  'Tree(Tree(), Tree())',
                                                  'Tree of height 2',
                                                  'Tree of height 2',
                                                  'Tree of height 3',
                                                  'Tree of height 3',
                                                  'Tree of height 3']

    def test_depth(self, trees):
        assert trees[5].depth == 0
        assert trees[5].left_subtree.depth == 1
        assert trees[5].left_subtree.left_subtree.depth == 2

    def test__len__(self, trees):
        assert [len(tree) for tree in trees[:7]] == [1, 3, 5, 7, 7, 9, 9]

    def test_is(self, trees):
        assert trees[1] is trees[1]
        assert trees[1] is not trees[1].left_subtree

    def test_in_list(self, trees):
        list_of_subtrees = [trees[2]]
        # assert trees[2] in list_of_subtrees
        assert trees[2].left_subtree not in list_of_subtrees

    def test__contains__(self, trees):
        assert trees[2].right_subtree in trees[2]
        assert trees[3].left_subtree in trees[3]
        assert not trees[3] in trees[3].right_subtree

    def test_iter(self, trees):
        for subtree in trees[0]:
            assert subtree is trees[0]

        assert len([subtree for subtree in trees[1]]) == 3
        assert len([subtree for subtree in trees[2]]) == 5

    def test_replace_leaf_by_stump(self, trees):
        tree = Tree(Tree(), Tree())
        tree.left_subtree.replace_subtree(deepcopy(tree))
        assert tree == trees[2]
        assert [t.height for t in tree] == [2, 1, 0, 0, 0]
        assert [t.depth for t in tree] == [0, 1, 2, 2, 1]

    def test_replace_stump_by_leaf(self, trees):
        tree = Tree(Tree(), Tree())
        tree.replace_subtree(Tree())
        assert tree.is_leaf()

    def test_replace_leaf_by_leaf(self, trees):
        tree = Tree(Tree(), Tree())
        tree.left_subtree.replace_subtree(Tree())
        assert tree == trees[1]

    def test_split_leaf(self, trees):
        tree = Tree()
        tree.split_leaf()
        assert tree == trees[1]

    def test_remove_subtree(self, trees):
        trees[2].left_subtree.remove_subtree()
        assert trees[2] == trees[1]

    def test_copy(self, trees):
        object_passed_by_reference_by_default = {'a':1}
        for subtree in trees[3]:
            subtree.some_reference = object_passed_by_reference_by_default
        shallow_copy_of_tree3 = copy(trees[3])
        assert shallow_copy_of_tree3 is not trees[3]
        assert shallow_copy_of_tree3 == trees[3]
        assert all(copy_of_subtree is not subtree for copy_of_subtree, subtree in zip(shallow_copy_of_tree3, trees[3]))
        assert all(copy_of_subtree.some_reference is subtree.some_reference for copy_of_subtree, subtree in zip(shallow_copy_of_tree3, trees[3]))

    def test_deepcopy(self, trees):
        object_passed_by_reference_by_default = {'a':1}
        for subtree in trees[3]:
            subtree.some_reference = object_passed_by_reference_by_default
        deepcopy_of_tree3 = deepcopy(trees[3])
        assert deepcopy_of_tree3 is not trees[3]
        assert deepcopy_of_tree3 == trees[3]
        assert all(deepcopy_of_subtree is not subtree for deepcopy_of_subtree, subtree in zip(deepcopy_of_tree3, trees[3]))
        assert all(deepcopy_of_subtree.some_reference is not subtree.some_reference for deepcopy_of_subtree, subtree in zip(deepcopy_of_tree3, trees[3]))

    def test_path_from_root(self, trees):
        tree = trees[-1]
        subtree = tree.left_subtree.right_subtree
        assert subtree.path_from_root() == ['left', 'right']
        
    def test_follow_path(self, trees):
        assert trees[9].left_subtree.left_subtree is trees[9].follow_path(['left', 'left'])
        assert trees[9].left_subtree.right_subtree is trees[9].follow_path(['left', 'right'])
        subtree = trees[9].left_subtree.right_subtree
        assert subtree is subtree.root.follow_path(subtree.path_from_root())