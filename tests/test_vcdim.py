from partitioning_machines import vcdim_upper_bound, vcdim_lower_bound, Tree


def test_vcdim_upper_bound():
    leaf = Tree()
    stump = Tree(leaf, leaf)
    tree = Tree(stump, leaf)

    assert vcdim_upper_bound(leaf, 10) == 1
    assert vcdim_upper_bound(stump, 10) == 6
    assert vcdim_upper_bound(tree, 10) == 16


def test_vcdim_lower_bound():
    leaf = Tree()
    stump = Tree(leaf, leaf)
    tree = Tree(stump, leaf)

    assert vcdim_lower_bound(leaf, 10) == 1
    assert vcdim_lower_bound(stump, 10) == 6
    assert vcdim_lower_bound(tree, 10) == 7
