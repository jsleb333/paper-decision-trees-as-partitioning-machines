"""
In this script, we compute the VCdim upper bound for the first 11 non-equivalent binary trees.
"""
from partitioning_machines import Tree, vcdim_upper_bound


leaf = Tree() # tree 1
stump = Tree(leaf, leaf) # tree 2
tree3 = Tree(stump, leaf)
tree4 = Tree(stump, stump)
tree5 = Tree(tree3, leaf)
tree6 = Tree(tree4, leaf)
tree7 = Tree(tree3, stump)
tree8 = Tree(tree3, tree3)
tree9 = Tree(tree4, stump)
tree10 = Tree(tree4, tree3)
tree11 = Tree(tree4, tree4)

trees = [
    leaf,
    stump,
    tree3,
    tree4,
    tree5,
    tree6,
    tree7,
    tree8,
    tree9,
    tree10,
    tree11
]

n_features = 10

for tree in trees:
    print(vcdim_upper_bound(tree, n_features))
