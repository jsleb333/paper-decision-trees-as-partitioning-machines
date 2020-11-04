import numpy as np
try:
    import python2latex as p2l
except ImportError:
    raise ImportError("The drawing of trees rely on the package python2latex. Please install it with 'pip install python2latex'.")
try:
    import seaborn as sns
except ImportError:
    sns = None
    from warnings import warn
    warn('Consider installing the seaborn package to have access to easy color coding in the pictures.')

from partitioning_machines import Tree

def compute_nodes_position(tree):
    _init_position(tree)
    _deoverlap_position(tree)

def _init_position(tree, position=0):
    tree.position = position
    if not tree.is_leaf():
        _init_position(tree.left_subtree, position - 1)
        _init_position(tree.right_subtree, position + 1)

def _deoverlap_position(tree):
    if tree.is_leaf():
        return
    else:
        _deoverlap_position(tree.left_subtree)
        _deoverlap_position(tree.right_subtree)
        overlap = _find_largest_overlap(tree)
        if overlap >= -1:
            _shift_tree(tree.left_subtree, -overlap/2 - 1)
            _shift_tree(tree.right_subtree, overlap/2 + 1)

def _find_largest_overlap(tree):
    rightest_position = _find_extremal_position_by_layer(tree.left_subtree, 'max')
    leftest_position = _find_extremal_position_by_layer(tree.right_subtree, 'min')
    overlaps = [r - l for l, r in zip(leftest_position, rightest_position)]
    return max(overlaps)

def _find_extremal_position_by_layer(tree, mode):
    extremal_position_by_layer = []
    subtrees_in_layer = [tree]
    while subtrees_in_layer:
        subtrees_in_next_layer = []
        extremal_pos = subtrees_in_layer[0].position
        for subtree in subtrees_in_layer:
            if mode == 'max' and subtree.position > extremal_pos:
                extremal_pos = subtree.position
            elif mode == 'min' and subtree.position < extremal_pos:
                extremal_pos = subtree.position
            if not subtree.is_leaf():
                subtrees_in_next_layer.append(subtree.left_subtree)
                subtrees_in_next_layer.append(subtree.right_subtree)
        extremal_position_by_layer.append(extremal_pos)
        subtrees_in_layer = subtrees_in_next_layer

    return extremal_position_by_layer

def _shift_tree(tree, shift):
    tree.position += shift
    if not tree.is_leaf():
        _shift_tree(tree.left_subtree, shift)
        _shift_tree(tree.right_subtree, shift)


def tree_struct_to_tikz(tree, min_node_distance=1.3, level_distance=1.6, node_size=.6):
    pic = p2l.TexEnvironment('tikzpicture')
    pic.options += f"""leaf/.style={{draw, diamond, minimum width={node_size}cm, minimum height={2*node_size}cm, inner sep=0pt}}""",
    pic.options += f"""internal/.style={{draw, circle, minimum width={node_size}cm, inner sep=0pt}}""",

    compute_nodes_position(tree)

    for node, subtree in enumerate(tree):
        style = 'leaf' if subtree.is_leaf() else 'internal'
        pic += f'\\node[{style}](node{node}) at ({min_node_distance*subtree.position/2:.3f}, {-level_distance*subtree.depth:.3f}) {{}};'
        subtree.node_id = node

    for subtree in tree:
        if not subtree.is_leaf():

            if subtree.left_subtree.is_leaf():
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.left_subtree.node_id}.north);'
            else:
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.left_subtree.node_id});'

            if subtree.right_subtree.is_leaf():
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.right_subtree.node_id}.north);'
            else:
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.right_subtree.node_id});'

    return pic


def decision_tree_to_tikz(decision_tree,
                          classes,
                          min_node_distance=1.45,
                          level_distance=1.6,
                          label_color_palette=None,
                          node_size=.6,
                          show_rule=True,
                          feature_names=None,
                          show_impurity=False,
                          show_n_examples_by_label=False):
    pic = p2l.TexEnvironment('tikzpicture')
    pic.options += f"""leaf/.style={{draw, diamond, minimum width={node_size}cm, minimum height={2*node_size}cm, inner sep=0pt}}""",
    pic.options += f"""internal/.style={{draw, rectangle, minimum width={node_size}cm, inner sep=4pt}}""",

    if label_color_palette is None and sns is not None:
        label_color_palette = sns.color_palette(n_colors=len(classes))

    if label_color_palette is not None:
        colors = [p2l.Color(*color) for color in label_color_palette]
        for color in colors:
            pic.preamble.extend(color.preamble)
    else:
        colors = []

    compute_nodes_position(decision_tree.tree)

    for node, subtree in enumerate(decision_tree.tree):
        if subtree.is_leaf():
            style = 'leaf'
            leaf_label = np.argmax(subtree.label)
            if colors:
                style += f', fill={colors[leaf_label]}'
            node_label = str(subtree.n_examples)
        else:
            style = 'internal'
            node_label = []
            if show_rule:
                feat = f"x_{{{subtree.rule_feature}}}" if feature_names is None else f"\textrm{{{feature_names[subtree.rule_feature]}}}"
                node_label.append(f'${feat} \le {subtree.rule_threshold:.2f}$')
            if show_impurity:
                node_label.append(f'Impurity: ${subtree.impurity_score:.2f}$')
            if show_n_examples_by_label:
                node_label.append('$[' + ', '.join(str(int(n)) for n in subtree.n_examples_by_label) + ']$')
            node_label = ' \\\\ '.join(node_label)

        color = ''
        pic += f'\\node[{style}, align=center](node{node}) at ({min_node_distance*subtree.position/2:.3f}, {-level_distance*subtree.depth:.3f}) {{{node_label}}};'

        subtree.node_id = node

    for subtree in decision_tree.tree:
        if not subtree.is_leaf():

            if subtree.left_subtree.is_leaf():
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.left_subtree.node_id}.north);'
            else:
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.left_subtree.node_id});'

            if subtree.right_subtree.is_leaf():
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.right_subtree.node_id}.north);'
            else:
                pic += f'\\draw (node{subtree.node_id}) -- (node{subtree.right_subtree.node_id});'

    legend_start_y = -level_distance * (decision_tree.tree.height + .75)
    for i, label in enumerate(classes):
        pic += (f'\\node[inner sep=0pt, minimum height=10pt, minimum width=10pt, fill={colors[i]}, draw]'
                f'(legend{i}) at (0, {legend_start_y - 0.4233*i:.3f}) {{}};')
        pic += f'\\node[anchor=west] at (legend{i}.east) {{{label}}};'

    return pic


def draw_tree_structure(tree, show_pdf=True):
    doc = p2l.Document(str(tree).replace(' ', '_'), options=('varwidth',), doc_type='standalone', border='1cm')
    doc.add_package('tikz')
    del doc.packages['geometry']
    doc.add_to_preamble('\\usetikzlibrary{shapes}')
    doc += tree_struct_to_tikz(tree)
    doc.build(show_pdf=show_pdf)


def draw_decision_tree(decision_tree, show_pdf=True):
    doc = p2l.Document(str(decision_tree.tree).replace(' ', '_'), options=('varwidth',), doc_type='standalone', border='1cm')
    doc.add_package('tikz')
    del doc.packages['geometry']
    doc.add_to_preamble('\\usetikzlibrary{shapes}')
    doc += decision_tree_to_tikz(decision_tree, decision_tree.label_encoder.labels)
    doc.build(show_pdf=show_pdf)
