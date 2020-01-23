"""Functions to help with sampling trees."""

import pickle
import numpy as np
import random


def gen_samples(trees_x, trees_y, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    for tree_x, tree_y, label in zip(trees_x, trees_y, labels):
        x_nodes = []
        x_children = []
        y_nodes = []
        y_children = []

        # encode labels as one-hot vectors
        onehot_label = _onehot_multi(label, 5)
        # onehot_label = _onehot(label, )

        queue1 = [(tree_x, -1)]
        queue2 = [(tree_y, -1)]
        while queue1:
            x_node, x_parent_ind = queue1.pop()
            x_node_ind = len(x_nodes)

            # add children and the parent index to the queue
            queue1.extend([(child, x_node_ind) for child in x_node['children']])

            # create a list to store this node's children indices
            x_children.append([])

            # add this child to its parent's child list
            if x_parent_ind > -1:
                x_children[x_parent_ind].append(x_node_ind)
            x_nodes.append(vectors[vector_lookup[x_node['node']]])

        while queue2:
            y_node, y_parent_ind = queue2.pop()
            y_node_ind = len(y_nodes)

            # add children and the parent index to the queue
            queue2.extend([(child, y_node_ind) for child in y_node['children']])

            # create a list to store this node's children indices
            y_children.append([])

            # add this child to its parent's child list
            if y_parent_ind > -1:
                y_children[y_parent_ind].append(y_node_ind)
            y_nodes.append(vectors[vector_lookup[y_node['node']]])

        yield (x_nodes, x_children, y_nodes, y_children, onehot_label)


def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    x_nodes, x_children, y_nodes, y_children, labels = [], [], [], [], []
    samples = 0
    for x_n, x_c, y_n, y_c, l in gen:
        x_nodes.append(x_n)
        y_nodes.append(y_n)
        x_children.append(x_c)
        y_children.append(y_c)
        labels.append(l)
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(x_nodes, x_children, y_nodes, y_children, labels)
            x_nodes, x_children, y_nodes, y_children, labels = [], [], [], [], []
            samples = 0

    if x_nodes:
        yield _pad_batch(x_nodes, x_children, y_nodes, y_children, labels)


def _pad_batch(x_nodes, x_children, y_nodes, y_children, labels):
    if not x_nodes:
        return [], [], [], [], []
    max_x_nodes = max([len(x) for x in x_nodes])
    max_y_nodes = max([len(x) for x in y_nodes])

    max_x_children = max([len(x) for x in x_children])
    max_y_children = max([len(x) for x in y_children])

    feature_x_len = len(x_nodes[0][0])
    feature_y_len = len(y_nodes[0][0])
    x_child_len = max([len(c) for n in x_children for c in n])
    y_child_len = max([len(c) for n in y_children for c in n])

    x_nodes = [n + [[0] * feature_x_len] * (max_x_nodes - len(n)) for n in x_nodes]
    y_nodes = [n + [[0] * feature_y_len] * (max_y_nodes - len(n)) for n in y_nodes]

    # pad batches so that every batch has the same number of nodes
    x_children = [n + ([[]] * (max_x_children - len(n))) for n in x_children]
    y_children = [n + ([[]] * (max_y_children - len(n))) for n in y_children]

    # pad every child sample so every node has the same number of children
    x_children = [[c + [0] * (x_child_len - len(c)) for c in sample] for sample in x_children]
    y_children = [[c + [0] * (y_child_len - len(c)) for c in sample] for sample in y_children]

    return x_nodes, x_children, y_nodes, y_children, labels


def _onehot(i, total):
    # if i == 1:
    #     return [1.0, 0.0, 0.0, 0.0, 0.0]
    # elif i == 2:
    #     return [0.0, 1.0, 0.0, 0.0, 0.0]
    # elif i == 3:
    #     return [0.0, 0.0, 1.0, 0.0, 0.0]
    # elif i == 4:
    #     return [0.0, 0.0, 0.0, 1.0, 0.0]
    # elif i == 5:
    #     return [0.0, 0.0, 0.0, 0.0, 1.0]
    return [1.0 if j == i else 0.0 for j in range(total)]

def _onehot_multi(i, total):
    if i == 1:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif i == 2:
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    elif i == 3:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    elif i == 4:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    elif i == 5:
        return [0.0, 0.0, 0.0, 0.0, 1.0]
    # return [1.0 if (j+1) == i else 0.0 for j in range(total)]


def _onehot_v2(i):
    if i >= 1:
        return [0.0, 1.0]
    else:
        return [1.0, 0.0]


def bi_gen_samples(trees_x, trees_y, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    for tree_x, tree_y, label in zip(trees_x, trees_y, labels):
        x_nodes = []
        x_children = []
        y_nodes = []
        y_children = []

        # encode labels as one-hot vectors
        onehot_label = _onehot_v2(label)
        original_label = [label]

        queue1 = [(tree_x, -1)]
        queue2 = [(tree_y, -1)]
        while queue1:
            x_node, x_parent_ind = queue1.pop()
            x_node_ind = len(x_nodes)

            # add children and the parent index to the queue
            queue1.extend([(child, x_node_ind) for child in x_node['children']])

            # create a list to store this node's children indices
            x_children.append([])

            # add this child to its parent's child list
            if x_parent_ind > -1:
                x_children[x_parent_ind].append(x_node_ind)
            x_nodes.append(vectors[vector_lookup[x_node['node']]])

        while queue2:
            y_node, y_parent_ind = queue2.pop()
            y_node_ind = len(y_nodes)

            # add children and the parent index to the queue
            queue2.extend([(child, y_node_ind) for child in y_node['children']])

            # create a list to store this node's children indices
            y_children.append([])

            # add this child to its parent's child list
            if y_parent_ind > -1:
                y_children[y_parent_ind].append(y_node_ind)
            y_nodes.append(vectors[vector_lookup[y_node['node']]])

        yield (x_nodes, x_children, y_nodes, y_children, onehot_label, original_label)


def bi_batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    x_nodes, x_children, y_nodes, y_children, bi_labels, true_labels = [], [], [], [], [], []
    samples = 0
    for x_n, x_c, y_n, y_c, bi_l, true_l in gen:
        x_nodes.append(x_n)
        y_nodes.append(y_n)
        x_children.append(x_c)
        y_children.append(y_c)
        bi_labels.append(bi_l)
        true_labels.append(true_l)
        samples += 1
        if samples >= batch_size:
            yield _bi_pad_batch(x_nodes, x_children, y_nodes, y_children, bi_labels, true_labels)
            x_nodes, x_children, y_nodes, y_children, bi_labels, true_labels = [], [], [], [], [], []
            samples = 0

    if x_nodes:
        yield _bi_pad_batch(x_nodes, x_children, y_nodes, y_children, bi_labels, true_labels)


def _bi_pad_batch(x_nodes, x_children, y_nodes, y_children, labels, true_labels):
    if not x_nodes:
        return [], [], [], [], [], []
    max_x_nodes = max([len(x) for x in x_nodes])
    max_y_nodes = max([len(x) for x in y_nodes])

    max_x_children = max([len(x) for x in x_children])
    max_y_children = max([len(x) for x in y_children])

    feature_x_len = len(x_nodes[0][0])
    feature_y_len = len(y_nodes[0][0])
    x_child_len = max([len(c) for n in x_children for c in n])
    y_child_len = max([len(c) for n in y_children for c in n])

    x_nodes = [n + [[0] * feature_x_len] * (max_x_nodes - len(n)) for n in x_nodes]
    y_nodes = [n + [[0] * feature_y_len] * (max_y_nodes - len(n)) for n in y_nodes]

    # pad batches so that every batch has the same number of nodes
    x_children = [n + ([[]] * (max_x_children - len(n))) for n in x_children]
    y_children = [n + ([[]] * (max_y_children - len(n))) for n in y_children]

    # pad every child sample so every node has the same number of children
    x_children = [[c + [0] * (x_child_len - len(c)) for c in sample] for sample in x_children]
    y_children = [[c + [0] * (y_child_len - len(c)) for c in sample] for sample in y_children]

    return x_nodes, x_children, y_nodes, y_children, labels, true_labels

