"""Parse nodes from a given data source."""

import javalang
import pickle
from collections import defaultdict
from utils import get_children


def parse(args):
    """Parse nodes with the given args."""
    print ('Loading pickle file')
    
    with open(args.infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
    print ('Pickle load finished')

    node_counts = defaultdict(int)
    samples = []

    has_capacity = lambda x: args.per_node < 0 or node_counts[x] < args.per_node
    can_add_more = lambda: args.limit < 0 or len(samples) < args.limit

    for item in data_source['code']:
        root = item
        new_samples = [
            {
                'node': _name(root),
                'parent': None,
                'children': [_name(x) for x in get_children(root)]
            }
        ]
        gen_samples = lambda x: new_samples.extend(_create_samples(x))
        _traverse_tree(root, gen_samples)
        for sample in new_samples:
            if has_capacity(sample['node']):
                samples.append(sample)
                node_counts[sample['node']] += 1
            if not can_add_more:
                break
        if not can_add_more:
            break
    print('dumping sample')

    with open(args.outfile, 'wb') as file_handler:
        pickle.dump(samples, file_handler)
        file_handler.close()

    print('Sampled node counts:')
    print(node_counts)
    print('Total: %d' % sum(node_counts.values()))

def _create_samples(node):
    """Convert a node's children into a sample points."""
    samples = []
    for child in get_children(node):
        sample = {
            "node": _name(child),
            "parent": _name(node),
            "children": [_name(x) for x in get_children(child)]
        }
        samples.append(sample)

    return samples

def _traverse_tree(tree, callback):
    """Traverse a tree and execute the callback on every node."""

    queue = [tree]
    while queue:
        current_node = queue.pop(0)
        children = list(get_children(current_node))
        queue.extend(children)
        callback(current_node)

def _name(node):
    """Get the name of a node."""
    return node.__class__.__name__


import argparse

node_parser = argparse.ArgumentParser(
    description="Sample trees or nodes from a crawler file.",
)
node_parser.add_argument('--infile', type=str, help='Data file to sample from')
node_parser.add_argument('--outfile', type=str, help='File to store samples in')
node_parser.add_argument(
    '--per-node', type=int, default=-1,
    help='Sample up to a maxmimum number for each node type'
)
node_parser.add_argument(
    '--limit', type=int, default=-1,
    help='Maximum number of samples to store.'
)

args = node_parser.parse_args()
parse(args)
