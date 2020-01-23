import pandas as pd
import javalang
from javalang.ast import Node
from AST_Tree import ASTNode, BlockNode
import sys
sys.setrecursionlimit(10000)

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'#node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def get_sequence(node, sequence):
    token, children = get_token(node), get_children(node)
    sequence.append(token)

    for child in children:
        get_sequence(child, sequence)

    if token in ['ForStatement', 'WhileStatement', 'DoStatement','SwitchStatement', 'IfStatement']:
        sequence.append('End')

