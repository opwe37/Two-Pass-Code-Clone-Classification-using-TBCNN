"""Parse trees from a data source."""
import javalang
import sys
import pickle
import random
import pandas as pd
from collections import defaultdict
from utils import get_children
import argparse


def parse(args):
    """Parse trees with the given arguments."""
    print ('Loading pickle file')

    with open(args.infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)

    print('Pickle file load finished')

    tree_samples = []

    def merge(data_path, sources):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)

        # 클론이 아닌 코드 조각 쌍
        label_0 = pairs.loc[pairs['label'] == 0, :].sample(frac=1).reset_index(drop=True)

        # 클론 코드 조각 쌍
        label_1 = pairs.loc[pairs['label'] == 1, :].sample(frac=1).reset_index(drop=True)
        label_2 = pairs.loc[pairs['label'] == 2, :].sample(frac=1).reset_index(drop=True)
        label_3 = pairs.loc[pairs['label'] == 3, :].sample(frac=1).reset_index(drop=True)
        label_4 = pairs.loc[pairs['label'] == 4, :].sample(frac=1).reset_index(drop=True)
        label_5 = pairs.loc[pairs['label'] == 5, :].sample(frac=1).reset_index(drop=True)

        # train & test data split _ binary
        label_0 = label_0[0:18315]
        train_l0 = label_0.sample(frac=0.3, random_state=200)
        test_l0 = label_0.drop(train_l0.index)

        label_1 = label_1[0:3663]
        train_l1 = label_1.sample(frac=0.3, random_state=200)
        test_l1 = label_1.drop(train_l1.index)

        label_2 = label_2[0:3663]
        train_l2 = label_2.sample(frac=0.3, random_state=200)
        test_l2 = label_2.drop(train_l2.index)

        label_3 = label_3[0:3663]
        train_l3 = label_3.sample(frac=0.3, random_state=200)
        test_l3 = label_3.drop(train_l3.index)

        label_4 = label_4[0:3663]
        train_l4 = label_4.sample(frac=0.3, random_state=200)
        test_l4 = label_4.drop(train_l4.index)

        label_5 = label_5[0:3663]
        train_l5 = label_5.sample(frac=0.3, random_state=200)
        test_l5 = label_5.drop(train_l5.index)

        # train & test data split _ multi-class
        # label_0 = label_0[0:3663]
        # train_l0 = label_0.sample(frac=0.3, random_state=200)
        # test_l0 = label_0.drop(train_l0.index)
        #
        # label_1 = label_1[0:3663]
        # train_l1 = label_1.sample(frac=0.3, random_state=200)
        # test_l1 = label_1.drop(train_l1.index)
        #
        # label_2 = label_2[0:3663]
        # train_l2 = label_2.sample(frac=0.3, random_state=200)
        # test_l2 = label_2.drop(train_l2.index)
        #
        # label_3 = label_3[0:3663]
        # train_l3 = label_3.sample(frac=0.3, random_state=200)
        # test_l3 = label_3.drop(train_l3.index)
        #
        # label_4 = label_4[0:3663]
        # train_l4 = label_4.sample(frac=0.3, random_state=200)
        # test_l4 = label_4.drop(train_l4.index)
        #
        # label_5 = label_5[0:3663]
        # train_l5 = label_5.sample(frac=0.3, random_state=200)
        # test_l5 = label_5.drop(train_l5.index)

        # train = label_0.append(label_1).append(label_2).append(label_3).append(label_4).append(label_5)
        # test = label_0.append(label_1).append(label_2).append(label_3).append(label_4).append(label_5)

        train = train_l0.append(train_l1).append(train_l2).append(train_l3).append(train_l4).append(train_l5)
        test = test_l0.append(test_l1).append(test_l2).append(test_l3).append(test_l4).append(test_l5)

        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

        train_df = pd.merge(train, sources, how='left', left_on='id1', right_on='id')
        train_df = pd.merge(train_df, sources, how='left', left_on='id2', right_on='id')
        train_df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        train_df.dropna(inplace=True)

        test_df = pd.merge(test, sources, how='left', left_on='id1', right_on='id')
        test_df = pd.merge(test_df, sources, how='left', left_on='id2', right_on='id')
        test_df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        test_df.dropna(inplace=True)

        return train_df, test_df

    for tree_id, tree in zip(data_source['id'], data_source['code']):
        root = tree
        sample, size = _traverse_tree(root)

        # if size > args.maxsize or size < args.minsize:
        #     continue

        datum = {'id': tree_id, 'tree': sample}
        tree_samples.append(datum)

    sample_sources = pd.DataFrame(tree_samples, columns=['id', 'tree'])
    train_data, test_data = merge('C:/Users/user/Desktop/영빈/2019/tbcnn_cloneDetection/data/bcb_pair_ids.pkl', sample_sources)

    # create a list of unique labels in the data
    print('Dumping sample')
    train_data.to_pickle('C:/Users/user/Desktop/영빈/2019/tbcnn_cloneDetection/data/train/binary.pkl')
    test_data.to_pickle('C:/Users/user/Desktop/영빈/2019/tbcnn_cloneDetection/data/test/binary.pkl')
    print('dump finished')


def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),
        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)

        children = list(get_children(current_node))
        queue.extend(children)
        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes


def _name(node):
    return node.__class__.__name__


tree_parser = argparse.ArgumentParser(
    description="Sample trees or nodes from a crawler file.",
)
tree_parser.add_argument('--infile', type=str, help='Data file to sample from')
tree_parser.add_argument('--outfile', type=str, help='File to store samples in')
tree_parser.add_argument(
    '--maxsize', type=int, default=10000,
    help='Ignore trees with more than --maxsize nodes'
)
tree_parser.add_argument(
    '--minsize', type=int, default=100,
    help='Ignore trees with less than --minsize nodes'
)

args = tree_parser.parse_args()
parse(args)

