
import os
import logging
import pickle
import tensorflow as tf
import numpy as np
from tbcnn import network as network
import tbcnn.sampling as sampling
from tbcnn.parameters import TEST_BATCH_SIZE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import pandas as pd
import sys
import json

def test_model(logdir, infile, embedfile):

    with open(infile, 'rb') as fh:
        datas = pickle.load(fh)

    labels = []
    for num in datas['label']:
        labels.append(int(num))

    code_x_trees = datas['tree_x']
    code_y_trees = datas['tree_y']

    with open(embedfile, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)
        num_feats = len(embeddings[0])

    n_classess = 5

    left_nodes_node, left_children_node, left_pooling_node = network.init_multi_net(num_feats)
    right_nodes_node, right_children_node, right_pooling_node = network.init_multi_net(num_feats)

    merge_node = tf.concat([left_pooling_node, right_pooling_node], -1)

    hidden_node = network.hidden_layer(merge_node, 100, 200)
    hidden_node = network.hidden_layer(hidden_node, 200, 400)
    hidden_node = network.hidden_layer(hidden_node, 400, 300)
    hidden_node = network.hidden_layer(hidden_node, 300, n_classess)

    out_node = network.out_layer(hidden_node)

    # init the graph
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return

    correct_labels = []
    predictions = []

    print('Computing testing accuracy...')
    for i, batch in enumerate(sampling.batch_samples(
            sampling.gen_samples(code_x_trees, code_y_trees, labels, embeddings, embed_lookup), TEST_BATCH_SIZE
    )):
        l_nodes, l_children, r_nodes, r_children, batch_labels = batch

        model_result = sess.run([out_node], feed_dict={
            left_nodes_node: l_nodes,
            left_children_node: l_children,
            right_nodes_node: r_nodes,
            right_children_node: r_children
        })
        predictions.append(np.argmax(model_result[0]))
        correct_labels.append(np.argmax(batch_labels))

    target_names = ["T1", "T2", "ST3", "MT3", "WT3/4"]
    print('Accuracy:', accuracy_score(correct_labels, predictions))
    print(np.unique(predictions))
    print(classification_report(correct_labels, predictions, target_names=target_names, labels=np.unique(correct_labels)))
    print(confusion_matrix(correct_labels, predictions))


def main():
    test_model('./clone_detection/model save/multi',
               './data/test/multi.pkl',
               './vec/data/vectors.pkl')


if __name__ == "__main__":
    main()