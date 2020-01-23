
import os
import numpy as np
import pickle
import tensorflow as tf
from tbcnn import network as network
import tbcnn.sampling as sampling
from tbcnn.parameters import LEARN_RATE, EPOCHS, BATCH_SIZE, CHECKPOINT_EVERY


def train_model(logdir, infile, embedfile, epochs=EPOCHS, with_drop_out=1):

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

    n_classess = 2

    left_nodes_node, left_children_node, left_pooling_node = network.init_binary_net(num_feats)
    right_nodes_node, right_children_node, right_pooling_node = network.init_binary_net(num_feats)

    merge_node = tf.concat([left_pooling_node, right_pooling_node], -1)

    hidden_node = network.hidden_layer(merge_node, 100, 100)
    hidden_node = network.hidden_layer(hidden_node, 100, 150)
    hidden_node = network.hidden_layer(hidden_node, 150, n_classess)

    out_node = network.out_layer(hidden_node)

    labels_node, loss_node = network.loss_layer(hidden_node, n_classess)

    optimizer = tf.train.AdamOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)

    # init the graph
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))  # config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    checkfile = os.path.join(logdir, 'cnn_tree.ckpt')
    steps = 0

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(sampling.bi_batch_samples(
                sampling.bi_gen_samples(code_x_trees, code_y_trees, labels, embeddings, embed_lookup), BATCH_SIZE
        )):
            l_nodes, l_children, r_nodes, r_children, batch_labels, origin_labels = batch

            if not l_nodes:
                continue  # don't try to train on an empty batch

            _, err = sess.run(
                [train_step, loss_node],
                    feed_dict={
                        left_nodes_node: l_nodes,
                        right_nodes_node: r_nodes,
                        left_children_node: l_children,
                        right_children_node: r_children,
                        labels_node: batch_labels
                    }
            )

            print('Epoch:', epoch, 'Step:', steps, 'Loss:', err)

            if steps % CHECKPOINT_EVERY == 0:
                # save state so we can resume later
                saver.save(sess, os.path.join(checkfile), steps)
                print('Checkpoint saved.')

            steps += 1
        steps = 0


train_model('./clone_detection/model save',
            './data/train/binary.pkl',
            './vec/data/vectors.pkl')
