"""
train.py
"""

import tensorflow as tf
from tensorflow.contrib.layers import optimize_loss
import numpy as np
from tifffile import imread
from utils import *
from model import flood_filling_network
from sklearn.metrics import roc_auc_score
from queue import Queue
import threading
import yaml
import matplotlib.pyplot as plt


def run(config):
    """Trains the flood filling network model.

    Args:
        config: dict
            configuration variables (see config.yml)
    """
   # Define graph input
    fw, fh, fd, fc = config['FOV_SHAPE']
    x = tf.placeholder(tf.float32, [None, fw, fh, fd, 1])
    m = tf.placeholder(tf.float32, [None, fw, fh, fd, 1])
    y = tf.placeholder(tf.float32, [None, fw, fh, fd, 1])
    wloss = tf.placeholder(tf.float32)
    # Define model
    inp = tf.concat([x, m], axis = -1)
    out = flood_filling_network(inp, config['NUM_CONV_MODULES'], config['KERNEL_SIZE'])
    # Define loss and optimizer
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = out, labels = y)
    loss = tf.reduce_mean(loss)
    #loss = tf.reduce_mean(tf.reshape(loss, [-1, fw*fh*fd]), axis=1)
    # Weight loss using the proportion of non active voxels
    #loss = wloss * loss
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = optimize_loss(loss, global_step, learning_rate=config['LEARNING_RATE'], optimizer='Adam')
    # Get new masks
    out = tf.nn.sigmoid(out)
    # Create saver
    saver = tf.train.Saver()
    # Create session
    with tf.Session() as sess:
        # Initialize graph
        sess.run(tf.global_variables_initializer())
        # Load data
        data_path, labels_path = config['DATA'], config['LABELS']
        data, labels = imread(data_path), imread(labels_path)
        data = (data - np.mean(data)) / 255.0
        # Split training and testing
        split = int(config['TRAINING'] * data.shape[0])
        train_data, train_labels = data[:split, :, :, 0:1], labels[:split]
        test_data, test_labels = data[split:, :, :, 0:1], labels[split:]
        # Reshape data
        train_data, train_labels = np.moveaxis(train_data, 0, 2), np.moveaxis(train_labels, 0, 2)
        test_data, test_labels = np.moveaxis(test_data, 0, 2), np.moveaxis(test_labels, 0, 2)
        locations = prepare_data(train_labels, config['CLASSES'], config['SUBVOL_SHAPE'])
        # Train
        def train_function():
            # TODO: Better termination condition, e.g. using a `max_steps` counter.
            for epoch in range(config['EPOCHS']):
                indices = np.random.permutation(len(locations))
                for location in locations[indices]:
                    # random_loc(train_data.shape[:-1], config['SUBVOL_SHAPE'][:-1])
                    # Load training examples
                    subvol_data, subvol_labels = batch(train_data, train_labels, config['SUBVOL_SHAPE'],location)
                    weight = 1.0 - np.mean(subvol_labels)
                    subvol_mask = mask(config['SUBVOL_SHAPE'])
                    n, w, h, d, c = subvol_data.shape
                    # Create FOV dicts, add center locations
                    V = {(w // 2, h // 2, d // 2)}
                    queue = Queue()
                    queue.put([w // 2, h // 2, d // 2])
                    # Compute upper and lower bounds
                    upper = [w - fw // 2, h - fh // 2, d - fd // 2]
                    lower = [fw // 2, fh //2, fd // 2]
                    while not queue.empty():
                        # Get new list of FOV locations
                        current_loc = np.array(queue.get(), np.int32)
                        # Center around FOV
                        fov_data = get_data(subvol_data, current_loc, config['FOV_SHAPE'])
                        fov_labels = get_data(subvol_labels, current_loc, config['FOV_SHAPE'])
                        fov_mask = get_data(subvol_mask, current_loc, config['FOV_SHAPE'])
                        # Add merging of old and new mask
                        _, (out_mask,), l = sess.run([optimizer, out, loss], feed_dict={x:fov_data, y:fov_labels, m:fov_mask, wloss:weight})
                        if np.any(fov_labels[0] < 0.5):
                            train_auc = roc_auc_score(fov_labels[0].flatten() > 0.5, out_mask.flatten())
                            print("Step:", epoch, "| Loss:", l,"| AUC:", train_auc)
                        set_data(subvol_mask, current_loc, out_mask)
                        # Compute new locations
                        new_locations = get_new_locs(out_mask, config['DELTA'], config['TMOVE'])
                        for new in new_locations:
                            new = np.array(new, np.int32) + current_loc
                            bounds = [lower[j] <= new[j] < upper[j] for j in range(3)]
                            stored_loc = tuple([new[i] // config['DELTA'][i] for i in range(3)])
                            if all(bounds) and stored_loc not in V:
                                V.add(stored_loc)
                                queue.put(new)

        # Create multiple threads to run `train_function()` in parallel
        train_threads = []
        for _ in range(config['BATCH_SIZE']):
          train_threads.append(threading.Thread(target=train_function))

        # Start the threads, and block on their completion.
        for t in train_threads:
          t.start()
        for t in train_threads:
          t.join()
        # Save results
        saver.save(sess, config["SAVE"])



if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    run(config)
