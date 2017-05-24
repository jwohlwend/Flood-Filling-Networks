"""
train.py
"""

import tensorflow as tf
import numpy as np
from tiffile import imread
from utils import prepare_data
from model import flood_filling_network
import yaml


def run(config):
    """Trains the flood filling network model.

    Args:
        config: dict
            configuration variables (see config.yml)
    """
    # Define graph input
    fw, fh, fd = config['FOV_SHAPE']
    x = tf.placeholder(tf.float32, [None, fw, fh, fd, 1])
    m = tf.placeholder(tf.float32, [None, fw, fh, fd, 1])
    y = tf.placeholder(tf.float32, [None, fw, fh, fd, 1])
    # Define model
    inp = tf.concat([x, m], axis = -1)
    out = flood_filling_network(inp, config['NUM_CONV_MODULES'])
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = out, label = y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=config['LEARNING_RATE']).minimize(loss)
    # Create session
    with tf.Session() as sess:
        # Initialize graph
        sess.run(tf.global_variables_initializer())
        # Load data
        data_path, labels_path = config['DATA'], config['LABELS']
        data, labels = imread(data_path), imread(labels_path)
        # Split training and testing
        split = int(config['TRAINING'] * data.shape[0])
        train_data, train_labels = data[:split, :, :, 0], labels[:split]
        test_data, test_labels = data[split:, :, :, 0], labels[split:]
        # Reshape data
        train_data, train_labels = np.moveaxis(train_data, 0, 2), np.moveaxis(train_labels, 0, 2)
        test_data, test_labels = np.moveaxis(test_data, 0, 2), np.moveaxis(test_labels, 0, 2)
        # Load training examples
        locations = prepare_data(train_labels, config['CLASSES'], config['PATCH_SHAPE'], config['NUM_EXAMPLES'])
        locations = np.array_split(locations, locations.shape[0] // config['BATCH_SIZE'])
        # Train
        for batch in locations:
            w, h, d = config['PATCH_SHAPE']
            # Data
            batch_data = [train_data[i:i + w, j: j + h, k: k + d] for (i, j, k) in batch]
            # Targets, make binary
            batch_target = [train_labels[i:i + w, j: j + h, k: k + d] for (i, j, k) in batch]
            batch_target = [0.9 * (vol == vol[w // 2, h // 2, d // 2]) + 0.05 for vol in batch_target]
            # Initial masks
            batch_mask = [0.05 * np.ones_like(vol) for vol in tar]
            for mask in batch_mask:
                mask[w // 2, h // 2, d // 2] = 0.95
            # Center around FOV
            ow, oh, od = (w - fw) // 2, (h - fh) // 2, (d - fd) // 2
            fov_data = [vol[ow:ow + fw, oh:oh + fh, od:od + fd] for vol in batch_data]
            fov_labels = [vol[ow:ow + fw, oh:oh + fh, od:od + fd] for vol in batch_target]
            fov_masks = [vol[ow:ow + fw, oh:oh + fh, od:od + fd] for vol in batch_mask]
            #Create FOV dicts, add center locations
            all_visited = [{(w // 2, h // 2, d // 2)} for _ in range(len(batch))]
            #Add merging of old and new mask
            _, out_masks = s.run([optimizer, out], feed_dict={x:fov_data, y:fov_labels, m:fov_masks})
            #Compute new FOVs






if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    run(config)
