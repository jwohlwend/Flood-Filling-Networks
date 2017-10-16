from rand import adapted_rand
import numpy as np
import yaml
from tifffile import imread
import sys

def run(config):
    image_path, labels_path, raw_path = config['SAVE_PRED'], config['LABELS_DENSE'], config['DATA']
    seg, labels, raw = imread(image_path)[2:12], imread(labels_path)[-16:-6], imread(raw_path)[-16:-6,:,:,0]
    pixel_error = np.mean(seg != labels)
    rand_error = adapted_rand(seg, labels)
    print("pixel error: {}, rand error {}".format(pixel_error, rand_error))

if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    run(config)