import numpy as np
from itertools import product

def prepare_data(labels, classes, patch_shape):
    """Preprocesses the data, by sampling subvolumes such that
    the distribution is uniform accorss the predefined set of
    classes. Classes are based on the proportion of active
    voxels in the subvolume.

    Args:
        labels: (x, y, z) numpy, int32
            the labeled volume (tiff stack)
        classes: list of floats
            the intervals to bin the subvolumes in
        patch_shape: (x, y, z) int tuple
            the shape of the training subvolumes
    """
    locations = [set() for _ in range(len(classes) - 1)]
    w, h, d, _ = patch_shape
    bounds = np.array(labels.shape) - np.array((w, h, d))
    for x, y, z in product(range(0, bounds[0], 3), range(0, bounds[1], 3), range(0, bounds[2], 1)):
        mx, my, mz = x + w // 2, y + h // 2, z + d // 2
        if labels[mx, my, mz] == 0: continue
        #Get data, and compute active voxel ratio
        data = labels[x:x + w, y:y + h, z:z + d]
        active = np.mean(data == labels[mx, my, mz])
        #Bin or resample
        index, = np.searchsorted(classes, [active], side = 'left')
        locations[index - 1].add((x, y, z))
    return equalize(locations)

def prepare_sparse_dense(labels, classes, patch_shape):
    locations = [set() for _ in range(len(classes) - 1)]
    w, h, d = labels.shape
    pw, ph, pd, _ = patch_shape
    X, Y, Z = np.nonzero(labels)
    for x, y, z in zip(X, Y, Z):
        if not (pw//2 <= x < w - pw//2): continue
        elif not (ph//2 <= y < h - ph//2): continue
        elif not (pd//2 <= z < d - pd//2): continue
        lw, lh, ld = np.array((x, y, z)) - np.array((pw, ph, pd), np.int32) // 2
        hw, hh, hd = np.array((x, y, z)) + np.array((pw, ph, pd), np.int32) // 2 + 1
        data = labels[lw:hw, lh:hh, ld:hd]
        active = np.mean(data == labels[x, y, z])
        index, = np.searchsorted(classes, [active], side = 'left')
        locations[index - 1].add((x - pw // 2, y -  ph // 2, z - pd // 2))
    return equalize(locations)


def equalize(locations):
    """Returns a version of locations, where each class
    has the same number of elements.

    Args:
        locations: list of set of (x, y, z) tuples
            list of locations per class
    Returns:
        new_locations: list of (x, y, z) tuples
            the equalized flatten list of locations
    """
    smallest = min([len(c) for c in locations])
    print([len(c) for c in locations])
    smallest=100
    new_locations = []
    for c in locations:
        new_locations.extend(list(c)[:smallest])
    indices = np.random.permutation(len(new_locations))
    return np.array(new_locations)[indices, :]


def batch(data, labels, subvol_shape, location):
    w, h, d, c = subvol_shape
    subvol_data = np.zeros((1, w, h, d, c), np.float32)
    subvol_labels = np.zeros((1, w, h, d, 1), np.float32)
    x, y, z = location
    subvol_data[0] = data[x:x+w, y:y+h, z:z+d]
    subvol_labels[0, :, :, :, 0] = labels[x:x+w, y:y+h, z:z+d] == labels[x+w//2, y+h//2, z+d//2]
    subvol_labels[0] = 0.9 * subvol_labels[0] + 0.05
    return subvol_data, subvol_labels

def mask(subvol_shape):
    w, h, d, c = subvol_shape
    subvol_mask = 0.05 * np.ones((1, w, h, d, 1), np.float32)
    subvol_mask[0, w // 2, h // 2, d //2, 0] = 0.95
    return subvol_mask


def get_data(volume, center, shape):
    w, h, d, c = shape
    subvol = np.zeros((1, w, h, d, c), np.float32)
    lw, lh, ld = center - np.array((w, h, d), np.int32) // 2
    hw, hh, hd = center + np.array((w, h, d), np.int32) // 2 + 1
    subvol[0] = volume[0, lw:hw, lh:hh, ld:hd]
    return subvol


def set_data(volume, center, subvol):
    _, w, h, d, c = subvol.shape
    lw, lh, ld = center -  np.array((w, h, d), np.int32) // 2
    hw, hh, hd = center +  np.array((w, h, d), np.int32) // 2 + 1
    volume[0, lw:hw, lh:hh, ld:hd] = subvol[0]

def get_new_locs(mask, delta, tmove):
    new_locs = []
    dx, dy, dz = delta
    _, x, y, z, c = np.array(mask.shape) // 2
    submask = mask[0, x - dx: x+dx+1, y-dy:y+dy+1,z-dz:z+dz+1, 0]
    xminus, xplus = submask[0, :, :], submask[2*dx,:,:]
    yminus, yplus = submask[:, 0, :], submask[:,2*dy,:]
    zminus, zplus = submask[:, :, 0], submask[:,:,2*dz]

    i, j = np.unravel_index(xminus.argmax(), xminus.shape)
    if xminus[i, j] >= tmove:
        new_locs.append((-dx, i - dy, j - dz, xminus[i, j]))
    i, j = np.unravel_index(xplus.argmax(), xplus.shape)
    if xplus[i, j] >= tmove:
        new_locs.append((dx, i - dy, j - dz, xplus[i, j]))
    i, j = np.unravel_index(yminus.argmax(), yminus.shape)
    if yminus[i, j] >= tmove:
        new_locs.append((i-dx,- dy, j - dz, yminus[i, j]))
    i, j = np.unravel_index(yplus.argmax(), yplus.shape)
    if yplus[i, j] >= tmove:
        new_locs.append((i-dx, dy, j - dz, yplus[i, j]))
    i, j = np.unravel_index(zminus.argmax(), zminus.shape)
    if zminus[i, j] >= tmove:
        new_locs.append((i-dx, j - dy,- dz, zminus[i, j]))
    i, j = np.unravel_index(zplus.argmax(), zplus.shape)
    if zplus[i, j] >= tmove:
        new_locs.append((i-dx, j - dy, dz, zplus[i, j]))

    new_locs = sorted(new_locs, key=lambda l:l[-1], reverse = True)
    new_locs = np.array(new_locs)[:,0:3] if new_locs else []
    return new_locs


def random_loc(volume_shape, patch_shape):
    """Samples a random (x, y, z) in coordinate, such that a
    subvolume of shape patch_shape can be safely extracted
    from the output coordinate.

    Args:
        volume_shape: (x, y, z) int tuple
            the shape of the training volume
        patch_shape: (x, y, z) int tuple
            the shape of the training subvolumes
    Returns:
        loc: (x, y, z) int tuple
            a random location in the volume
    """
    #Compute random location
    high = [volume_shape[i] - patch_shape[i] for i in range(3)]
    loc = [np.random.randint(0, h) for h in high]
    return loc





