import numpy as np

def prepare_data(labels, classes, patch_shape, num_examples):
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
        num_examples: int
            total number of examples to sample before resampling
    """
    assert(num_examples < np.prod(data.shape - patch_shape))
    locations = [set() for _ in range(len(classes))]
    examples = 0
    while examples < num_examples:
        x, y, z = random_loc(data.shape, patch_shape)
        w, h, d = patch_shape
        mx, my, mz = x + w // 2, y + h // 2, z + d // 2
        #Get data, and compute active voxel ratio
        data = labels[x:x + w, y:y + h, z:z + d]
        active = np.mean(data == labels[mx, my, mz])
        #Bin or resample
        index, = np.searchsorted(classes, [active], side = 'left')
        if (x, y, z) not in locations[index]:
            locations[index].add((x, y, z))
            examples += 1
    return equalize(locations)

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
    loc = [np.random.randint(0, h) for l, h in zip(low, high)]
    return loc

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
    new_locations = []
    for c in locations:
        new_locations.extend(list(c)[:smallest])
    indices = np.random.permutations(len(new_locations))
    return np.array(new_locations)[indices, :]
