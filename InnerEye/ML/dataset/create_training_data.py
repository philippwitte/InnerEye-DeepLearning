# Split data into training data sets

import matplotlib.pyplot as plt
import h5py, sys, os, datetime, csv
import numpy as np

#######################################################################################################################

# Compute indices of patches
def compute_indices(shape, patchsize, stride):

    num_patches = int((shape - 2*int(patchsize/2)) / stride)
    indices = []
    
    for i in range(num_patches):
        i_start = i*stride
        i_end = i*stride + patchsize
        indices.append((i_start, i_end))
    indices.append((shape - patchsize, shape))
    
    return indices


# Extract patches and write as hdf5 files
def create_and_write_patches(shape, patchsize, stride, X, Y, path, basename, 
    num_chars=None, image_label='image', segmentation_labels=None, xtype='float32', ytype='uint8'):

    # Convert to correct types
    X = X.astype(xtype)
    Y = Y.astype(ytype)

    # Indices for each dimension
    idz = compute_indices(shape[0], patchsize, stride)
    idx = compute_indices(shape[1], patchsize, stride)
    idy = compute_indices(shape[2], patchsize, stride)

    # Length of file ID
    if num_chars is None:
        num_train = len(idz)*len(idx)*len(idy)
        num_chars = len(str(num_train)) + 1

    count = 0
    fields = ['subject', 'filePath', 'channel']
    rows = []
    num_class = np.max(Y) - np.min(Y) + 1
    for i, z in enumerate(idz):
        for j, x in enumerate(idx):
            for k, y in enumerate(idy):
                x_curr = X[z[0]:z[1], x[0]:x[1], y[0]:y[1]]
                y_curr = Y[z[0]:z[1], x[0]:x[1], y[0]:y[1]]

                # Write to file
                filename = path + '/' + basename + str(count) + '.h5'
                hf = h5py.File(filename, 'w')
                hf.create_dataset('volume', data=x_curr.reshape(1, patchsize, patchsize, patchsize))
                hf.create_dataset('segmentation', data=y_curr.reshape(1, patchsize, patchsize, patchsize))
                count += 1

                # Row for image volume in CSV file
                rows.append([str(count), filename + '|' + 'volume' + '|' + '0', image_label])

                # Rows for segmentation classes in CSV file
                for l in range(1, num_class+1):
                    if segmentation_labels is None:
                        seg_label = 'segmentation_' + str(l)
                    else:
                        seg_label = segmentation_labels[l-1]
                    rows.append([str(count),  filename + '|' + 'segmentation' + '|' + '0' + '|' + str(l), seg_label])
    
    # Create CSV file
    with open('dataset.csv', 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)

#######################################################################################################################
# Main

# Load data
hf = h5py.File('training_data.h5', 'r')
X = hf.get('X') # X, Y, Z
Y = hf.get('Y')

# Change to Z, X, Y
X = np.moveaxis(X, (0,1,2), (1,2,0))
Y = np.moveaxis(Y, (0,1,2), (1,2,0))
shape = X.shape

# Extract patches and write to separate files
patchsize = 128
stride = 64

segmentation_labels = ["basement", "slope_mudstone_a", "mass_transport_deposit", "slope_mudstone_b", "slope_valley", "submarine_canyon"]

# Create directory for data
path = 'training_data'
try:
    os.mkdir(path)
except:
    print('Directory already exists.')
basename = 'training_data_patch_'

# Create files
create_and_write_patches(shape, patchsize, stride, X, Y, path, basename, num_chars=4, segmentation_labels=segmentation_labels)

