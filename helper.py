import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt


def load_data(directory, target_dim=(512, 512), num_images=None):
    """
    Loads data and reshapes them to the specified target dimensions.

    Parameters:
    directory (str): The path of the directory containing the images to load.
    target_dim (tuple): A tuple of two integers representing the target dimensions.
                        Default is (512, 512).
    num_images (int): The maximum number of images to load. Default is None.

    Returns:
    list: A list of numpy arrays, where each array represents an image.
          The shape of each array is (target_dim[0], target_dim[1]).
    """
    images = []
    counter = 0
    # Iterate over files in the provided directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Checking if it is a file
        if os.path.isfile(filepath):
            # Processing and adding image
            image = np.array(Image.open(filepath).convert('L'))
            if image.shape == target_dim:
                counter += 1
                images.append(image)

            if num_images is not None and counter >= num_images:
                break
    return images


def plot(image, num_images=1):
    """
    Plots image(s)
    """
    if num_images == 1:
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        for i in range(num_images):
            fig = plt.figure
            plt.imshow(image[i], cmap='gray')
            plt.show()


def plot_training_results(train_rate_array, train_distortion_array, test_rate_array, test_distortion_array, test_bpp_array):
    """
    Plots the training and test rate and distortion arrays on the same plot with two y-axes, and plots the
    training and test bpp arrays on a separate plot.

    Args:
        train_rate_array (list or array-like): An array of training rate values.
        train_distortion_array (list or array-like): An array of training distortion values.
        test_rate_array (list or array-like): An array of test rate values.
        test_distortion_array (list or array-like): An array of test distortion values.
        test_bpp_array (list or array-like): An array of test bpp values.

    Returns:
        None.
    """

    # Create an array of epoch values (assuming the arrays have the same length)
    epoch_array = range(len(train_rate_array))

    # Create the figure and axes for the first plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5),gridspec_kw={'wspace': 0.5})

    # Plot the training rate on the primary y-axis
    ax1.plot(epoch_array, train_rate_array, color='tab:red', linestyle='--', label='Train')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Rate', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for the training distortion
    ax2 = ax1.twinx()

    # Plot the training distortion on the secondary y-axis
    ax2.plot(epoch_array, train_distortion_array, color='tab:blue', linestyle='--')
    ax2.set_ylabel('Distortion', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Plot the test rate on the primary y-axis
    ax1.plot(epoch_array, test_rate_array, color='tab:red', linestyle='-', label='Test')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot the test distortion on the secondary y-axis
    ax2.plot(epoch_array, test_distortion_array, color='tab:blue', linestyle='-')

    legend = ax1.legend(loc='upper left')
    for line in legend.get_lines():
        line.set_linewidth(1.5)
        line.set_color('black')

    # Add a title to the first plot
    ax1.set_title('Train and Test Rates and Distortions vs Epoch')

    # Create the second plot
    ax3 = plt.subplot(1, 2, 2)

    # Plot the training bpp on the primary y-axis
    ax3.plot(epoch_array, test_bpp_array, color='black', linestyle='-', label='Test')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('BPP', color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    
    ax3.set_title('Test BPP vs Epoch')
    
    # Plot the test bpp on the primary y-axis
    ax3.plot


def get_subblocks(arr, row_dim, col_dim):
    """
    Divide a 2D array into subblocks of size row_dim x col_dim.

    Parameters:
    arr (numpy.ndarray): 2D input array.
    row_dim (int): number of rows in each subblock.
    col_dim (int): number of columns in each subblock.

    Returns:
    numpy.ndarray: 3D array of shape (num_blocks, row_dim, col_dim),
                   where num_blocks is the number of subblocks in arr.
                   Each subblock preserves the "physical" layout of arr.
    """
    num_rows, num_cols = arr.shape
    assert num_rows % row_dim == 0, f"{num_rows} rows is not evenly divisible by {row_dim}"
    assert num_cols % col_dim == 0, f"{num_cols} cols is not evenly divisible by {col_dim}"
    num_blocks = (num_rows // row_dim) * (num_cols // col_dim)
    return (arr.reshape(num_rows // row_dim, row_dim, -1, col_dim)
               .swapaxes(1, 2)
               .reshape(num_blocks, row_dim, col_dim))


def get_original_image(blocks, row_dim, col_dim):
    """
    Combine subblocks of size (row_dim x col_dim) into a 2D array.

    Parameters:
    blocks (numpy.ndarray): 3D array of shape (num_blocks, row_dim, col_dim),
                            where num_blocks is the number of subblocks.
    row_dim (int): number of rows in each subblock.
    col_dim (int): number of columns in each subblock.

    Returns:
    numpy.ndarray: 2D array of shape (num_blocks*row_dim, num_blocks*col_dim),
                   which is the original image that was split into subblocks.
    """
    num_blocks, num_rows, num_cols = blocks.shape
    assert num_rows == row_dim, f"Block height {num_rows} does not match expected height {row_dim}"
    assert num_cols == col_dim, f"Block width {num_cols} does not match expected width {col_dim}"
    num_rows_original = int(np.sqrt(num_blocks * row_dim / col_dim))
    num_cols_original = int(num_blocks / num_rows_original)
    return (blocks.reshape(num_rows_original, num_cols_original, row_dim, col_dim)
                 .swapaxes(1, 2)
                 .reshape(num_rows_original*row_dim, num_cols_original*col_dim))


def run_model_full_image(image, model, block_dim):
    """
        Runs the compression model on an input image in a block-wise manner 
        and returns the reconstructed output image.
        
        Parameters:
        - image: input image to run the model on
        - model: compression model to apply to the image
        - block_dim: size of blocks to split the image into
        
        Returns:
        - reconstructed_image: output image reconstructed from the model's output blocks
        """
    # Split image into blocks
    image_blocks = get_subblocks(image, block_dim, block_dim)
    
    # Preprocess blocks for model input
    image_blocks = np.array(image_blocks, dtype=np.float32)
    image_blocks = np.reshape(image_blocks, (-1, block_dim, block_dim, 1))
    
    # Run model on blocks
    model_blocks = model.predict(image_blocks)
    model_blocks = np.squeeze(model_blocks)
    
    # Reconstruct image from model output blocks
    reconstructed_image = get_original_image(model_blocks, block_dim, block_dim)
    
    return reconstructed_image