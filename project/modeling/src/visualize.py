import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io

def plot_layer_activations(params, model, data_split, images_per_row=16):
    '''extremely helpful reference; one (main) chunk is verbatim!
    
    https://towardsdatascience.com/visualizing-intermediate-activations-of-a-cnn-trained-on-the-mnist-dataset-2c34426416c8
    '''
    
    # all the conv / pooling layers only, which means this is super specific to cnn.get_model, and will need
    # to change to something more generic when we use this for other models next term
    layers = model.layers[1:6]
    am = tf.keras.Model(inputs=model.input, outputs=[l.output for l in layers])
    am_layer_names = [l.name for l in layers]
    
    image, _ = list(data_split)[0]

    save_path = os.path.join(params.save_dir,
                            f'{os.path.basename(params.model_dir)}_original_image.png')
    
    plt.imsave(file_io.FileIO(save_path, 'w'),
               image.numpy().reshape(params.image_size, params.image_size),
               vmin=0, vmax=255, cmap='gray')
    
    activations = am.predict(image)
    
    for layer_ix, (layer_name, activation) in enumerate(zip(am_layer_names, activations)):
        num_features = activation.shape[-1]
        size = activation.shape[1]
        
        num_cols = num_features // images_per_row
        display_grid = np.zeros((size * num_cols, size * images_per_row), dtype='float16')
        
        for c_ix in range(num_cols):
            for r_ix in range(images_per_row):
                #print(c_ix, r_ix, c_ix * images_per_row + r_ix, activation.shape)
                channel_image = activation[0, :, :, c_ix * images_per_row + r_ix]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[c_ix*size:(c_ix + 1)*size,
                                r_ix*size:(r_ix + 1)*size
                                ] = channel_image
        
        scale = 1. / size
        
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        save_path = os.path.join(params.save_dir,
                                 f'{os.path.basename(params.model_dir)}_{layer_ix}_{layer_name}_activations.png')
        plt.imsave(file_io.FileIO(save_path, 'w'), display_grid, cmap='viridis')