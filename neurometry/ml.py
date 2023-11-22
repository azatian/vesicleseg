import pandas as pd
import numpy as np
from tifffile import imread
import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from keras.models import load_model
#mport tensorflow as tf
#from albumentations import (
#    Compose, Rotate, VerticalFlip, HorizontalFlip, ElasticTransform, Transpose, Sharpen
#)
#import random
#random.seed(123)

'''
transforms = Compose([
            Rotate(),
            VerticalFlip(),
            Transpose(),
            HorizontalFlip(),
            #Sharpen()
            #ElasticTransform()
        ])
'''

CONFIG_PATH = "input/"

def ingestor(df):
    id_to_vol = {}
    for index, row in df.iterrows():
        name = row["wk_id"]
        vol1 = np.transpose(imread("cutouts/"+name+"/img/vol.tiff"), (1,2,0))
        seg1 = np.transpose(imread("cutouts/"+name+"/cellseg/cellseg.tiff"), (1,2,0))
        masked1 = (seg1/255)*vol1
        #id_to_vol[name] = masked1
        #Divide by 255 here
        id_to_vol[name] = np.array(masked1/255.0).astype('float32')
    
    return id_to_vol

def collapsor(id_to_vol):
    collapsed_id_to_vol = {}
    for key, value in id_to_vol.items():
        z = value.shape[2]
        for i in range(z):
            _id = key + "_" + str(i)
            collapsed_id_to_vol[_id] = value[:,:,i]

    return collapsed_id_to_vol

#remove outliars
def remover(collapsed_id_to_vol, outliars):
    outliars = outliars[outliars["include"] == "yes"]
    for index, row in outliars.iterrows():
        _key = row["wk_id"] + "_" + str(int(row["section_number"]))
        del collapsed_id_to_vol[_key]
    
    return collapsed_id_to_vol

#collapsed_id to cleaned annotation
def construct_annotations(collapsed_id_to_vol, annotations):
    collapsed_id_to_annotation = {}
    wkids = set()
    for key, value in collapsed_id_to_vol.items():
        substrings = key.split("_")
        original = substrings[0]
        index = substrings[1]
        collapsed_id_to_annotation[key] = annotations[original][:,:,int(index)]
        wkids.add(original)
    #use the wkids set to filter out the wk_id_to_rating dataframe for stratification in model
    return collapsed_id_to_annotation, wkids

class UNet(nn.Module):
    """UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # _conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1, depth=4, final_activation=None):
        super().__init__()

        assert depth < 10, "Max supported depth is 9"

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = depth

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(
                final_activation, nn.Module
            ), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList(
            [
                self._conv_block(in_channels, 16),
                self._conv_block(16, 32),
                self._conv_block(32, 64),
                self._conv_block(64, 128),
                self._conv_block(128, 256),
                self._conv_block(256, 512),
                self._conv_block(512, 1024),
                self._conv_block(1024, 2048),
                self._conv_block(2048, 4096),
            ][:depth]
        )
        # the base convolution block
        if depth >= 1:
            self.base = self._conv_block(2 ** (depth + 3), 2 ** (depth + 4))
        else:
            self.base = self._conv_block(1, 2 ** (depth + 4))
        # modules of the decoder path
        self.decoder = nn.ModuleList(
            [
                self._conv_block(8192, 4096),
                self._conv_block(4096, 2048),
                self._conv_block(2048, 1024),
                self._conv_block(1024, 512),
                self._conv_block(512, 256),
                self._conv_block(256, 128),
                self._conv_block(128, 64),
                self._conv_block(64, 32),
                self._conv_block(32, 16),
            ][-depth:]
        )

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList(
            [
                self._upsampler(8192, 4096),
                self._upsampler(4096, 2048),
                self._upsampler(2048, 1024),
                self._upsampler(1024, 512),
                self._upsampler(512, 256),
                self._upsampler(256, 128),
                self._upsampler(128, 64),
                self._upsampler(64, 32),
                self._upsampler(32, 16),
            ][-depth:]
        )
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv2d(16, out_channels, 1)
        self.activation = final_activation

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
        return config
#need to edit for more control later
def get_final_activation():
    return torch.nn.Sigmoid()

def load_weights(model, path):
    model.load_state_dict(torch.load(path))





'''
def change_target(row):
    #completely random
    #if random.uniform(0, 1) < .25:
    #    return 0
    #else:
    #    return 1
    #switching target variables, NONE should always be 0, anything else 1
    if row["pre_syn_label"] == "NONE" or row["pre_syn_label"] == "FEW":
        return 0
    else:
        return 1
    #elif row["pre_syn_label"] == "FEW":
    #    return 1
    #else:
    #    return 2

def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    #aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.reshape(aug_img, [160,160,16])
    aug_img = tf.expand_dims(aug_img, axis=3)
    return aug_img

def process_data(volume, label):
    aug_volume = tf.numpy_function(func=aug_fn, inp=[volume], Tout=tf.float32)
    return aug_volume, label

def train_preprocessing(volume, label):
    """Process training data."""
    # Divide by 255
    #volume = volume/255
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data."""
    #volume = volume/255
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def get_model(width=160, height=160, depth=16):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))



    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same")(inputs)
    #x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Flatten()(x)
    #x = layers.GlobalMaxPooling3D()(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    #outputs = layers.Dense(units=3, activation="softmax")(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn-try4")
    return model


def loader(filepath):
    model = load_model(filepath)
    return model

'''