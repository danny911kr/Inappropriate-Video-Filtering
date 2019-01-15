# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import tensorflow as tf
import data_helper
from random import shuffle

from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--strmaxlen', type=int, default=150)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=10)
    args.add_argument('--embedding', type=int, default=256)
    args.add_argument('--featuresize', type=int, default=129) # ascii code 기준 0~127 + 1
    config = args.parse_args()

    inputs = layers.Input((config.strmaxlen,))  
    layer = layers.Embedding(config.featuresize, config.embedding, input_length=config.strmaxlen, mask_zero = True)(inputs)
    layer = layers.Bidirectional(layers.GRU(256, return_sequences=True))(layer)
    layer = layers.Bidirectional(layers.GRU(256, return_sequences=False))(layer)

    layer_dense = layers.Dense(3)(layer)
    outputs_softmax = layers.Activation('softmax')(layer_dense)

    model = models.Model(inputs=inputs, outputs=outputs_softmax)
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.001,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    
    filename = "model-epoch 7"
    model.load_weights(filename)
    
    preprocessed_data = data_helper.preprocess("i am a boy how are you?", config.strmaxlen)
    output_prediction = model.predict(preprocessed_data)[0].flatten().tolist()
    output_prediction_label = output_prediction.index(max(output_prediction))
    print(output_prediction)
    print(output_prediction_label)