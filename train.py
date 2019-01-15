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
    layer = layers.Bidirectional(layers.GRU(128, return_sequences=True))(layer)
    layer = layers.Bidirectional(layers.GRU(128, return_sequences=False))(layer)

    layer_dense = layers.Dense(3)(layer)
    outputs_softmax = layers.Activation('softmax')(layer_dense)

    model = models.Model(inputs=inputs, outputs=outputs_softmax)
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.001,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    
    file_train_instances = "sample50.csv"  #데이터셋 파일 이름
    
    # Load data
    print("Loading data...") 
    sentences, sentimentlabels = data_helper.load_data_and_labels(file_train_instances,config.strmaxlen)
    
    dataset_len = len(sentences)
    one_batch_size = dataset_len//config.batch
    if dataset_len % config.batch != 0:
        one_batch_size += 1
    
    sentiment_dataset = list(zip(sentences,sentimentlabels))
    print(sentiment_dataset)
    
    # epoch마다 학습을 수행합니다.
    for epoch in range(config.epochs):
        avg_loss = 0.0
        avg_acc = 0.0
        
        shuffle(sentiment_dataset)
        for batch in enumerate(data_helper._batch_loader(sentiment_dataset, config.batch)):
            i = batch[0] # enumerate - index
            data, labels = zip(*batch[1]) #batch[1] = (data,labels) -> 이걸 data, labels로 분리한다.
            data = np.array(data) # numpy array화
            labels = np.array(labels) # numpy array화
            loss, acc = model.train_on_batch(data, labels)
            print('Batch : ', i + 1, '/', one_batch_size,
                      ', loss in this minibatch: ', float(loss),
                      ', acc in this minibatch: ', float(acc))
            avg_loss += float(loss)
            avg_acc += float(acc)
        
        print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size), ' train_acc:', float(avg_acc/one_batch_size))   
        filename = str("model-epoch "+ str(epoch))
        model.save(filename)