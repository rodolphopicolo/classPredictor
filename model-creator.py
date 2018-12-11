# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:16:18 2018

@author: rodolpho
"""

import random
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

def create_model():
    input_data_quantity = 4
    layers_neuron_quantity = [32, 3]
    
    model = Sequential()
    model.add(Dense(units=layers_neuron_quantity[0], input_dim=input_data_quantity, activation='sigmoid'))
#    model.add(Dense(units=layers_neuron_quantity[1], input_dim=input_data_quantity, activation='sigmoid'))
#    model.add(Dropout(15, noise_shape=None, seed=None))
    model.add(Dense(units=layers_neuron_quantity[1], activation='softmax'))
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    return model

dataset_file = open('./iris-dataset.txt', 'r')
dataset = []

classes_by_name = {}
classes_by_id = {}
classes_counter = -1

attributes_quantity = None

for l in dataset_file:
    if l == '':
        continue
    fields = l.split(',')
    
    if attributes_quantity == None:
        attributes_quantity = len(fields) - 1

    fields[attributes_quantity] = fields[attributes_quantity].replace('\n', '')

    for i in range(0,attributes_quantity):
        fields[i] = float(fields[i])
        
    if not fields[attributes_quantity] in classes_by_name:
        classes_counter = classes_counter + 1
        class_name = fields[attributes_quantity]
        classes_by_name[class_name] = {'quantity':0, 'id':classes_counter}
        classes_by_id[classes_counter] = {'quantity':0, 'name':class_name}
        
    classes_by_name[class_name]['quantity'] = classes_by_name[class_name]['quantity'] + 1
    classes_by_id[classes_by_name[class_name]['id']]['quantity'] = classes_by_id[classes_by_name[class_name]['id']]['quantity'] + 1
    
    fields.append(classes_by_name[fields[attributes_quantity]]['id'])
    dataset.append(fields)
    

random.shuffle(dataset)

attributes = [row[0:attributes_quantity] for row in dataset]
labels = [row[attributes_quantity] for row in dataset]

encoder = LabelEncoder()
encoded = encoder.fit_transform(labels)
classes = np_utils.to_categorical(encoded)


model = create_model()

np_attributes = np.asarray(attributes, dtype=np.float32)
np_classes = np.asarray(classes, dtype=np.float32)


model.fit(np_attributes, np_classes, batch_size=5, epochs=5000, validation_split=0.3, verbose=1, shuffle=True)

json_model = model.to_json()
archtecture_model_file = open('model.json', 'w')
archtecture_model_file.write(json_model)
archtecture_model_file.close()
model.save_weights('weights.hd5f')

classes_json_file = open('classes.json', 'w')
json.dump(classes_by_id, classes_json_file)
classes_json_file.close()