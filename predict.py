# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:13:22 2018

@author: rodolpho
"""
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import json

model = Sequential()

classes_json_file = open('classes.json', 'r')
classes_json = json.load(classes_json_file)
classes_json_file.close()

json_file = open('model.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
print(model)
model.load_weights('weights.hd5f')

data_to_predict = np.asarray([[5.9,3.0,4.2,1.5]])

prediction = model.predict(data_to_predict, batch_size=None, verbose=0, steps=1)

classes_quantity = len(classes_json)
for i in range(0, classes_quantity):
    chance = prediction[0][i] * 100
    clazz = classes_json[str(i)]
    class_name = clazz['name']
    print(class_name + ': %0.2f' % chance)