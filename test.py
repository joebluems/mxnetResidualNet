import os, urllib,json
import pandas as pd
import mxnet as mx
from flask import Flask, jsonify, request

### this is the post stuff###
names = ['image']
df = pd.read_csv('./data/input.csv', names=names)
df = df.head(5)

"""Converting Pandas Dataframe to json """
data = df.to_json(orient='records')

#### PASS TO API #####
#### this is the flask part ###
test = pd.read_json(data, orient='records')
print test 

##################
### READ MODEL ###
##################
def download(url,prefix=''):
    filename = prefix+url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)

download('model/resnet-152-symbol.json', 'full-')
download('model/resnet-152-0000.params', 'full-')
download('model/synset.txt', 'full-')

with open('full-synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

sym, arg_params, aux_params = mx.model.load_checkpoint('full-resnet-152', 0)

mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)

#######################
###HELPER FUNCTION ####
#######################
import cv2
import numpy as np
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def predict(filename, mod, synsets):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    img = img[np.newaxis, :] 
    
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    a = np.argsort(prob)[::-1]    
    classes = []
    for i in a[0:5]:
        classes.append(synsets[i])

    return json.dumps(classes)

########################
### MAKE PREDICTIONS ###
########################
for a in test['image']:
  predictions = predict('data/'+a, mod, synsets)
  print predictions

