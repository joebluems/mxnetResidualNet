import os, urllib
import mxnet as mx
import pandas as pd
import json
import cv2
import numpy as np
from collections import namedtuple
from flask import Flask, jsonify, request

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
  
    return "{ 'p1':'"+classes[0]+"', 'p2':'"+classes[1]+"' }"

##########################
###### API PIECE #########
##########################

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def apicall():
  predictions="{'p1':'NA'}"
  try:
    test_json = request.get_json()
    test = pd.read_json(test_json, orient='records')
    predictions = predict('data/'+test.iloc[0,0], mod, synsets)
  except Exception as e:
    print e

  responses = jsonify(predictions)
  responses.status_code = 200

  return (responses)
