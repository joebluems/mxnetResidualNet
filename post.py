import pandas as pd
import json
import requests

"""Setting the headers to send and accept json responses """
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

names = ['image']
df = pd.read_csv('./data/input.csv', names=names)

for a in df['image']:

  """Converting Pandas Dataframe to json """
  df=pd.DataFrame([a],columns=['file'])
  data=df.to_json(orient='records')

  """POST <url>/predict """
  resp = requests.post("http://0.0.0.0:5000/predict", data=json.dumps(data),headers= header)
  print resp.json()
