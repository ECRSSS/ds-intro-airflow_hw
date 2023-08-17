import glob
import pathlib
from datetime import datetime

import pandas as pd
import dill
import json


def predict():
    with open(sorted(list(pathlib.Path('data/models').glob('*')), reverse=True)[0].absolute(), 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob('data/test/*.json'):
        with open(filename, 'r') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': form['id'], 'pred': y[0]}
            df1 = pd.DataFrame([x])
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'data/predictions/predict_{datetime.now().strftime("%d-%m-%Y-%H-%M")}.csv', sep=',', index=False)


if __name__ == '__main__':
    predict()
