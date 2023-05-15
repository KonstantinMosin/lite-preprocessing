import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from lite import preproccesing

def main():
    df = pd.read_csv('diabetes.csv')
    df['gender'].replace('Other', np.nan, inplace=True)
    df['smoking_history'].replace('No Info', np.nan, inplace=True)

    X = df.drop('diabetes', axis=1).to_numpy()
    y = df['diabetes'].to_numpy()

    # df = pd.read_csv('marketing_campaign.csv')

    # X = df.drop('Response', axis=1).to_numpy()
    # y = df['Response'].to_numpy()

    X = preproccesing.encode_and_impute(X, y)
    X = preproccesing.scale(X, y)
    X = preproccesing.select(X, y)
    X = preproccesing.decompose(X, y)

if __name__ == '__main__':
    main()