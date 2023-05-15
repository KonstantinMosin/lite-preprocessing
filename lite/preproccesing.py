import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

def encode_and_impute(X, y):
    print('encode and impute')

    encoders = [
        LabelEncoder(),
        OneHotEncoder(sparse_output=False),
        OrdinalEncoder()
    ]

    imputers = [
        SimpleImputer(),
        KNNImputer(),
        IterativeImputer()
    ]

    model = RandomForestClassifier()

    X_ = X.copy()
    for i in range(X.shape[1]):
        x = np.array(X[:,i].tolist())
        if np.issubdtype(x.dtype, np.number) is False:
            x = encoders[0].fit_transform(X[:, i])
        if np.vectorize(lambda x: np.isnan(x))(x).any():
            x = imputers[0].fit_transform(X[:, i].reshape(-1, 1))
        X_[:, i] = x.ravel()

    accuracy = estimate(X_, y, model)
    print('accuracy =', accuracy)

    for i in range(X.shape[1]):
        x = np.array(X[:,i].tolist())
        if np.issubdtype(x.dtype, np.number) is False:
            for encoder in encoders:
                x = encoder.fit_transform(X[:, i].reshape(-1,1))
                
                if np.vectorize(lambda x: np.isnan(x))(x).any():
                    for imputer in imputers:
                        x = imputer.fit_transform(x)

                        copy = insert_instead(X_, x, i)

                        copy_accuracy = estimate(copy, y, model)
                        print('encode+impute, accuracy =', copy_accuracy)
                        if copy_accuracy > accuracy:
                            print('selected', encoder, '+', imputer, 'for feature = ', i)
                            accuracy = copy_accuracy
                            X_ = copy

                copy = insert_instead(X_, x, i)

                copy_accuracy = estimate(copy, y, model)
                print('encode, accuracy =', copy_accuracy)
                if copy_accuracy > accuracy:
                    print('selected', encoder, 'for feature = ', i)
                    accuracy = copy_accuracy
                    X_ = copy

        elif np.vectorize(lambda x: np.isnan(x))(x).any():
            for imputer in imputers:
                x = imputer.fit_transform(x.reshape(-1, 1))

                copy = insert_instead(X_, x, i)
                copy_accuracy = estimate(copy, y, model)
                print('impute, accuracy =', copy_accuracy)
                if copy_accuracy > accuracy:
                    print('selected', imputer, 'for feature = ', i)
                    accuracy = copy_accuracy
                    X_ = copy

    print('accuracy =', accuracy)

    return X_

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    SplineTransformer,
    FunctionTransformer
)

def scale(X, y):
    print('scale')

    scalers = [
        StandardScaler(),
        MinMaxScaler(),
        MaxAbsScaler(),
        RobustScaler()
    ]

    transformers= [
        PowerTransformer(),
        QuantileTransformer(),
        SplineTransformer(),
        FunctionTransformer()
    ]

    model = RandomForestClassifier()
    X_ = X.copy()
    accuracy = estimate(X_, y, model)
    print('accuracy =', accuracy)

    for i in range(X.shape[1]):
        for scaler in scalers:
            x = scaler.fit_transform(X[:,i].reshape(-1, 1))            
            copy = insert_instead(X_, x, i)
            copy_accuracy = estimate(copy, y, model)
            print('scale, accuracy =', copy_accuracy)
            if copy_accuracy > accuracy:
                print('selected', scaler, 'for feature = ', i)
                accuracy = copy_accuracy
                X_ = copy

            for transformer in transformers:
                x_ = transformer.fit_transform(x)
                copy = insert_instead(X_, x_, i)
                copy_accuracy = estimate(copy, y, model)
                print('transform after scale, accuracy =', copy_accuracy)
                if copy_accuracy > accuracy:
                    print('selected', transformer, 'after scale for feature = ', i)
                    accuracy = copy_accuracy
                    X_ = copy

        for transformer in transformers:
            x = transformer.fit_transform(X[:,i].reshape(-1, 1))
            copy = insert_instead(X_, x_, i)
            copy_accuracy = estimate(copy, y, model)
            print('transform, accuracy =', copy_accuracy)
            if copy_accuracy > accuracy:
                print('selected', transformer, 'for feature = ', i)
                accuracy = copy_accuracy
                X_ = copy
    
    print('accuracy =', accuracy)
    return X_

from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    RFE,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
    SelectFpr,
    SelectFdr,
    SelectFwe
)

def select(X, y):
    print('select')

    model = RandomForestClassifier()

    selectors = [
        SelectKBest(k=int(X.shape[1] * 0.7)),
        SelectFromModel(model),
        RFE(model),
        SelectPercentile(),
        SequentialFeatureSelector(model, n_jobs=-1),
        VarianceThreshold(),
        SelectFpr(),
        SelectFdr(),
        SelectFwe()
    ]

    X_ = X.copy()
    accuracy = estimate(X_, y, model)
    print('accuracy = ', accuracy)

    for selector in selectors:
        copy = selector.fit_transform(X, y)
        copy_accuracy = estimate(copy, y, model)
        print('select, accuracy =', copy_accuracy)
        if copy_accuracy > accuracy:
            print('selected', selector)
            accuracy = copy_accuracy
            X_ = copy

    return X_

from sklearn.decomposition import (
    PCA,
    NMF,
    FastICA,
    FactorAnalysis,
    LatentDirichletAllocation,
    SparsePCA,
    MiniBatchSparsePCA,
    KernelPCA,
    TruncatedSVD
)

def decompose(X, y):
    decomposers = [
        PCA(),
        NMF(),
        FastICA(),
        FactorAnalysis(),
        LatentDirichletAllocation(),
        SparsePCA(),
        MiniBatchSparsePCA(),
        TruncatedSVD()
    ]

    model = RandomForestClassifier()
    X_ = X.copy()
    accuracy = estimate(X_, y, model)

    for decomposer in decomposers:
        copy = decomposer.fit_transform(X)
        copy_accuracy = estimate(copy, y, model)
        print('decompose, accuracy =', copy_accuracy)
        if copy_accuracy > accuracy:
            print('selected', decomposer)
            accuracy = copy_accuracy
            X_ = copy

    return X_

def balance():
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def estimate(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def insert_instead(X, x, i):
    X_ = X[:,:i]
    X_ = np.concatenate((X_, x.reshape(-1, 1) if x.ndim == 1 else x), axis=1)
    X_ = np.concatenate((X_, X[:,i + 1:]), axis=1)
    return X_
