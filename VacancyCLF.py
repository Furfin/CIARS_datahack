import json
import re
from pprint import pprint

import numpy as np
import pandas as pd
from IPython.core.display import HTML, display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline



class FullDescriptionCreator(BaseEstimator, TransformerMixin):
    patt = re.compile("[^\s\w]")

    def __init__(self, responsibilities):
        self.responsibilities = responsibilities

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["responsibilities"] = self.responsibilities
        X["full_description"] = (
            X["name"] + " " + X["responsibilities"].fillna("")
        ).map(str.lower)
        X.loc[:, "full_description"] = X["full_description"].str.replace(
            self.patt, " ", regex=True
        )
        return X


train = pd.read_csv("./train.csv", index_col="index")
print(f"{train.shape}")
print("Типы столбцов: ")
display(train.dtypes)
print("Фрагмент данных: ")
display(train.head())

train = train.query("target != -1")
print(f"{train.shape=}")
display(train.head())

display(HTML(train.at[169939030, "description"]))

train.at[169939030, "description"]

with open(
    "./vacancy_descriptions/2_parsed.json", "r", encoding="utf8"
) as fp:
    descriptions = json.load(fp)
    
pizza_description = [
    description for description in descriptions if description["ID"] == 169939030
][0]

responsibilities = pd.Series({
    description["ID"]: r[0]
    if (r := description["Content"].get("Обязанности")) is not None
    else None
    for description in descriptions
}, name="responsibilities")
responsibilities.head(3)


train["responsibilities"] = responsibilities
display(train.head(3))
display(train.info())

train["full_description"] = (
    train["name"] + " " + train["responsibilities"].fillna("")
).map(str.lower)
display(train.head(3))

patt = re.compile("[^\w\s]")
train.loc[:, "full_description"] = train["full_description"].str.replace(
    patt, " ", regex=True
)
display(train.head(3))

X_train_raw, y_train = train["full_description"], train["target"]
vectorizer = TfidfVectorizer(max_features=128)
X_train = vectorizer.fit_transform(X_train_raw)





train = pd.read_csv("./train.csv", index_col="index")
train = train.query("target != -1")

with open(
    "./vacancy_descriptions/2_parsed.json", "r", encoding="utf8"
) as fp:
    descriptions = json.load(fp)

responsibilities = pd.Series(
    {
        description["ID"]: r[0]
        if (r := description["Content"].get("Обязанности")) is not None
        else None
        for description in descriptions
    },
    name="responsibilities",
)

X_train, y_train = train.drop(columns=["target"]), train["target"]

