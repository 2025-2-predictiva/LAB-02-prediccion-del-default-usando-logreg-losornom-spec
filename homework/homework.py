# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default". (Ready)
# - Remueva la columna "ID". (Ready)
# - Elimine los registros con informacion no disponible. (Ready)
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default" (Ready)
# y remueva la columna "ID". (Ready)
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test. (Ready)
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



# Se cargan las librerías necesarias y los datos

import os
import gzip
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


dataTrain0 = pd.read_csv("../files/input/train_data.csv.zip",
                         compression="zip")
dataTest0 = pd.read_csv("../files/input/test_data.csv.zip",
                         compression="zip")


# Paso 1: Depurado de los datos

def Debugger(Datos):
    # Renombrar la columna "default payment next month"
    Datos.rename({"default payment next month": "default"}, axis=1, inplace=True)
    # Eliminar la columna "ID"
    Datos.drop("ID", inplace=True, axis=1)
    # Se elimina la categoria 0 de EDUCATION and MARRIAGE
    Datos["EDUCATION"] = np.where(Datos["EDUCATION"] == 0, np.nan, Datos["EDUCATION"])
    Datos["MARRIAGE"] = np.where(Datos["MARRIAGE"] == 0, np.nan, Datos["MARRIAGE"])
    # Eliminar los registros con información no disponible
    Datos.dropna(axis=0, inplace=True)
    # Agrupación de EDUCATION
    Datos["EDUCATION"] = np.where(Datos["EDUCATION"]>4, 4, Datos["EDUCATION"])
    return Datos

# Depurado de los datos
dataTrain = Debugger(dataTrain0.copy())
dataTest = Debugger(dataTest0.copy())


# Paso 2: Division de los datos

def DivData(Datos):
    y = Datos["default"]
    x = Datos.drop("default", axis=1)
    return x, y

x_train, y_train = DivData(dataTrain)
x_test, y_test = DivData(dataTest)


# Paso 3: Crear el pipeline

Variables_Cat = x_train.select_dtypes(include = ["object"]).columns.tolist()
Variables_Num = x_train.select_dtypes(exclude = ["object"]).columns.tolist()

Transformador = ColumnTransformer([
    ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), Variables_Cat),
    ("numericas", MinMaxScaler(), Variables_Num)
    ]
)

pipeline = Pipeline(
    steps=[
        ("tranformador", Transformador),
        ("selector", SelectKBest()),
        ("interacciones", PolynomialFeatures(include_bias=False)),
        ("reglogis", LogisticRegression())
    ]
)


# Paso 4: Optimizar los parametros del modelo

param = {
    "selector__k": list(range(1, len(Variables_Num)+len(Variables_Cat))),
    "selector__score_func": [f_classif, mutual_info_classif],
    "reglogis__C": [0.01, 0.1, 1, 10, 100],
    "reglogis__penalty": ["l1", "l2"],
    "reglogis__solver": ["liblinear", "saga"],
    "interacciones__degree": [1, 2, 3]
}

gridSearch = GridSearchCV(
    estimator=pipeline,
    cv = 10,
    param_grid= param,
    scoring="balanced_accuracy",
    n_jobs=-1
    )


gridSearch.fit(x_train, y_train)


# Paso 5: Guardado del modelo

os.makedirs("../files/models/", exist_ok=True)

with gzip.open("../files/models/model.pkl.gz", "wb") as f:
    pickle.dump(gridSearch, f)


y_train_predic = gridSearch.predict(X=x_train)


# Paso 6: Calculo de metricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score

y_train_predic = gridSearch.predict(X=x_train)
y_test_predic = gridSearch.predict(X=x_test)

def Metricas(y_true, y_fit, name):

    return{
        "type": "metrics",
        "dataset": name,
        "precision": accuracy_score(y_true=y_true, y_pred=y_fit),
        "balanced_accuracy" : balanced_accuracy_score(y_true=y_true, y_pred=y_fit),
        "recall": recall_score(y_true=y_true, y_pred=y_fit),
        "f1_score": f1_score(y_true=y_true, y_pred=y_fit)
    }

Metrics = []

Metrics.append(Metricas(y_true=y_train, y_fit=y_train_predic, name="train"))
Metrics.append(Metricas(y_true=y_test, y_fit=y_test_predic, name="test"))



# Paso 7: Calculo de la matriz de confusion

def ConfusionMatrix(x, y, name):
    y_true = y
    y_predic = gridSearch.predict(X=x)
    mat = confusion_matrix(y_true=y_true, y_pred=y_predic)
    x1 = int(mat[0,0])
    x2 = int(mat[1,1])

    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {"predicted_0": x1, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": x2}
    }

Metrics.append(ConfusionMatrix(x=x_train, y=y_train, name="train"))
Metrics.append(ConfusionMatrix(x=x_test, y=y_test, name="test"))



os.makedirs("../files/output/", exist_ok=True)

with open("../files/output/metrics.json", "w", encoding="utf-8") as file:
    for metric_dict in Metrics:
        json_line = json.dumps(metric_dict)
        file.write(json_line + "\n")













