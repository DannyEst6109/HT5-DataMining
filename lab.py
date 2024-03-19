import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

# Cargar datos
datos = pd.read_csv("./train.csv")

# Crear la variable Estado según los percentiles
percentil_1 = datos['SalePrice'].quantile(1/3)
percentil_2 = datos['SalePrice'].quantile(2/3)

datos['Estado'] = pd.cut(datos['SalePrice'], bins=[0, percentil_1, percentil_2, float('inf')], labels=['Economica', 'Intermedia', 'Cara'])

# Dividir datos en conjunto de entrenamiento y prueba
train, test = train_test_split(datos, test_size=0.3, random_state=1234)

# Seleccionar características y variable objetivo
features = ["GrLivArea", "YearBuilt", "BsmtUnfSF", "TotalBsmtSF", "GarageArea", "YearRemodAdd", "SalePrice", "LotArea"]
X_train = train[features]
y_train = train["Estado"]
X_test = test[features]
y_test = test["Estado"]

# Entrenar modelo de Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Hacer predicciones
pred_nb = nb_model.predict(X_test)

# Calcular precisión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, pred_nb))
print("\nReporte de Clasificación:")
print(classification_report(y_test, pred_nb))
