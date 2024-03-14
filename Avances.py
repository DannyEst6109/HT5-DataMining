import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import preprocessing

# 1. Cargar los datos
datos = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# 2. Eliminar columnas no deseadas
drop = ["LotFrontage", "Alley", "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "Electrical", "FireplaceQu", "GarageType", "GarageYrBlt",
        "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
datos.drop(columns=drop, inplace=True)
test.drop(columns=drop, inplace=True)

# 3. Dividir el conjunto de datos en entrenamiento y prueba
train, test = train_test_split(datos, test_size=0.3, random_state=1234)

# 4. Crear variable categórica de precio de las casas
train['Estado'] = pd.cut(train['SalePrice'], bins=[-float('inf'), 129975, 163000, float('inf')],
                         labels=['Economica', 'Intermedia', 'Cara'])
test['Estado'] = pd.cut(test['SalePrice'], bins=[-float('inf'), 129975, 163000, float('inf')],
                        labels=['Economica', 'Intermedia', 'Cara'])

# 5. Preparar los datos para el modelo de Naive Bayes
X_train = train.drop(columns=['SalePrice', 'Estado'])
y_train = train['Estado']
X_test = test.drop(columns=['SalePrice', 'Estado'])
y_test = test['Estado']

# Convertir columnas a tipo numérico
columns_to_convert = ['GrLivArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'YearRemodAdd',
                      'SalePrice', 'LotArea']
X_train[columns_to_convert] = X_train[columns_to_convert].apply(pd.to_numeric)
X_test[columns_to_convert] = X_test[columns_to_convert].apply(pd.to_numeric)

# 6. Crear y entrenar el modelo de Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Realizar predicciones
predBayes = modelo.predict(X_test)

# 7. Calcular la eficiencia del modelo
accuracy = accuracy_score(y_test, predBayes)
print("Eficiencia del modelo de clasificación:", accuracy)

# 8. Calcular el error cuadrático medio
mse = mean_squared_error(y_test, predBayes)
print("Error cuadrático medio:", mse)
