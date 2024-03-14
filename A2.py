import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri

# Cargar datos
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Definir función para convertir el precio de venta en categorías
def categorize_price(price):
    if price <= 129975:
        return 'Economica'
    elif price <= 163000:
        return 'Intermedia'
    else:
        return 'Cara'

train_data['Estado'] = train_data['SalePrice'].apply(categorize_price)

# Seleccionar características y variable objetivo
features = ['GrLivArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'YearRemodAdd', 'SalePrice', 'LotArea']
X_train = train_data[features]
y_train = train_data['Estado']

# Entrenar modelo Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train.drop(columns=['SalePrice']), y_train)

# Preparar datos de prueba
X_test = test_data[features]
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convertir todas las características a numéricas

# Predicción usando Naive Bayes
pred_bayes = gnb.predict(X_test.drop(columns=['SalePrice']))

# Convertir las predicciones a categorías
pred_bayes_categories = pd.Categorical.from_codes(pred_bayes, categories=['Economica', 'Intermedia', 'Cara'])

# Calcular eficiencia del modelo Naive Bayes
accuracy_bayes = (pred_bayes_categories == test_data['Estado']).mean()
print("Eficiencia del modelo Naive Bayes:", accuracy_bayes)

# Entrenar modelo de regresión lineal
lm = LinearRegression()
lm.fit(X_train.drop(columns=['SalePrice']), X_train['SalePrice'])

# Predecir con modelo de regresión lineal
pred_lm = lm.predict(X_test.drop(columns=['SalePrice']))

# Calcular error cuadrático medio (MSE) del modelo de regresión lineal
mse_lm = mean_squared_error(test_data['SalePrice'], pred_lm)
print("MSE del modelo de regresión lineal:", mse_lm)

# Entrenar árbol de regresión
rpart = importr("rpart")
rpart.plot = importr("rpart.plot")

# Convertir datos a formato compatible con R
pandas2ri.activate()
r_X_train = pandas2ri.py2ri(X_train)
r_y_train = pandas2ri.py2ri(y_train)

# Entrenar árbol de regresión con R
arbol_3 = rpart.rpart(formula="SalePrice ~ .", data=r_X_train)

# Visualizar árbol de regresión
rpart.plot.rpart(arbol_3)

# Convertir datos de prueba a formato compatible con R
r_X_test = pandas2ri.py2ri(X_test)

# Predecir con árbol de regresión
pred_arbol = r.predict(arbol_3, newdata=r_X_test)

# Calcular MSE del árbol de regresión
mse_arbol = mean_squared_error(test_data['SalePrice'], pandas2ri.ri2py(pred_arbol))
print("MSE del árbol de regresión:", mse_arbol)
