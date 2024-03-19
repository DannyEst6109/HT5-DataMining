import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

# Cargar datos
datos = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# Crear variable 'Estado' según los percentiles
percentil_1 = datos['SalePrice'].quantile(1/3)
percentil_2 = datos['SalePrice'].quantile(2/3)

datos['Estado'] = pd.cut(datos['SalePrice'], bins=[0, percentil_1, percentil_2, float('inf')], labels=['Economica', 'Intermedia', 'Cara'])

# Separar conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos[['GrLivArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'YearRemodAdd', 'SalePrice', 'LotArea']], datos['Estado'], test_size=0.3, random_state=1234)

# Entrenar modelo de Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train[['GrLivArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'YearRemodAdd', 'SalePrice', 'LotArea']], y_train)

# Predecir con el conjunto de prueba
pred_nb = nb_model.predict(X_test[['GrLivArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'YearRemodAdd', 'SalePrice', 'LotArea']])

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, pred_nb)
print("Precisión:", accuracy)

# Matriz de confusión y reporte de clasificación
print("Matriz de confusión:")
print(confusion_matrix(y_test, pred_nb))
print("\nReporte de clasificación:")
print(classification_report(y_test, pred_nb))

# Validación cruzada
kf = KFold(n_splits=10, random_state=123, shuffle=True)
y_pred_cv = cross_val_predict(nb_model, datos[['GrLivArea', 'YearBuilt', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'YearRemodAdd', 'SalePrice', 'LotArea']], datos['Estado'], cv=kf)

# Matriz de confusión para validación cruzada
print("\nMatriz de confusión para validación cruzada:")
print(confusion_matrix(datos['Estado'], y_pred_cv))
