import pickle
import pandas as pd

# Cargar nombres de las variables esperadas
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print("Features esperadas por el modelo:")
print(feature_names)

# Crear datos simulados con esas columnas
X_test = pd.DataFrame([
     [1, 30, 75, 1.75, 180, 150, 60, 1, 300, 2.0, 2, 120, 300, 0.8, 0.9, 3],
    [1, 45, 65, 1.60, 170, 140, 70, 0.5, 200, 1.2, 1, 100, 400, 0.7, 0.85, 2],
    [0, 28, 90, 1.80, 190, 160, 65, 1.5, 500, 2.5, 3, 125, 333, 0.9, 0.95, 4]
], columns=feature_names)

print("\nDatos de prueba:")
print(X_test)

# Cargar preprocesador
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Transformar datos
X_test_transformed = preprocessor.transform(X_test)

# Cargar modelo
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predecir
preds = model.predict(X_test_transformed)

print("\nPredicciones:")
print(preds)