import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import joblib


df = pd.read_csv('/content/gym_members_exercise_tracking.csv')
df.head()
# Revisión de distribuciones de variables ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribución de {col}')
plt.tight_layout()
plt.show()

# Detección de valores atípicos con Boxplots ---
plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()
# Definir la categoría de grasa corporal según género
def categorize_fat_percentage(row):
    fat = row['Fat_Percentage']
    gender = row['Gender']  # Suponiendo que Gender es 0 = Male, 1 = Female

    if gender == 0:  # Varones
        if 10 <= fat < 20:
            return 'Normal'
        elif 20 <= fat < 25:
            return 'Sobrepeso'
        elif fat >= 25:
            return 'Obeso'
        else:
            return 'Undefined'  # Para valores inesperados

    elif gender == 1:  # Mujeres
        if 20 <= fat < 30:
            return 'Normal'
        elif 30 <= fat < 35:
            return 'Sobrepeso'
        elif fat >= 35:
            return 'Obesa'
        else:
            return 'Undefined'
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Codificación de variables categóricas
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns  # Seleccionar columnas categóricas

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convertir a valores numéricos
    label_encoders[col] = le  # Guardar el encoder por si queremos revertir la conversión
joblib.dump(label_encoders, "label_encoders.pkl")
print("Columnas categóricas convertidas con LabelEncoder:", categorical_cols)

# Asegurar que todas las variables sean numéricas
df = df.apply(pd.to_numeric, errors='coerce')

# --- 4. Tratamiento de valores atípicos (Outliers) ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    # Aplicar recorte por percentiles
    lower_percentile = df[col].quantile(0.01)  # 1er percentil
    upper_percentile = df[col].quantile(0.99)  # 99vo percentil
    df[col] = np.clip(df[col], lower_percentile, upper_percentile)

    # Aplicar transformación logarítmica a columnas con alta dispersión
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col])  # log(1+x) para evitar log(0)
# --- 2. Feature Engineering: Crear nuevas variables ---

df['Heart_Rate_Difference'] = df['Max_BPM'] - df['Resting_BPM']
df['Calories_per_Hour'] = df['Calories_Burned'] / (df['Session_Duration (hours)'] + 1e-5)
df['Weight_Height_Ratio'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
df['Effort_Ratio'] = df['Avg_BPM'] / df['Max_BPM']
df['Hydration_Level'] = df['Water_Intake (liters)'] / df['Weight (kg)']
df['Activity_Index'] = df['Workout_Frequency (days/week)'] * df['Session_Duration (hours)']
df['Experience_Activity_Ratio'] = df['Experience_Level'] / (df['Workout_Frequency (days/week)'] + 1)

# Normalización de los datos
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
joblib.dump(scaler, "scaler.pkl")

# Selección de características con Random Forest
X = df.drop(columns=['Fat_Percentage'])
y = df['Fat_Percentage']

selector = RandomForestRegressor(n_estimators=100, random_state=42)
selector.fit(X, y)
feature_importances = pd.Series(selector.feature_importances_, index=X.columns)

# Eliminar variables con baja importancia (umbral < 0.01)
selected_features = feature_importances[feature_importances > 0.01].index.tolist()
X = X[selected_features]
# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir pipelines para modelos
pipelines = {
    'rf': make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42)),
    'gb': make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42)),
    'xgb': make_pipeline(StandardScaler(), XGBRegressor(objective='reg:squarederror', random_state=42))
}

# Definir hiperparámetros para optimización
grid = {
    'rf':{
        'randomforestregressor__n_estimators':[100,150,200],
        'randomforestregressor__max_depth':[None, 10, 20, 30]
    },
    'gb':{
        'gradientboostingregressor__n_estimators':[100, 150, 200, 250, 300],
        'gradientboostingregressor__learning_rate':[0.01, 0.05, 0.1],
        'gradientboostingregressor__max_depth':[3, 5, 10]
    },
    'xgb':{
        'xgbregressor__n_estimators':[100, 200, 300, 400],
        'xgbregressor__learning_rate':[0.01, 0.05, 0.1],
        'xgbregressor__max_depth':[3, 5, 10]
    }
}

# Validación cruzada con K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
search = {name: GridSearchCV(pipeline, grid[name], cv=kf, scoring='r2', n_jobs=-1)
          for name, pipeline in pipelines.items()}

# Entrenar modelos y optimizar hiperparámetros
for name, model in search.items():
    model.fit(X_train, y_train)
    print(f"Mejores hiperparámetros para {name}: {model.best_params_}")

# Evaluación de Modelos
for name, model in search.items():
    y_pred = model.best_estimator_.predict(X_test)
    print(f"\nResultados para {name}:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
# Definir el modelo
modelo_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Entrenar el modelo con los datos
modelo_xgb.fit(X_train, y_train)

# Guardar el modelo entrenado
modelo.save_model("modelo_xgboost.json") 