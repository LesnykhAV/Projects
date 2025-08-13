import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from catboost import CatBoostRegressor
import numpy as np

# Загружаем очищенные данные 
df = pd.read_csv("df_cleaned.csv") # очищенные, но без кодирования и масштабирования признаков
X = df.drop(columns=["target"])
y = df["target"]

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Признаки 
categorical_features = ['status', 'propertyType', 'city', 'zipcode', 'state']
numerical_features = [
    'baths', 'sqft', 'beds', 'stories', 'heating', 'cooling',
    'parking', 'lotsize', 'house_age', 'was_remodeled', 'remodeled_age',
    'average_school_rating', 'average_school_distance', 'school_count_in_area'
]

# Создаём препроцессор, который кодирует категориальные признаки и масштабирует числовые признаки
preprocessor = ColumnTransformer([
    ("cat", ce.BinaryEncoder(cols=categorical_features), categorical_features),
    ("num", RobustScaler(), numerical_features)
])

# Обучаем препроцессор и трансформируем данные
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)

# Инициализация и обучение модели CatBoostRegressor
catboost_model = CatBoostRegressor(
    iterations=1947,
    learning_rate=0.07,
    depth=10,
    l2_leaf_reg=3,
    random_seed=42,
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=100,
    verbose=False,
    use_best_model=True
)

# Обучаем модель на преобразованных данных
catboost_model.fit(
    X_train_transformed, y_train,
    eval_set=(X_val_transformed, y_val),
)

# Создаём полный пайплайн для сохранения
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", catboost_model)
])

# Сохраняем пайплайн
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Пайплайн с обученной моделью сохранён в 'pipeline.pkl'")