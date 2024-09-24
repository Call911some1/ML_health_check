import pandas as pd
import numpy as np
import streamlit as st
from catboost import CatBoostClassifier  # Модель CatBoost

# Preprocessing and Models
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Заголовок и приветствие
st.title("Health Check: Предсказание Сердечных Заболеваний")
st.write("""
    Это приложение предсказывает вероятность сердечного заболевания на основе ваших данных.
    Пожалуйста, заполните форму ниже и нажмите "Предсказать".
""")

# Ввод данных от пользователя
age = st.slider('Возраст (Чем старше пациент, тем выше риск сердечных заболеваний)', 20, 100, 50)
sex = st.selectbox('Пол (Мужчины чаще подвержены сердечным заболеваниям)', ('M', 'F'))  # Мужчина или Женщина
st.write("Пол: M - Мужчина, F - Женщина")

chest_pain = st.selectbox('Тип боли в груди (Тип боли может указывать на риск сердечного заболевания)', 
                          ('TA', 'ATA', 'NAP', 'ASY'))
st.write("TA: Типичная стенокардия, ATA: Атипичная стенокардия, NAP: Нестенокардическая боль, ASY: Без симптомов")

resting_bp = st.number_input('Давление в покое (Нормальное давление ~120)', value=120)
st.write("Обычное давление: 90-120 мм рт. ст.")

cholesterol = st.number_input('Уровень холестерина (Нормальный уровень до 200)', value=200)
st.write("Холестерин: уровни выше 200 мг/дл повышают риск сердечных заболеваний")

fasting_bs = st.selectbox('Сахар натощак больше 120 мг/дл? (Повышенный уровень сахара может указывать на диабет)', 
                          ('Нет', 'Да'))
st.write("Если уровень сахара больше 120 мг/дл: Да = 1, Нет = 0")

max_hr = st.slider('Максимальная частота сердечных сокращений (Нормальная частота для здорового человека)', 60, 220, 150)
st.write("Максимальная частота сердечных сокращений в норме варьируется в зависимости от возраста")

exercise_angina = st.selectbox('Ангина при физической нагрузке? (Положительный ответ увеличивает риск)', ('N', 'Y'))
st.write("Ангина: Y - Да, N - Нет")

oldpeak = st.number_input('Oldpeak (Измерение депрессии ST, которая может указывать на ишемию)', value=0.0)
st.write("Oldpeak: показатели депрессии ST выше 1 указывают на риск")

resting_ecg = st.selectbox('ЭКГ в покое (Отклонения могут свидетельствовать о проблемах с сердцем)', 
                           ('Normal', 'ST', 'LVH'))
st.write("ST: ST-T волновые аномалии, LVH: гипертрофия левого желудочка, Normal: нормальная ЭКГ")

st_slope = st.selectbox('Наклон ST сегмента (Показатель состояния сердечной мышцы)', ('Up', 'Flat', 'Down'))
st.write("Up: восходящий сегмент ST (норма), Flat: плоский сегмент ST (риск), Down: нисходящий сегмент ST (высокий риск)")

# Маппинг значений
fasting_bs = 1 if fasting_bs == 'Да' else 0  # Переводим "Да/Нет" в 1 и 0

# Преобразование данных пользователя в DataFrame
user_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'RestingECG': [resting_ecg],
    'ST_Slope': [st_slope],
    'HR_Difference': [(220 - age) - max_hr]  # Добавляем колонку HR_Difference
})


# Ожидаемые колонки для обработки
expected_columns = [
    'Age', 'HR_Difference', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
    'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS'
]

# Подготовка данных с помощью того же Pipeline
my_imputer = ColumnTransformer(
    transformers=[
        ('num_imputer_age', SimpleImputer(strategy='median'), ['Age']),
        ('num_imputer_rest', SimpleImputer(strategy='median', missing_values=0), ['RestingBP']),
        ('num_imputer_chol', SimpleImputer(strategy='median', missing_values=0), ['Cholesterol']),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

scaler_and_encoder = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'HR_Difference', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']),
        ('cat', OneHotEncoder(sparse_output=False), ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])  # Добавляем sparse_output=False
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

preprocessor = Pipeline([
    ('imputer', my_imputer),
    ('scaler_encoder', scaler_and_encoder)
])

# Загружаем тренировочные данные, чтобы обучить preprocessor
train = pd.read_csv('heart.csv')
train['HR_Difference'] = (220 - train['Age']) - train['MaxHR']
X = train.drop('HeartDisease', axis=1)
y = train['HeartDisease']

# Обучаем preprocessor на тренировочных данных
preprocessor.fit(X, y)

# Применение предобработки к пользовательским данным
try:
    user_data_transformed = preprocessor.transform(user_data)
except ValueError as e:
    st.error(f"Error during preprocessing: {e}")

# Загрузка обученной модели
model = CatBoostClassifier(verbose=0)
try:
    model.load_model('catboost_model.cbm')  # Загружаем модель
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")

# Предсказание
if st.button('Предсказать'):
    try:
        prediction = model.predict(user_data_transformed)
        probability = model.predict_proba(user_data_transformed)[0][1]

        # Вывод результата
        if prediction == 1:
            st.error(f"Высокая вероятность сердечного заболевания! Вероятность: {probability:.2f}")
        else:
            st.success(f"Низкая вероятность сердечного заболевания. Вероятность: {probability:.2f}")
    except Exception as e:
        st.error(f"Ошибка во время предсказания: {e}")
