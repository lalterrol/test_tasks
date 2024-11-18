import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Загрузка тренировочных данных
train_df = pd.read_csv('D:\\task1\\train.csv')
train_df.dropna(inplace=True)
train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})

# Проверка на наличие NaN значений и их удаление
train_df = train_df.dropna(subset=['label'])

# Загрузка тестовых данных
test_df = pd.read_csv('D:\\task1\\test.csv')
test_df.dropna(inplace=True)
test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})

# Проверка на наличие NaN значений и их удаление
test_df = test_df.dropna(subset=['label'])

# Объединение тренировочных и тестовых данных для визуализации и анализа
df = pd.concat([train_df, test_df], ignore_index=True)

# Визуализация распределения меток
sns.countplot(x=df['label'])
plt.show()

# Токенизация и подготовка данных
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_df['email'])  # Используем только тренировочные данные для обучения токенизатора
train_sequences = tokenizer.texts_to_sequences(train_df['email'])
test_sequences = tokenizer.texts_to_sequences(test_df['email'])

# Найти максимальную длину последовательностей
max_len = max(len(seq) for seq in train_sequences + test_sequences)

# Добавляем padding к последовательностям
X_train = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len, padding='post')
y_train = train_df['label'].values
X_test = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len, padding='post')
y_test = test_df['label'].values

# Проверка на наличие NaN значений в метках
print(f"NaN в тренировочных метках: {np.isnan(y_train).sum()}")
print(f"NaN в тестовых метках: {np.isnan(y_test).sum()}")

# Оверсэмплинг с использованием SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Преобразование меток в целые числа
y_train_res = y_train_res.astype(int)

# Проверка нового распределения меток после оверсэмплинга
print(f"Распределение классов после оверсэмплинга: {np.bincount(y_train_res)}")

# Создание и обучение модели Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)

# Предсказание и оценка модели Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Точность модели Random Forest: {accuracy_rf}")
print("Отчет по классификации Random Forest:\n", classification_report(y_test, y_pred_rf))

# Вычисление и визуализация матрицы ошибок для Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(f"Confusion Matrix (Test):\n{conf_matrix_rf}")
