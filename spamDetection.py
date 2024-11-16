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
from sklearn.metrics import accuracy_score, classification_report

# Загрузка тренировочных данных
train_df = pd.read_csv('D:\\test_model\\train.csv')
train_df.dropna(inplace=True)
train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})

# Проверка на наличие NaN значений и их удаление
train_df = train_df.dropna(subset=['label'])

# Загрузка тестовых данных
test_df = pd.read_csv('D:\\test_model\\test.csv')
test_df.dropna(inplace=True)
test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})

# Проверка на наличие NaN значений и их удаление
test_df = test_df.dropna(subset=['label'])

# Объединение тренировочных и тестовых данных для визуализации и анализа
df = pd.concat([train_df, test_df], ignore_index=True)

# Визуализация распределения меток
sns.countplot(x=df['label'])
plt.show()

avgWordsLen = round(sum([len(i.split()) for i in df['email']]) / len(df['email']))
print(f"Средняя длина слов: {avgWordsLen}")

unique_words = set()
for sent in df['email']:
    for word in sent.split():
        unique_words.add(word)

total_words_length = len(unique_words)
print(f"Общее количество уникальных слов: {total_words_length}")

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

# Создание и обучение модели Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Предсказание и оценка модели Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Точность модели Random Forest: {accuracy_rf}")
print("Отчет по классификации Random Forest:\n", classification_report(y_test, y_pred_rf))

# Создание и обучение модели логистической регрессии
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Предсказание и оценка модели логистической регрессии
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Точность модели логистической регрессии: {accuracy_lr}")
print("Отчет по классификации логистической регрессии:\n", classification_report(y_test, y_pred_lr))