### ТУТ У НАС СРАВНИВАЮТСЯ МОДЕЛИ: СЛУЧАНЫЙ ЛЕС, РЕГРЕССИЯ И НАИВНЫЙ БАЙЕС  ### 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Загрузка данных

df = pd.read_csv('data/imdb_reviews.csv', encoding='utf-8')


def preprocessor(text):
    if isinstance(text, float):  # Защита от NaN
        return ""
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]*>', '', text)
    
    # Ищем смайлики (эмотиконы) 
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    
    # Удаляем все не-буквенные символы 
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', '')
    
    return text.strip()


# Очищаем все отзывы
print("Очищаем текст...")
df['cleaned_review'] = df['review'].apply(preprocessor)


# Разделение на обучающую и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
)


# Векторизация текста

vectorizer = TfidfVectorizer(
    max_features=5000,      # берем 5000 самых частых слов
    ngram_range=(1, 3),     # учитываем 1,2,3-граммы
    stop_words='english'    # удаляем слова-паразиты
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


models = {
    "LogisticRegression": LogisticRegression(),
    "NaiveBayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier()
}

best_model = []
accuracy_massiv = []
best_model_name = ""
best_accuracy = 0
#1. Сравнение точности
for key, value in models.items(): 
    value.fit(X_train_vec, y_train)
    y_pred = value.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность {key} метода: {accuracy}")
    accuracy_massiv.append(accuracy)
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_model_name = key

best_model.append(max(accuracy_massiv))
print(f"Лучшая модель:", best_model_name, "Ее точность:", best_model[0])

# 2. ROC-кривые всех моделей

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')

colors = ["darkorange", "green", "red"]  # Фиксированные цвета для каждой модели

for i, (key, model) in enumerate(models.items()): 
    # Получаем вероятности для положительного класса
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Рисуем кривую с соответствующим цветом
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'{key} (AUC = {roc_auc:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые всех моделей')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()



# =============================================
# АНАЛИЗ ОШИБОК: ЛОЖНО-ПОЗИТИВНЫЕ И ЛОЖНО-НЕГАТИВНЫЕ ОТЗЫВЫ
# =============================================

print("\n" + "="*70)
print("🔍 АНАЛИЗ ОШИБОК МОДЕЛИ")
print("="*70)


# ВЫЯВЛЕНИЕ ОШИБОК

# 1. Ложно-позитивные (на самом деле НЕГАТИВНЫЕ, но предсказано ПОЗИТИВНЫЕ)
false_positives = X_test[(y_test == 0) & (y_pred == 1)]
print(f"\n ЛОЖНО-ПОЗИТИВНЫЕ (негативные отзывы, распознанные как позитивные): {len(false_positives)} шт.")
print("Примеры:")
for i, (_, text) in enumerate(false_positives[:3].items()):
    original_review = df.loc[df['cleaned_review'] == text, 'review'].iloc[0]  # оригинальный текст с HTML/пунктуацией
    print(f"  {i+1}. {original_review[:150]}...")

# 2. Ложно-негативные (на самом деле ПОЗИТИВНЫЕ, но предсказано НЕГАТИВНЫЕ)
false_negatives = X_test[(y_test == 1) & (y_pred == 0)]
print(f"\n ЛОЖНО-НЕГАТИВНЫЕ (позитивные отзывы, распознанные как негативные): {len(false_negatives)} шт.")
print("Примеры:")
for i, (_, text) in enumerate(false_negatives[:3].items()):
    original_review = df.loc[df['cleaned_review'] == text, 'review'].iloc[0]
    print(f"  {i+1}. {original_review[:150]}...")

# 3. Интерпретация: ищем подозрительные паттерны
print("\n" + "-"*70)
print(" ИНТЕРПРЕТАЦИЯ:")
print("- Ложные позитивы часто содержат сарказм или иронию (например, 'Oh great...').")
print("- Ложные негативы могут быть краткими, с неоднозначной лексикой или без явных 'ключевых' слов.")
print("- Модель не учитывает порядок слов и контекст без n-грамм.")
print("- Эмодзи иногда спасают, но не всегда компенсируют иронию.")












