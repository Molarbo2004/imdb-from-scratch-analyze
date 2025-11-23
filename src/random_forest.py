import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

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

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
print("Обучение Методом случайного леса ...")
model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

# Оценка точности
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report для Naive Bayes:")
print(classification_report(y_test, y_pred))


# Предсказания вероятностей для первых 5 текстов
y_pred_proba = model.predict_proba(X_test_vec[:5])

# Выводим результаты
print("\n" + "="*60)
print("ПЕРВЫЕ 5 ТЕКСТОВ И ИХ ПРЕДСКАЗАНИЯ:")
print("="*60)

for i in range(5):
    print(f"\nТекст {i+1}:")
    print(f"   {X_test.iloc[i][:100]}...")  # Первые 100 символов
    
    print(f"Предсказание: {'ПОЗИТИВНЫЙ' if y_pred[i] == 1 else 'НЕГАТИВНЫЙ'}")
    print(f"Вероятности: [Негативный: {y_pred_proba[i][0]:.3f}, Позитивный: {y_pred_proba[i][1]:.3f}]")
    print(f"Реальный класс: {'ПОЗИТИВНЫЙ' if y_test.iloc[i] == 1 else 'НЕГАТИВНЫЙ'}")
    print(f"Совпадение: {'ВЕРНО' if y_pred[i] == y_test.iloc[i] else 'ОШИБКА'}")


# Матрица ошибок графиком
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['Негативные', 'Позитивные'])
disp.plot(cmap='Blues')
plt.title('Матрица ошибок')
plt.show()


# ROC кривая 
y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC-AUC: {roc_auc:.4f}")  

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()


# GRIDSEARCH ДЛЯ RANDOM FOREST

print("Поиск оптимальных параметров для Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_vec, y_train)

print("Лучшие параметры:", grid.best_params_)
print("Лучший ROC-AUC:", grid.best_score_)

# ИСПОЛЬЗУЕМ ЛУЧШУЮ МОДЕЛЬ
best_model = grid.best_estimator_

# Оценка точности лучшей модели
y_pred = best_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report для Random Forest GridSearch:")
print(classification_report(y_test, y_pred))

# ROC-кривые для сравнения с Grid и Без

models = {
    "RandomForest": model,
    "GridForest": grid.best_estimator_
}


plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')

colors = ["darkorange", "green"]  # Фиксированные цвета для каждой модели

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
plt.title('ROC-кривые GridSearchForest и RandomForest')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

