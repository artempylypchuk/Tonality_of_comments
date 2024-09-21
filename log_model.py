from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

# Імпорт класу TextVectorizer з файлу text_vectorizing.py
from text_vectorizing import TextVectorizer

class MultinomialLR():
    def __init__(self):
        # Завантаження моделі
        self.load_model()
        
        # Ініціалізація об'єкту TextVectorizer для векторизації тексту
        self.vectorizer = TextVectorizer()

    # Функція для тренування моделі
    def train_model(self):
        # Ініціалізація мультиноміальної логістичної регресії з певними параметрами
        self.mlr = LogisticRegression(multi_class='multinomial',
                                      solver='sag',
                                      penalty='l2',
                                      max_iter=1000,
                                      random_state=42,
                                      C=1,
                                      tol=1e-4)
        
        # Навчання моделі для векторизації тексту за допомогою TextVectorizer
        self.vectorizer.train_vectorizer()
        
        # Отримання тренувального та тестового наборів даних з TextVectorizer
        X_train, X_test, y_train, y_test = self.vectorizer.get_train_test()
        
        # Тренування моделі на тренувальних даних
        self.mlr.fit(X_train, y_train)

        # Передбачення міток класів для тестових даних
        y_pred = self.mlr.predict(X_test)

        # Формування звіту про класифікацію
        cr = classification_report(y_test,
                                   y_pred,
                                   target_names=['negative', 'neutral', 'positive'])

        # Виведення звіту про класифікацію
        print(f"\nClassification report: \n{cr}\n")
        
        # Побудова матриці помилок
        self.draw_confusion_matrix(y_pred, y_test)
        
        # Збереження навченої моделі
        self.save_model()

    # Функція для побудови матриці помилок
    def draw_confusion_matrix(self, y_pred, y_test):
        # Обчислення матриці помилок
        cm = confusion_matrix(y_pred, y_test)
        labels = ['negative', 'neutral', 'positive']  # Мітки класів

        # Побудова теплокарти
        plt.figure(figsize=(8, 6))  # Розміри графіку
        sns.set(font_scale=1.2)  # Масштаб шрифту
        ax = sns.heatmap(cm,  # Виведення теплокарти
                         annot=True,  # Виведення значень в клітинах
                         fmt="d",  # Формат значень: десятковий цілочисельний
                         cmap="Blues",  # Колірна карта
                         annot_kws={"size": 14},  # Розмір шрифту анотацій
                         xticklabels=labels,  # Мітки по осі X
                         yticklabels=labels)  # Мітки по осі Y
        sns.set(font_scale=1.4)  # Масштаб шрифту
        ax.invert_xaxis()  # Обертання вісі X

        # Додавання підписів до вісей та заголовку графіку
        plt.xlabel('Predicted polarity', fontsize=16)  # Підпис для осі X
        plt.ylabel('Expected polarity', fontsize=16)  # Підпис для осі Y
        plt.title('Confusion Matrix', fontsize=18)  # Заголовок графіку
        plt.show()  # Показати графік

    # Функція для збереження моделі
    def save_model(self):
        # Збереження моделі за допомогою joblib
        joblib.dump(self.mlr, "models/mlr.pkl")

    # Функція для завантаження моделі
    def load_model(self):
        # Перевірка наявності файлу з моделлю
        if os.path.exists("models/mlr.pkl"):
            # Завантаження моделі за допомогою joblib
            self.mlr = joblib.load("models/mlr.pkl") 
        else:
            # Повідомлення про відсутність моделі
            print("No MLR model found.")
                    
    # Функція для передбачення ймовірностей тексту
    def predict_pr(self, text):
        # Векторизація попередньо обробленого тексту
        vectorized_text = self.vectorizer.vectorize_text(text)
        # Передбачення ймовірностей класів за допомогою моделі
        return self.mlr.predict_proba(vectorized_text)

    # Функція для передбачення тональності тексту
    def predict_ton(self, text):
        # Отримання передбачень ймовірностей
        prediction = self.predict_pr(text)
        tonality = ['negative', 'neutral', 'positive']  # Мітки класів
        
        # Передбачення тональності тексту за допомогою максимальної ймовірності
        return tonality[np.argmax(prediction)]