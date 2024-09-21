import joblib
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Імпорт класу TextPreprocessor з файлу text_preprocessing.py
from text_preprocessing import TextPreprocessor 

class TextVectorizer:
    def __init__(self):    
        # Ініціалізація класу TextVectorizer
        self.load_vectorizer()  # Завантаження моделі для векторизації тексту
        self.textPreprocessor = TextPreprocessor()  # Ініціалізація TextPreprocessor для попередньої обробки тексту
        
    # Функція для встановлення моделі для векторизації тексту
    def set_vectorizer(self, ngram_range=(1, 1), max_df=1.0, min_df=0, norm='l2'):
        # Створення об'єкта TfidfVectorizer для векторизації тексту з заданими параметрами
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                      max_df=max_df,
                                      min_df=min_df,
                                      norm=norm)
        
        # Ініціалізація SMOTE для оверсемплінгу
        self.smote = SMOTE(random_state=42, k_neighbors=1)
      
    # Функція для навчання моделі для векторизації тексту
    def train_vectorizer(self, test_size=0.2, random_state=42):
        # Завантаження набору даних
        dataset = pd.read_csv("dataset/pre_semeval_dataset.csv", delimiter='\t', encoding='utf-8')
        dataset = dataset.dropna()

        # Видобуття тексту та міток з набору даних
        self.text = dataset["neg_cleaned_text"]
        self.label = dataset["label"]

        # Встановлення моделі для векторизації тексту та моделі для оверсемплінгу
        self.set_vectorizer()
        self.vectorized_text = self.vectorizer.fit_transform(self.text)
        self.X, self.y = self.smote.fit_resample(self.vectorized_text, self.label)

        # Розділення даних на навчальний та тестувальний набори
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                stratify=self.y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)
      
        # Збереження навченої моделі для векторизації тексту
        self.save_vectorizer()

    # Функція для векторизації тексту
    def vectorize_text(self, text):
        # Попередня обробка вхідного тексту
        text = self.textPreprocessor.cleaning_text(text)  # Очищення тексту
        text = self.textPreprocessor.detect_negations(text)  # Виявлення заперечень у тексті
        
        # Векторизація попередньо обробленого тексту
        return self.vectorizer.transform([text])
     
    # Функція для отримання навчальних та тестових наборів
    def get_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    # Функція для збереження навченої моделі для векторизації тексту
    def save_vectorizer(self):
        # Збереження моделі векторизації за допомогою бібліотеки joblib
        joblib.dump(self.vectorizer, "models/vectorizer.pkl") 

    # Функція для завантаження навченої моделі для векторизації тексту
    def load_vectorizer(self):
        # Перевірка наявності файлу з моделлю
        if os.path.exists("models/vectorizer.pkl"): 
            # Завантаження моделі векторизації за допомогою joblib
            self.vectorizer = joblib.load("models/vectorizer.pkl")
        else:
            # Повідомлення про відсутність моделі
            print("No vectorizer model found.")
            # Якщо модель не знайдено, встановити vectorizer як None
            self.vectorizer = None
