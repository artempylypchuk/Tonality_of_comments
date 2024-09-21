import pandas as pd
import matplotlib.pyplot as plt
from text_preprocessing import TextPreprocessor  # Імпорт класу TextPreprocessor з файлу text_preprocessing.py

# створення екземпляру класу TextPreprocessor
textPreprocessor = TextPreprocessor() 

# Завантаження датасету
dataset = pd.read_csv("dataset/semeval-2017.csv", delimiter='\t', encoding='utf-8')

# Виведення кількості елементів у кожній категорії міток
print(dataset['label'].value_counts())

# Створення списку для позначень класів
s_label = ['Neutral', 'Positive', 'Negative']

# Отримання міток класів та їх кількостей, сортування за індексом, конвертація індексів у слова
x_axis = dataset['label'].value_counts().sort_index().index
x_axis = [s_label[x] for x in x_axis]

# Отримання кількостей класів, сортування за індексом
y_axis = dataset['label'].value_counts().sort_index()

# Створення стовпчикової діаграми з мітками по x_axis та кількостями по y_axis
plt.bar(x_axis, y_axis)
# Налаштування розміру шрифту для підписів по осі x
plt.xticks(fontsize=12)
# Налаштування розміру шрифту для підписів по осі y
plt.yticks(fontsize=12)

# Додавання заголовку до графіку
plt.title('The count of tonality classes', fontsize=16)
# Додавання підпису до осі x
plt.xlabel('Tonality', fontsize=14)
# Додавання підпису до осі y
plt.ylabel('Count', fontsize=14)

# Відображення графіку
plt.show()

# Фільтрація даних за міткою класу
d_pos = dataset[dataset['label'] == 1].reset_index(drop=True)
d_neu = dataset[dataset['label'] == 0].reset_index(drop=True)
d_neg = dataset[dataset['label'] == -1].reset_index(drop=True)

# Виконання очищення тексту для кожного класу
d_pos['cleaned_text'] = d_pos['text'].apply(textPreprocessor.cleaning_text)
d_neu['cleaned_text'] = d_neu['text'].apply(textPreprocessor.cleaning_text)
d_neg['cleaned_text'] = d_neg['text'].apply(textPreprocessor.cleaning_text)

# Виконання виявлення заперечень у тексті для кожного класу
d_pos['neg_cleaned_text'] = d_pos['cleaned_text'].apply(textPreprocessor.detect_negations)
d_neg['neg_cleaned_text'] = d_neg['cleaned_text'].apply(textPreprocessor.detect_negations)
d_neu['neg_cleaned_text'] = d_neu['cleaned_text'].apply(textPreprocessor.remove_stop_words)

# Виконання лематизації тексту для кожного класу
d_pos['lemmatized'] = d_pos['neg_cleaned_text'].apply(textPreprocessor.lemmatize)
d_neu['lemmatized'] = d_neu['neg_cleaned_text'].apply(textPreprocessor.lemmatize)
d_neg['lemmatized'] = d_neg['neg_cleaned_text'].apply(textPreprocessor.lemmatize)

# Виконання стемінгу тексту для кожного класу
d_pos['stemmed'] = d_pos['neg_cleaned_text'].apply(textPreprocessor.stem)
d_neu['stemmed'] = d_neu['neg_cleaned_text'].apply(textPreprocessor.stem)
d_neg['stemmed'] = d_neg['neg_cleaned_text'].apply(textPreprocessor.stem)

# Об'єднання оброблених даних для всіх класів
dataset = pd.concat([d_pos, d_neu, d_neg])
# Випадкове перемішування даних та збереження порядку індексів
dataset = dataset.sample(frac=1).reset_index(drop=True)
# Видалення рядків з нульовими значеннями
dataset = dataset.dropna()

# Збереження обробленого датасету у CSV-файл
dataset.to_csv("dataset/pre_semeval_dataset.csv", sep='\t', index=False)