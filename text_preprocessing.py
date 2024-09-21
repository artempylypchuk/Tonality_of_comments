import re	
import nltk
import contractions
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

# Завантаження необхідних ресурсів NLTK
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

class TextPreprocessor:
	def __init__(self):	
		# Ініціалізація лематизатора та стемера
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = PorterStemmer()
		# Отримання списку стоп-слів
		self.s_words = stopwords.words('english')
		self.n_words = 3  # Кількість слів після заперечення, які будуть перевірені на антоніми
		self.negation_words = ["not", "no", "n't"]  # Список слів, що вказують на заперечення
		self.n_prefixes = ["a", "un", "in", "im",  
				   "il", "ir", "non", "dis",
				   "less", "ab", "an",
				   "mis", "anti", "ig"] # Префікси для створення негативних форм слів

	# Функція для переведення тексту в нижній регістр
	def lower_text(self, text):
		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (lower_text): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Переведення тексту в нижній регістр та повернення результату
			return text.lower()

	# Функція для видалення URL-адрес та доменів з тексту
	def remove_url_and_domain(self, text):
		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (remove_url_and_domain): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Видалення URL-адрес з тексту
			text = re.sub(r'\b(?:https?|ftp)://\S+|www\.\S+', '', text)
			# Видалення доменів з тексту
			text = re.sub(r'(?:\S+\.)+(?:com|es|org|net)\b', '', text)
			return text  # Повернення очищеного тексту

	# Функція для видалення пунктуації та цифр з тексту
	def remove_punctuation_and_digits(self, text):
		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (remove_punctuation_and_digits): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Видалення табуляції та заміна її пробілом
			cleaned_text = re.sub(r'\t', ' ', text)
			# Видалення згадок (@username)
			cleaned_text = re.sub(r'@(\w+)', '', cleaned_text)
			# Видалення слешів (/word)
			cleaned_text = re.sub(r'/(\w+)', '', cleaned_text)
			# Видалення хештегів (#word)
			cleaned_text = re.sub(r'#(\w+)', '', cleaned_text)
			# Видалення апострофів та зворотних апострофів
			cleaned_text = re.sub(r"'|`", '', cleaned_text)
			# Видалення всіх інших пунктуаційних знаків, окрім пробілів та літер
			cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
			# Видалення слова "rt", яке часто зустрічається в ретвітах
			cleaned_text = re.sub(r'rt ', '', cleaned_text)
			# Видалення всіх однолітерних слів
			cleaned_text = re.sub(r'\b\w\b', '', cleaned_text)
			# Видалення всіх цифр
			cleaned_text = re.sub(r'\d+', '', cleaned_text)
			# Заміна декількох пробілів на однин пробіл
			cleaned_text = re.sub(r' +', ' ', cleaned_text)
			# Видалення пробілів з початку та кінця тексту
			cleaned_text = cleaned_text.strip()

			# Повернення очищеного тексту
			return cleaned_text

	# Функція для видалення стоп-слів з тексту
	def remove_stop_words(self, text):
		new_tokens = []  # Ініціалізація списку для зберігання нових токенів без стоп-слів

		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (remove_stop_words): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Токенізація тексту
			tokens = word_tokenize(text)
			# Перебір токенів
			for token in tokens:
				# Додавання токену до списку, якщо він не є стоп-словом
				if token not in self.s_words:
					new_tokens.append(token)

		# Об'єднання нових токенів у рядок
		text = ' '.join(new_tokens)

		# Повернення тексту без стоп-слів
		return text

	# Функція для повного очищення тексту (переведення в нижній регістр, видалення URL та доменів, пунктуації та цифр)
	def cleaning_text(self, text):
		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (cleaning_text): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Переведення тексту в нижній регістр
			text = self.lower_text(text)
			# Видалення URL-адрес та доменів
			text = self.remove_url_and_domain(text)
			# Видалення пунктуації та цифр
			text = self.remove_punctuation_and_digits(text)
			# Виправлення помилок у тексті (наприклад, скорочень)
			text = contractions.fix(text)

			# Повернення очищеного тексту
			return text

	# Функція для отримання негативної форми слова
	def get_negative_form(self, word):
		# Перевірка чи слово є рядком
		if type(word) is not str:
			# Виведення повідомлення про помилку, якщо тип слова не рядок
			print("Function (get_negative_form): Type of word is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Проходження по всім можливим префіксам, які зберігаються в self.n_prefixes
			for prefix in self.n_prefixes:
				negative_word = prefix + word  # Додавання префіксу до слова
				# Перевірка чи існує синсет (значення) для створеного слова в словнику WordNet
				if wordnet.synsets(negative_word):
					return negative_word  # Повернення негативної форми слова, якщо знайдено
			# Якщо жодна негативна форма не знайдена, повернути оригінальне слово
			return word

	# Функція для отримання антоніма слова
	def get_antonym(self, word):
		# Перевірка чи слово є рядком
		if type(word) is not str:
			# Виведення повідомлення про помилку, якщо тип слова не рядок
			print("Function (get_antonym): Type of word is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			antonyms = []  # Створення списку для зберігання антонімів
			# Проходження по всіх синсетах (групах синонімів) слова
			for syn in wordnet.synsets(word.lower()):
				for lemma in syn.lemmas():
					for antonym in lemma.antonyms():
						antonyms.append(antonym.name())  # Додавання антонімів до списку

			if antonyms == []:
				# Якщо антонімів немає, отримуємо негативну форму слова
				return self.get_negative_form(word)
			else:
				# Повернення першого знайденого антоніма
				return antonyms[0]

	# Функція для виявлення заперечень у тексті
	def detect_negations(self, text):
		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (detect_negations): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			tokens = word_tokenize(text)  # Токенізація тексту
			len_tokens = len(tokens)
			for token_index in range(len_tokens):
				if tokens[token_index] in self.negation_words:  # Перевірка чи токен є словом заперечення
					start_index = token_index + 1
					end_index = start_index + self.n_words
					for index in range(start_index, end_index):
						if index < len_tokens:
							tokens[index] = tokens[index] + "_neg"  # Додавання суфіксу "_neg" до наступних токенів

			new_tokens = []

			# Перебір токенів
			for token in tokens:
				if token.endswith("_neg"):  # Перевірка чи токен має суфікс "_neg"
					new_token = token.replace("_neg", "")  # Видалення суфіксу "_neg" з токену
					new_tokens.append(self.get_antonym(new_token))  # Заміна слова на антонім
				else:
					new_tokens.append(token)

			# Об'єднання нових токенів у рядок
			text = ' '.join(new_tokens)
			text = re.sub(r' +', ' ', text)  # Заміна декількох пробілів на один
			text = self.remove_stop_words(text)  # Видалення стоп-слів

			# Повернення тексту з виявленими запереченнями
			return text

	# Функція для лематизації тексту
	def lemmatize(self, text):
		# Перевірка чи текст є рядком
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (lemmatize): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			tokens = word_tokenize(text)  # Токенізація тексту
			tagged_tokens = pos_tag(tokens)  # Отримання тегів частини мови для токенів

			lemmatized_tokens = []

			# Проходження по кожному токену та його тегу
			for token, tag in tagged_tokens:
				wn_tag = None

				# Визначення відповідного тегу WordNet для лематизації
				if tag.startswith('J'):
					wn_tag = wordnet.ADJ  # Тег для прикметників
				elif tag.startswith('V'):
					wn_tag = wordnet.VERB  # Тег для дієслів
				elif tag.startswith('N'):
					wn_tag = wordnet.NOUN  # Тег для іменників
				elif tag.startswith('R'):
					wn_tag = wordnet.ADV  # Тег для прислівників

				# Лематизація токену з використанням відповідного тегу
				if wn_tag is not None:
					lemma = self.lemmatizer.lemmatize(token, pos=wn_tag)
				else:
					lemma = self.lemmatizer.lemmatize(token)  # Лематизація без спеціального тегу

				lemmatized_tokens.append(lemma)  # Додавання лематизованого токену до списку

			# Об'єднання лематизованих токенів у текст
			lemmatized_text = ' '.join(lemmatized_tokens)

			# Повернення лематизованого тексту
			return lemmatized_text

	# Функція для стемінгу тексту
	def stem(self, text):
		# Перевірка чи вхідний текст є строкою
		if type(text) is not str:
			# Виведення повідомлення про помилку, якщо тип тексту не рядок
			print("Function (stem): Type of text is not str")
			return None  # Повернення None у разі неправильного типу
		else:
			# Токенізація тексту
			tokens = word_tokenize(text)
			stemmed_tokens = []

			# Проходження по кожному токену
			for token in tokens:
				# Стемінг токену та переведення його в нижній регістр
				stemmed_tokens.append(self.stemmer.stem(token).lower())

			# Об'єднання стемінгованих токенів у текст
			stemmed_text = ' '.join(stemmed_tokens)

			# Повернення стемінгованого тексту
			return stemmed_text