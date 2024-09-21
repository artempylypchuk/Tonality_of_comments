import os
import sys
import json
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Імпорт класу MultinomialLR з файлу log_model.py
from log_model import MultinomialLR

class TextAnalysisApp(tk.Tk):	
	def __init__(self):
		# Виклик конструктора батьківського класу
		super().__init__()

		# Заголовок для графічного інтерфейсу
		self.title("Graphical interface for analyzing the tonality of text")

		# Словник для зберігання даних про тональність тексту
		self.tonality = {'negative': 0, 'neutral': 0, 'positive': 0}

		# Екземпляр моделі MultinomialLR для аналізу тональності
		self.mlr = MultinomialLR()

		# Ініціалізація змінних для додаткових вікон та текстових полів
		self.json_window = None
		self.json_text_area = None
		self.analysis_window = None

		# Видалення існуючого файлу "text_tonality.json", якщо такий існує
		if os.path.isfile("text_tonality.json"):
			os.remove("text_tonality.json")

		# Створення віджетів для інтерфейсу користувача
		self.create_widgets()

		# Заборона зміни розмірів вікна користувачем
		self.resizable(False, False)

		# Обробник події закриття вікна
		self.protocol("WM_DELETE_WINDOW", self.on_closing)

	# Функція для закриття вікна та завершення роботи програми
	def on_closing(self):
		self.destroy()
		sys.exit()

	# Функція для створення віджетів інтерфейсу користувача
	def create_widgets(self):
		# Створення фрейму для кнопок та його розміщення
		self.button_frame = tk.Frame(self)  # Створення фрейму для кнопок
		self.button_frame.pack(pady=10)  # Розміщення фрейму у вікні з певним відступом від країв

		# Кнопка для вибору .txt файлу
		self.browse_button = tk.Button(self.button_frame, text="Choose .txt file", command=self.browse_file)  # Створення кнопки
		self.browse_button.pack(side='left', padx=10)  # Розміщення кнопки зліва з відступом від краю

		# Кнопка для введення тексту вручну
		self.text_entry_button = tk.Button(self.button_frame, text="Input text", command=self.open_text_entry_window)  # Створення кнопки
		self.text_entry_button.pack(side='left', padx=10)  # Розміщення кнопки зліва з відступом від краю

		# Кнопка для перегляду прогнозу тональності тексту
		self.view_json_button = tk.Button(self.button_frame, text="View prediction of text", command=self.view_json_file)  # Створення кнопки
		self.view_json_button.pack(side='left', padx=10)  # Розміщення кнопки зліва з відступом від краю

		# Вимкнення кнопки перегляду прогнозу, якщо файл "text_tonality.json" не існує
		if not os.path.isfile('text_tonality.json'):  # Перевірка наявності файлу
			self.view_json_button.config(state='disabled')  # Вимкнення кнопки, якщо файл не існує

		# Створення фрейму для графіка та його розміщення
		self.graph_frame = tk.Frame(self)  # Створення фрейму для графіка
		self.graph_frame.pack(pady=(5, 0))  # Розміщення фрейму з відступом від верхнього краю

		# Створення графіка для відображення кількості класів тональності
		self.fig, self.ax = plt.subplots()  # Створення об'єктів графіка та вісей
		self.ax.set_title("The count of tonality classes")  # Встановлення заголовку графіка
		self.ax.set_xlabel("Tonality")  # Встановлення підпису для вісі X
		self.ax.set_ylabel("Count")  # Встановлення підпису для вісі Y
		self.ax.set_xticks([0, 1, 2])  # Встановлення міток на вісі X
		self.ax.set_ylim(0, 1)  # Встановлення меж для вісі Y
		self.ax.set_xticklabels(["negative", "neutral", "positive"])  # Встановлення підписів на вісі X
		self.bar_plot = self.ax.bar(["negative", "neutral", "positive"], [0, 0, 0], color='blue')  # Створення стовпчикової діаграми

		# Вбудовування графіка в Tkinter інтерфейс
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)  # Створення віджета графіка
		self.canvas.draw()  # Оновлення графіка
		self.canvas.get_tk_widget().pack()  # Розміщення графіка у вікні

	# Функція для вибору файлу та запуску аналізу
	def browse_file(self):
		file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
		if file_path:
			self.start_analysis()
			self.after(20, lambda: self.analyze_file(file_path))

	# Функція для аналізу текстового файлу
	def analyze_file(self, file_path):
		try:
			with open(file_path, 'r', encoding='utf-8') as file:
				self.tonality = {'negative': 0, 'neutral': 0, 'positive': 0}
				self.text_pred = {}
				
				# Читання файлу по рядках а визначення тональності
				for line in file:
					if line:
						line = line.replace('\n', '')
						prediction = self.mlr.predict_ton(line)
						self.tonality[prediction] += 1
						self.text_pred[line] = prediction
				
				# Оновлення графіка з новими даними
				self.update_plot()
				
				# Запис прогнозів у файл "text_tonality.json"
				with open('text_tonality.json', 'w') as f:
					json.dump(self.text_pred, f, indent=4)
				
				# Активування кнопки перегляду прогнозу
				self.view_json_button.config(state='normal')
				
				# Оновлення вмісту JSON вікна, якщо воно відкрите
				if self.json_window is not None:
					self.update_json_content(self.text_pred)
		
		# Обробка виключень, що виникають під час читання файлу
		except Exception as e:
			messagebox.showerror("Error", f"An error occurred while reading the file: {e}")
		
		# Кінець аналізу
		finally:
			self.stop_analysis()

	# Функція для відкриття вікна введення тексту
	def open_text_entry_window(self):
		# Вимкнення кнопок "Input text" і "Choose .txt file"
		self.text_entry_button.config(state='disabled')
		self.browse_button.config(state='disabled')

		# Створення нового вікна для введення тексту
		root = tk.Toplevel()
		root.title("Input text")
		
		# Заборона зміни розмірів вікна користувачем
		root.resizable(False, False)

		# Створення текстового поля для введення тексту
		text_entry = tk.Text(root, font=("Courier New", 14))
		text_entry.pack()

		# Функція для відновлення стану кнопок і закриття вікна
		def on_window_close():
			self.text_entry_button.config(state='normal')
			self.browse_button.config(state='normal')
			root.destroy()

		# Призначення обробника події закриття вікна
		root.protocol("WM_DELETE_WINDOW", on_window_close)

		# Функція для збереження та аналізу введеного тексту
		def save_text():
			# Початок аналізу
			self.start_analysis()
			self.after(20, lambda: self.analyze_text(text_entry))

		# Створення кнопки "Analyze" для аналізу введеного тексту
		save_button = tk.Button(root, text="Analyze", command=save_text)
		save_button.pack(padx=10)

	# Функція для аналізу введеного тексту
	def analyze_text(self, text_entry):
		try:
			self.tonality = {'negative': 0, 'neutral': 0, 'positive': 0}
			self.text_pred = {}
			entered_text = text_entry.get("1.0", tk.END)
			
			# Аналіз кожного рядка введеного тексту та його визначення тональності
			for line in entered_text.split('\n'):
				if line:
					prediction = self.mlr.predict_ton(line)
					self.tonality[prediction] += 1
					self.text_pred[line] = prediction

			# Оновлення графіка з новими даними
			self.update_plot()

			# Запис прогнозів у файл "text_tonality.json"
			with open('text_tonality.json', 'w') as f:
				json.dump(self.text_pred, f, indent=4)

			# Активування кнопки перегляду прогнозу
			self.view_json_button.config(state='normal')

			# Оновлення вмісту JSON вікна, якщо воно відкрите
			if self.json_window is not None:
				self.update_json_content(self.text_pred)

		# Обробка виключень, що виникають під час читання тексту
		except Exception as e:
			messagebox.showerror("Error", f"An error occurred while reading the text: {e}")
		
		# Кінець аналізу
		finally:
			self.stop_analysis()
		

	# Функція для перегляду JSON файлу з прогнозами тональності
	def view_json_file(self):
		if os.path.isfile('text_tonality.json'):
			with open('text_tonality.json', 'r') as file:
				data = json.load(file)
				self.show_json_content(data)
		else:
			messagebox.showerror("Error", "The text_tonality.json file does not exist.")

	# Функція для відображення вмісту JSON файлу у новому вікні
	def show_json_content(self, data):
		if self.json_window is None:
			self.json_window = tk.Toplevel(self)
			self.json_window.title("Text Tonality")
			self.json_window.protocol("WM_DELETE_WINDOW", self.on_json_window_close)
			
			self.json_text_area = tk.Text(self.json_window, wrap='word', font=("Courier New", 14))
			self.json_text_area.pack(expand=True, fill='both')
		
		self.update_json_content(data)

	# Функція для оновлення вмісту текстової області з JSON даними
	def update_json_content(self, data):
		self.json_text_area.config(state='normal')
		self.json_text_area.delete('1.0', tk.END)
		self.json_text_area.insert('1.0', json.dumps(data, indent=4))
		self.json_text_area.config(state='disabled')

	# Функція для обробки закриття JSON вікна
	def on_json_window_close(self):
		self.json_window.destroy()
		self.json_window = None

	# Функція для оновлення графіка з новими даними про тональність тексту
	def update_plot(self):
		# Створення списку лічильників для кожного класу тональності
		counts = [self.tonality['negative'], self.tonality['neutral'], self.tonality['positive']]

		# Оновлення висоти стовпців графіка
		for bar, count in zip(self.bar_plot, counts):
			bar.set_height(0)

		# Встановлення максимального значення по осі y
		y_max = max(counts) * 1.1
		if y_max == 0.0:
			self.ax.set_ylim(0, 1)
		else:
			self.ax.set_ylim(0, y_max)

		# Оновлення графіка
		self.bar_plot = self.ax.bar(["negative", "neutral", "positive"], counts, color='blue')
		self.canvas.draw()
		self.canvas.get_tk_widget().pack()

	# Функція для початку аналізу тексту
	def start_analysis(self):
		# Створення вікна для відображення процесу аналізу, якщо воно ще не створене
		if self.analysis_window is None:  # Перевірка, чи вікно аналізу вже створено
			self.analysis_window = tk.Toplevel(self)  # Створення нового вікна
			self.analysis_window.title("")  # Встановлення заголовку вікна
			self.analysis_window.geometry("200x100")  # Встановлення розміру вікна

			# Визначення положення вікна аналізу відносно головного вікна програми
			main_x = self.winfo_rootx()  # Отримання координати X головного вікна
			main_y = self.winfo_rooty()  # Отримання координати Y головного вікна
			main_width = self.winfo_width()  # Отримання ширини головного вікна
			main_height = self.winfo_height()  # Отримання висоти головного вікна

			analysis_width = 200  # Ширина вікна аналізу
			analysis_height = 100  # Висота вікна аналізу

			analysis_x = main_x + (main_width - analysis_width) // 2  # Розрахунок координати X для вікна аналізу
			analysis_y = main_y + (main_height - analysis_height) // 2  # Розрахунок координати Y для вікна аналізу

			self.analysis_window.geometry(f"+{analysis_x}+{analysis_y}")  # Встановлення положення вікна аналізу

			# Відображення напису "Analyzing..." під час аналізу
			self.analysis_label = tk.Label(self.analysis_window, text="Analyzing...", font=("Courier New", 12))  # Створення напису
			self.analysis_label.pack(expand=True)  # Розміщення напису у вікні аналізу

	# Функція для завершення аналізу тексту та закриття вікна процесу аналізу
	def stop_analysis(self):
		if self.analysis_window is not None:
			self.analysis_window.destroy()
			self.analysis_window = None

# Основний блок програми, який створює об'єкт класу TextAnalysisApp та запускає головний цикл подій
if __name__ == "__main__":
	app = TextAnalysisApp()
	app.mainloop()
