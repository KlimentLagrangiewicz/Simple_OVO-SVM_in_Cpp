import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import argparse
import timeit
import statistics as st

def main():
	parser = argparse.ArgumentParser(description='OVO-SVM классификатор с настройкой параметров')
	# Обязательные аргументы
	parser.add_argument('--train', type=str, required=True, help='Путь к обучающему CSV-файлу')
	parser.add_argument('--test', type=str, required=True, help='Путь к тестовому CSV-файлу')

	# Параметры модели
	parser.add_argument('--C', type=float, default=1.0, help='Параметр регуляризации (по умолчанию: 1.0)')
	parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help='Тип ядра (по умолчанию: rbf)')
	parser.add_argument('--degree', type=int, default=3, help='Степень полиномиального ядра (по умолчанию: 3)')
	parser.add_argument('--coef0', type=int, default=0, help='Степень полиномиального ядра (по умолчанию: 3)')
	parser.add_argument('--gamma', type=str, default='scale', choices=['scale', 'auto', 'float'], help='Коэффициент ядра (по умолчанию: scale)')
	parser.add_argument('--max_iter', type=int, default=-1, help='Максимальное количество итераций (-1 = без ограничения)')
	parser.add_argument('--tol', type=float, default=1e-3, help='Допуск остановки (по умолчанию: 0.001)')
	parser.add_argument('--class_weight', type=str, default=None, choices=['balanced', None], help='Взвешивание классов (по умолчанию: None)')
	parser.add_argument('--n_jobs', type=int, default=-1, help='Число потоков')
	
	args = parser.parse_args()
	try:
		# Загрузка данных
		train_data = pd.read_csv(args.train, header=None)		
		X_train = train_data.iloc[:, :-1]		
		y_train = train_data.iloc[:, -1]

		test_data = pd.read_csv(args.test, header=None)
		X_test = test_data.iloc[:, :-1]
		y_test = test_data.iloc[:, -1]
		
		scaler = StandardScaler()
		X_train_scalered = scaler.fit_transform(X_train)
		X_test_scalered = scaler.fit_transform(X_test)

		
		# Парсинг параметров
		params = {
			'C': args.C,
			'kernel': args.kernel,
			'degree': args.degree,
			'coef0': args.coef0,
			'gamma': args.gamma if args.gamma != 'float' else float(args.gamma),
			'max_iter': args.max_iter,
			'tol': args.tol,
			'class_weight': args.class_weight,
		}

		# Инициализация модели
		model = OneVsOneClassifier(SVC(**params), n_jobs=args.n_jobs)

		# Обучение модели
		code = "model.fit(X_train_scalered, y_train)"
		times = timeit.repeat(
			stmt=code,
			globals={
				"model": model,
				"X_train_scalered": X_train_scalered,
				"y_train": y_train
			},
			repeat=1,
			number=100
		)

		print(f"Набор данных: {args.train}, число потоков: {args.n_jobs}")
		print(f"Время обучения: {st.mean(times)/100:.6f} сек.")
		y_pred = model.predict(X_test_scalered)
		accuracy = accuracy_score(y_test, y_pred)
		balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
		print(f"Точность: {accuracy:.4f}")
		print(f"Взвешенная точность: {balanced_accuracy:.4f}")

	except Exception as e:
		print(f"\nОшибка: {str(e)}")

if __name__ == "__main__":
	main()