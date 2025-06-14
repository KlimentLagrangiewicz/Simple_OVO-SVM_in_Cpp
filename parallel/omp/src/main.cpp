#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>

#include "ovoSVM.hpp"
#include "vectors.hpp"


	// argv[1]  - количество используемых потоков
	// argv[2]  - название файла с обучающей выборкой
	// argv[3]  - название файла с тестовой выборкой
	// argv[4]  - название файла для печати результата
	// argv[5]  - параметр регуляризации
	// argv[6]  - порог для SVM (b)
	// argv[7]  - пороговая точность (acc)
	// argv[8]  - макимальное количество итераций
	// argv[9]  - тип ядра
	// argv[10]  - параметр 1 ядра (если линейное, то нет)
	// argv[11] - параметр 2 ядра (если линейное или RBF, то нет)

int main(int argc, char **argv) {
	try {
		if (argc < 4) {
			std::cout << "Not enough parameters!\n";
			std::exit(1);
		} else {
			
			std::size_t max_threads = std::stoul(argv[1]);
			if (max_threads == 0) max_threads = 1;
			omp_set_num_threads(max_threads);
			std::string trainfile(argv[2]), testfile(argv[3]);
			std::vector<std::vector<std::string>> strTrainData = readCSV(trainfile);
			if (!all_equal_size(strTrainData)) throw std::runtime_error("All vectors must have equal size\n");
			std::vector<std::vector<std::string>> strTestData = readCSV(testfile);
			if (!all_equal_size(strTestData)) throw std::runtime_error("All vectors must have equal size\n");
			trimStringMatrixSpaces(strTrainData);
			trimStringMatrixSpaces(strTestData);
			if (strTrainData[0].size() == strTestData[0].size()) { // Если размерности (число свойств + 1) данных совпадают, значит, вместе с тестовой выборкой поданы и их метки
				if (!validateAllButLastAsDouble(strTrainData[0])) { // Если была "шапка" у csv-файла, то удаляем первый вектор, содержащий её
					removeFirstElement(strTrainData);
				}
				std::vector<std::vector<double>> xTrain = convertToDoubleMatrix(strTrainData, strTrainData[0].size() - 1);
				autoscaling(xTrain); // Шкалируем входные данные
				std::vector<std::string> yTrain = getLastTable(strTrainData);
				const double c = (argc > 5) ? std::stod(argv[5]) : 1.0;
				const double b = (argc > 6) ? std::stod(argv[6]) : 0.0;
				const double acc = (argc > 7) ? std::stod(argv[7]) : 0.0001;
				const int maxIt = (argc > 8) ? std::stoi(argv[8]) : 10000;
				std::string kernelType = (argc > 9) ? std::string(argv[9]) : "liner";
				ovoSVM ovo_svm(xTrain, yTrain);
				ovo_svm.setParameters(kernelType, c, b, acc, maxIt);
				if (kernelType == "rbf") {
					const double gamma = (argc > 10) ? std::stod(argv[10]) : 1.0;
					ovo_svm.setGamma(gamma);
				} else if (kernelType == "poly") {
					const double degree = (argc > 10) ? std::stod(argv[10]) : 1.0;
					const double coef0 = (argc > 11) ? std::stod(argv[11]) : 0.0;
					ovo_svm.setPolyParam(degree, coef0);
				}
				auto start1 = std::chrono::steady_clock::now();
				
				// OpenMP;
				ovo_svm.omp_fit();
				
				auto end1 = std::chrono::steady_clock::now();
				auto duration1 = end1 - start1; // Время обучения модели
				auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration1).count();
				if (!validateAllButLastAsDouble(strTestData[0])) { // Если была "шапка" у csv-файла, то удаляем первый вектор, содержащий её
					removeFirstElement(strTestData);
				}
				std::vector<std::vector<double>> xTest = convertToDoubleMatrix(strTestData, strTestData[0].size() - 1);
				autoscaling(xTest); // Шкалируем входные данные
				std::vector<std::string> yTest = getLastTable(strTestData);
				auto start2 = std::chrono::steady_clock::now();
				std::vector<std::string> yPred = ovo_svm.predict(xTest);
				auto end2 = std::chrono::steady_clock::now();
				auto duration2 = end2 - start2; // Время предсказания результата
				auto ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2).count();
				const double accuracy = getAccuracy(yTest, yPred);
				const double balanced_accuracy = getBalancedAccuracy(yTest, yPred);
				std::cout << "Accuracy of classification: " << accuracy << std::endl;
				std::cout << "Balanced accuracy of classification: " << balanced_accuracy << std::endl;
				std::cout << "Execution time for training model: " << ms1 << " milliseconds\n";
				std::cout << "Execution time for get predictions: " << ms2 << " milliseconds\n";
				std::cout << "Total execution time: " << ms1 + ms2 << " milliseconds\n";
				//if (argc > 4) printResults(argv[4], yPred, accuracy);
				//else printResults(yPred, accuracy);
			} else {
				if (!validateAllButLastAsDouble(strTrainData[0])) { // Если была "шапка" у csv-файла, то удаляем первый вектор, содержащий её
					removeFirstElement(strTrainData);
				}
				std::vector<std::vector<double>> xTrain = convertToDoubleMatrix(strTrainData, strTrainData[0].size() - 1);
				autoscaling(xTrain); // Шкалируем входные данные
				std::vector<std::string> yTrain = getLastTable(strTrainData);
				const double c = (argc > 5) ? std::stod(argv[5]) : 1.0;
				const double b = (argc > 6) ? std::stod(argv[6]) : 0.0;
				const double acc = (argc > 7) ? std::stod(argv[7]) : 0.0001;
				const int maxIt = (argc > 8) ? std::stoi(argv[8]) : 10000;
				std::string kernelType = (argc > 9) ? std::string(argv[9]) : "liner";
				ovoSVM ovo_svm(xTrain, yTrain);
				ovo_svm.setParameters(kernelType, c, b, acc, maxIt);
				if (kernelType == "rbf") {
					const double gamma = (argc > 10) ? std::stod(argv[10]) : 1.0;
					ovo_svm.setGamma(gamma);
				} else if (kernelType == "poly") {
					const double degree = (argc > 10) ? std::stod(argv[10]) : 1.0;
					const double coef0 = (argc > 11) ? std::stod(argv[11]) : 0.0;
					ovo_svm.setPolyParam(degree, coef0);
				}
				auto start1 = std::chrono::steady_clock::now();
				
				// OpenMP;
				ovo_svm.omp_fit();
				
				auto end1 = std::chrono::steady_clock::now();
				auto duration1 = end1 - start1; // Время обучения модели
				auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration1).count();
				if (!validateAllButLastAsDouble(strTestData[0])) { // Если была "шапка" у csv-файла, то удаляем первый вектор, содержащий её
					removeFirstElement(strTestData);
				}
				std::vector<std::vector<double>> xTest = convertToDoubleMatrix(strTestData);
				autoscaling(xTest); // Шкалируем входные данные
				auto start2 = std::chrono::steady_clock::now();				
				std::vector<std::string> yPred = ovo_svm.predict(xTest);
				auto end2 = std::chrono::steady_clock::now();
				auto duration2 = end2 - start2; // Время предсказания результата
				auto ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2).count();
				std::cout << "Execution time for training model: " << ms1 << " milliseconds\n";
				std::cout << "Execution time for get predictions: " << ms2 << " milliseconds\n";
				std::cout << "Total execution time: " << ms1 + ms2 << " milliseconds\n";
				if (argc > 4) printResults(argv[4], yPred);
				else printResults(yPred);
			}
		}
	} catch (const std::exception& e) {
		std::cerr << "Error occurred: " << e.what() << std::endl;
	} catch (...) {
		std::cout << "Something went wrong\n"; 
	}
	return 0;
}
