#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/mpi/collectives.hpp>

#include "ovoSVM.hpp"
#include "vectors.hpp"


	// argv[1]  - название файла с обучающей выборкой
	// argv[2]  - название файла с тестовой выборкой
	// argv[3]  - название файла для печати результата
	// argv[4]  - параметр регуляризации
	// argv[5]  - порог для SVM (b)
	// argv[6]  - пороговая точность (acc)
	// argv[7]  - макимальное количество итераций
	// argv[8]  - тип ядра
	// argv[9]  - параметр 1 ядра (если линейное, то нет)
	// argv[10] - параметр 2 ядра (если линейное или RBF, то нет)

int main(int argc, char **argv) {
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
	try {
		if (argc < 3) {
			std::cout << "Not enough parameters!\n";
			std::exit(1);
		} else {		
			std::string trainfile(argv[1]), testfile(argv[2]);
			std::vector<std::vector<std::string>> strTrainData, strTestData;
			if (world.rank() == 0) {
				strTrainData = readCSV(trainfile);
				if (!all_equal_size(strTrainData)) throw std::runtime_error("All vectors must have equal size\n");
				strTestData = readCSV(testfile);
				if (!all_equal_size(strTestData)) throw std::runtime_error("All vectors must have equal size\n");
				trimStringMatrixSpaces(strTrainData);
				trimStringMatrixSpaces(strTestData);
			}			
			boost::mpi::broadcast(world, strTrainData, 0);
			boost::mpi::broadcast(world, strTestData, 0);
			if (strTrainData[0].size() == strTestData[0].size()) { // Если размерности (число свойств + 1) данных совпадают, значит, вместе с тестовой выборкой поданы и их метки
				if (!validateAllButLastAsDouble(strTrainData[0])) { // Если была "шапка" у csv-файла, то удаляем первый вектор, содержащий её
					removeFirstElement(strTrainData);
				}
				std::vector<std::vector<double>> xTrain = convertToDoubleMatrix(strTrainData, strTrainData[0].size() - 1);
				autoscaling(xTrain); // Шкалируем входные данные
				std::vector<std::string> yTrain = getLastTable(strTrainData);
				const double c = (argc > 4) ? std::stod(argv[4]) : 1.0;
				const double b = (argc > 5) ? std::stod(argv[5]) : 0.0;
				const double acc = (argc > 6) ? std::stod(argv[6]) : 0.0001;
				const int maxIt = (argc > 7) ? std::stoi(argv[7]) : 10000;
				std::string kernelType = (argc > 8) ? std::string(argv[8]) : "liner";				
				ovoSVM ovo_svm(xTrain, yTrain);
				ovo_svm.setParameters(kernelType, c, b, acc, maxIt);
				if (kernelType == "rbf") {
					const double gamma = (argc > 9) ? std::stod(argv[9]) : 1.0;
					ovo_svm.setGamma(gamma);
				} else if (kernelType == "poly") {
					const double degree = (argc > 9) ? std::stod(argv[9]) : 1.0;
					const double coef0 = (argc > 10) ? std::stod(argv[10]) : 0.0;
					ovo_svm.setPolyParam(degree, coef0);
				}
				auto start1 = std::chrono::steady_clock::now();
				ovo_svm.mpi_fit();
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
				if (world.rank() == 0) {
					std::cout << "Accuracy of classification: " << accuracy << std::endl;
					std::cout << "Balanced accuracy of classification: " << balanced_accuracy << std::endl;
					std::cout << "Execution time for training model: " << ms1 << " milliseconds\n";
					std::cout << "Execution time for get predictions: " << ms2 << " milliseconds\n";
					std::cout << "Total execution time: " << ms1 + ms2 << " milliseconds\n";
					if (argc > 3) printResults(argv[3], yPred, accuracy);
					else printResults(yPred, accuracy);
				}
			} else {
				if (!validateAllButLastAsDouble(strTrainData[0])) { // Если была "шапка" у csv-файла, то удаляем первый вектор, содержащий её
					removeFirstElement(strTrainData);
				}
				std::vector<std::vector<double>> xTrain = convertToDoubleMatrix(strTrainData, strTrainData[0].size() - 1);
				autoscaling(xTrain); // Шкалируем входные данные
				std::vector<std::string> yTrain = getLastTable(strTrainData);
				const double c = (argc > 4) ? std::stod(argv[4]) : 1.0;
				const double b = (argc > 5) ? std::stod(argv[5]) : 0.0;
				const double acc = (argc > 6) ? std::stod(argv[6]) : 0.0001;
				const int maxIt = (argc > 7) ? std::stoi(argv[7]) : 10000;
				std::string kernelType = (argc > 8) ? std::string(argv[8]) : "liner";
				ovoSVM ovo_svm(xTrain, yTrain);
				ovo_svm.setParameters(kernelType, c, b, acc, maxIt);
				if (kernelType == "rbf") {
					const double gamma = (argc > 9) ? std::stod(argv[9]) : 1.0;
					ovo_svm.setGamma(gamma);
				} else if (kernelType == "poly") {
					const double degree = (argc > 9) ? std::stod(argv[9]) : 1.0;
					const double coef0 = (argc > 10) ? std::stod(argv[10]) : 0.0;
					ovo_svm.setPolyParam(degree, coef0);
				}
				auto start1 = std::chrono::steady_clock::now();
				ovo_svm.mpi_fit();
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
				if (world.rank() == 0) {
					std::cout << "Execution time for training model: " << ms1 << " milliseconds\n";
					std::cout << "Execution time for get predictions: " << ms2 << " milliseconds\n";
					std::cout << "Total execution time: " << ms1 + ms2 << " milliseconds\n";
					if (argc > 3) printResults(argv[3], yPred);
					else printResults(yPred);
				}
			}
		}
	} catch (const std::exception& e) {
		std::cout << "Error occurred: " << e.what() << std::endl;
		boost::mpi::environment::abort(1);
	} catch (...) {
		std::cout << "Something went wrong\n";
		boost::mpi::environment::abort(1);
	}
	return 0;
}