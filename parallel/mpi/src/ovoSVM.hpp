#include <vector>
#include <string>
#include <set>
#include <stdexcept>
#include <algorithm>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include "vectors.hpp"
#include "binSVM.hpp"


class ovoSVM {
	// Уровни меток
	std::set<std::string> labels;
	
	// Классификаторы
	std::vector<binSVM> binClassifiers;
	public:
		//Конструктор по умолчанию
		ovoSVM();
		
		//Конструктор
		ovoSVM(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY);
		
		//Конструктор по умолчанию
		~ovoSVM();
		
		void setC(const double inC);
		
		//
		void setKernelType(const std::string &kT);
		
		//
		void setB(const double inB);
		
		//
		void setAcc(const double inAcc);
		
		//
		void setMaxIt(const int _maxIt);
		
		//
		void setGamma(const double gammaIn = 0.0);
		
		//
		void setPolyParam(const double degreeIn = 0.0, const double coef0In = 0.0);
		
		//
		void setParameters(const std::string &kT = "liner", const double inC = 1.0, const double inB = 0.0, const double inAcc = 0.0001, const int inMaxIt = 10000);
		
		//
		void setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY);
		
		//
		void fit();
		
		//
		void mpi_fit();
		
		//
		std::string predict(const std::vector<double>& _x);
		
		//
		std::vector<std::string> predict(const std::vector<std::vector<double>>& _x);
		
		//
		ovoSVM& operator = (const ovoSVM& other);
};