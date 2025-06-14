#include <vector>
#include <string>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <utility>


#include "vectors.hpp"

class binSVM {
	// Тип ядра
	std::string kernelType;
	
	// Матрица ядер
	std::vector<std::vector<double>> kernels;
	
	// Матрица с свойства обучающей выборки
	std::vector<std::vector<double>> x;
	
	// Метки обучающей выборки 
	std::vector<int> y;
	
	// Параметры опорных векторов
	std::vector<double> a;
	
	// номера опорных векторов
	std::vector<size_t> nums;
	
	// Параметр регуляризации
	double c;
	
	// Порог для SVM 
	double b;
	
	// Пороговая точность
	double acc;
	
	// Максимальное количество итераций
	int maxIt;
	
	// Уровни меток
	std::pair<std::string, std::string> labels;
	
	// Параметры полимиального ядра
	double degree, coef0;
	
	// Параметр RBF-ядра
	double gamma;
	
	// Проверка нарушения условий Каруша — Куна — Таккера
	bool check_kkt(const size_t check_idx);
	
	//
	inline bool check_kkt(const size_t check_idx, const double fxi);
	
	//
	std::pair<bool, double> get_violation(const size_t i, const double fxi);
	
	//
	inline double get_violation_value(const size_t i, const double fxi);
	
	//
	std::pair<std::vector<size_t>, std::vector<size_t>> get_candidates(const std::vector<double>& fx);
	
	//
	size_t get_feloner(const std::vector<double>& fx);
	
	//
	size_t get_feloner(const std::vector<double>& fx, const std::vector<size_t> &non_kkt);
	
	//
	size_t get_L1(const std::vector<double>& fx);
	
	//
	size_t get_L1(const std::vector<double>& fx, const std::vector<size_t>& non_kkt, const std::vector<size_t>& non_boundary_non_kkt);
	
	//
	void recalc_fx(std::vector<double> &vec);
	
	//
	void recalc_fx(std::vector<double> &vec, const size_t i, const size_t j, const double di, const double dj, const double db);
	
	//
	inline double get_error(const size_t i, const double fxi);
	
	//
	inline double get_delta(const size_t i, const size_t j, const std::vector<double> &fx);
	
	//
	size_t get_L2(const size_t i, const std::vector<double>& fx);
	
	//
	size_t get_L2(const size_t i, const std::vector<double>& fx, const std::vector<size_t> &vec);
	
	//
	std::pair<size_t, size_t> get_Ls(const std::vector<double>& fx);
	
	//
	void calcSMO(const size_t i, const size_t j, const std::vector<double>& fx);
	
	//
	double getKerlenPr(const std::vector<double>& _x, const size_t number);
	
	//
	double finalF(const std::vector<double>& _x);
		
	
	public:
		// конструктор
		binSVM(const std::string &kT = "liner", const double inC = 1.0, const double inB = 0.0, const double inAcc = 0.0001, const int inMaxIt = 10000);
		
		// метод для кэширования произведений ядер
		void kernelCaching();
		
		//
		void fit();

		//
		int predictInt(const std::vector<double>& _x);
		
		//
		std::string predict(const std::vector<double>& _x);
		
		std::pair<std::string, double> getPredictPair(const std::vector<double>& _x);
		
		//
		std::vector<int> predictInt(const std::vector<std::vector<double>>& _x);
		
		//
		std::vector<std::string> predict(const std::vector<std::vector<double>>& _x);
		
		//
		int getNumOfAttribytes();
		
		//
		void setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY);
		
		//
		void setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY, const std::string &first, const std::string &second);
		
		//
		void setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY, const std::string &f, const std::string &s, const std::vector<std::size_t>& index1, const std::vector<std::size_t>& index2);
		
		//
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
		
		binSVM& operator = (const binSVM& other);
};