#include "ovoSVM.hpp"


ovoSVM::ovoSVM() {
	//binClassifiers = {};
}

ovoSVM::ovoSVM(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY) {
	const auto& data_map = get_data_map(dataX, dataY);
	const auto s = data_map.size();
	if (s < 2) [[unlikely]] throw std::invalid_argument("Unsupported number of classes");
	binClassifiers.resize(s * (s - 1) / 2);
	size_t k = 0;
	for (auto i = data_map.begin(); i != data_map.end(); i++) {
		for (auto j = std::next(i); j != data_map.end(); j++) {
			binClassifiers[k].setData(dataX, dataY, i->first, j->first, i->second, j->second);
			k++;
		}
	}
}

ovoSVM::~ovoSVM() {
	if (!binClassifiers.empty()) binClassifiers.clear();
}

void ovoSVM::setC(const double inC) {
	for (auto &a: binClassifiers) {
		a.setC(inC);
	}
}

void ovoSVM::setKernelType(const std::string &kT) {
	for (auto &a: binClassifiers) {
		a.setKernelType(kT);
	}
}

void ovoSVM::setB(const double inB) {
	for (auto &a: binClassifiers) {
		a.setB(inB);
	}
}

void ovoSVM::setAcc(const double inAcc) {
	for (auto &a: binClassifiers) {
		a.setAcc(inAcc);
	}
}

void ovoSVM::setMaxIt(const int _maxIt) {
	for (auto &a: binClassifiers) {
		a.setMaxIt(_maxIt);
	}
}

void ovoSVM::setGamma(const double gammaIn) {
	for (auto &a: binClassifiers) {
		a.setGamma(gammaIn);
	}
}

void ovoSVM::setPolyParam(const double degreeIn, const double coef0In) {
	for (auto &a: binClassifiers) {
		a.setPolyParam(degreeIn, coef0In);
	}
}

void ovoSVM::setParameters(const std::string &kT, const double inC, const double inB, const double inAcc, const int inMaxIt) {
	for (auto &a: binClassifiers) {
		a.setKernelType(kT);
		a.setC(inC);
		a.setB(inB);
		a.setAcc(inAcc);
		a.setMaxIt(inMaxIt);
	}
}

void ovoSVM::setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY) {
	const auto& data_map = get_data_map(dataX, dataY);
	const auto s = data_map.size();
	if (s < 2) [[unlikely]] throw std::invalid_argument("Unsupported number of classes");
	binClassifiers.resize(s * (s - 1) / 2);
	size_t k = 0;
	for (auto i = data_map.begin(); i != data_map.end(); i++) {
		for (auto j = std::next(i); j != data_map.end(); j++) {
			binClassifiers[k].setData(dataX, dataY, i->first, j->first, i->second, j->second);
			k++;
		}
	}
}

void ovoSVM::fit() {
	for (auto &a: binClassifiers) {
		a.fit();
	}
}

std::string ovoSVM::predict(const std::vector<double>& _x) {
	if (binClassifiers.empty()) return "";
	std::vector<std::string> vec;
	vec.reserve(binClassifiers.size());
	for (auto &a: binClassifiers) {
		vec.push_back(a.predict(_x));
	}
	return mostFrequentString(vec, std::sqrt(binClassifiers.size()));
}

std::string ovoSVM::balancedPredict(const std::vector<double>& _x) {
	if (binClassifiers.empty()) return "";
	std::vector<std::pair<std::string, double>> vec;
	vec.reserve(binClassifiers.size());
	for (auto &a: binClassifiers) {
		vec.push_back(a.getPredictPair(_x));
	}
	return getMostBalancedStr(vec);
}

std::vector<std::string> ovoSVM::predict(const std::vector<std::vector<double>>& _x) {
	std::vector<std::string> res;
	res.reserve(_x.size());
	for (const auto &xi: _x) {
		res.push_back(predict(xi));
	}
	return res;
}

std::vector<std::string> ovoSVM::getBalancedPredictions(const std::vector<std::vector<double>>& _x) {
	std::vector<std::string> res;
	res.reserve(_x.size());
	for (const auto &xi: _x) {
		res.push_back(balancedPredict(xi));
	}
	return res;
}

ovoSVM& ovoSVM::operator = (const ovoSVM& other) {
	if (!other.binClassifiers.empty()) binClassifiers = other.binClassifiers;
	return *this;
}
