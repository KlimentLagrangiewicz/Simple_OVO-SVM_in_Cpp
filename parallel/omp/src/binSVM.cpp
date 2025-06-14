#include "binSVM.hpp"


binSVM::binSVM(const std::string &kT, const double inC, const double inB, const double inAcc, const int inMaxIt) {
	kernelType = kT;
	c = inC;
	b = inB;
	acc = inAcc <= 0.0 ? 0.0001 : inAcc;
	maxIt = inMaxIt < 1 ? 10000 : inMaxIt;
	degree = 0.0;
	coef0 = 0.0;
	gamma = 0.0;
}

void binSVM::kernelCaching() {
	if (!kernels.empty()) return;
	if (x.empty()) return;
	if (kernelType.empty()) return;
	to_lower(kernelType);
	if (kernelType == "rbf") {
		if (gamma == 0.0) kernels = getKernelMatrix(x);
		else kernels = getKernelMatrix(x, gamma);
	} else if (kernelType == "poly") {
		if (degree == 0.0) kernels = getKernelMatrix(x);
		else kernels = getKernelMatrix(x, coef0, degree);
	} else {
		kernels = getKernelMatrix(x);
	}
}

bool binSVM::check_kkt(const size_t check_idx) {
	const double alpha_idx = a[check_idx];
	const double score_idx = f(kernels[check_idx], y, a, b);
	const double r_idx = static_cast<double>(y[check_idx]) * score_idx - 1.0;
	const bool cond1 = (alpha_idx < c) && (r_idx < -acc);
	const bool cond2 = (alpha_idx > 0.0) && (r_idx > acc);
	return !(cond1 || cond2);
}

bool binSVM::check_kkt(const size_t i, const double fxi) {
	bool flag = false;
	if (a[i] < acc) flag = y[i] * fxi >= 1.0 - acc;
	else if (a[i] > c - acc) flag = y[i] * fxi <= 1.0 + acc;
	else flag = std::abs(y[i] * fxi - 1.0) <= acc;
	return flag;
}

std::pair<bool, double> binSVM::get_violation(const size_t i, const double fxi) {
	bool flag = false;
	const double vi = std::abs(fxi * y[i] - 1.0);
	if (a[i] < acc) flag = y[i] * fxi >= 1.0 - acc;
	else if (a[i] > c - acc) flag = y[i] * fxi <= 1.0 + acc;
	else flag = std::abs(y[i] * fxi - 1.0) <= acc;
	return std::make_pair(flag, vi);
}

double binSVM::get_violation_value(const size_t i, const double fxi) {
	return std::abs(fxi * y[i] - 1.0);
}

//std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>> binSVM::get_candidates(const std::vector<double>& fx) {
std::pair<std::vector<size_t>, std::vector<size_t>> binSVM::get_candidates(const std::vector<double>& fx) {
	std::vector<size_t> non_kkt, non_boundary_non_kkt;
	non_kkt.reserve(y.size());
	non_boundary_non_kkt.reserve(y.size());
	for (size_t i = 0; i < y.size(); ++i) {
		if (!check_kkt(i, fx[i])) {
			non_kkt.push_back(i);
			if (0.0 < a[i] && a[i] < c) non_boundary_non_kkt.push_back(i);
		}
	}
	//non_boundary_non_kkt.shrink_to_fit();
	//non_kkt.shrink_to_fit();
	return {std::move(non_kkt), std::move(non_boundary_non_kkt)};
}


size_t binSVM::get_feloner(const std::vector<double>& fx) {
	size_t imax = 0;
	while (imax < y.size() && check_kkt(imax, fx[imax])) ++imax;
	if (imax == y.size()) return y.size();
	double max_v = get_violation(imax, fx[imax]).second;
	for (size_t i = imax + 1; i < y.size(); ++i) {
		const auto cur = get_violation(i, fx[i]);
		if (!cur.first && cur.second > max_v) {
			max_v = cur.second;
			imax = i;
		}
	}
	return imax;
}


size_t binSVM::get_feloner(const std::vector<double>& fx, const std::vector<size_t> &non_kkt) {
	size_t imax = non_kkt[0];
	double max_v = get_violation_value(imax, fx[imax]);
	for (const auto &i: non_kkt) {
		const double cur = get_violation_value(i, fx[i]);
		if (cur > max_v) {
			max_v = cur;
			imax = i;
		}
	}
	return imax;
}

size_t binSVM::get_L1(const std::vector<double>& fx) {
	const auto &[non_kkt, non_boundary_non_kkt] = get_candidates(fx);
	if (non_kkt.empty()) return y.size();
	if (non_boundary_non_kkt.empty()) return get_feloner(fx, non_kkt);
	return get_feloner(fx, non_boundary_non_kkt);
}

size_t binSVM::get_L1(const std::vector<double>& fx, const std::vector<size_t>& non_kkt, const std::vector<size_t>& non_boundary_non_kkt) {
	if (non_kkt.empty()) return y.size();
	if (non_boundary_non_kkt.empty()) return get_feloner(fx, non_kkt);
	return get_feloner(fx, non_boundary_non_kkt);
}

void binSVM::recalc_fx(std::vector<double> &vec) {
	for (size_t i = 0; i < vec.size(); ++i) {
		vec[i] = f(kernels[i], y, a, b);
	}
}

void binSVM::recalc_fx(std::vector<double> &vec, const size_t i, const size_t j, const double di, const double dj, const double db) {
	const std::vector<double> &K_i = kernels[i];
	const std::vector<double> &K_j = kernels[j];
	for (size_t l = 0; l < y.size(); ++l) {
		vec[l] += db + di * y[i] * K_i[l] + dj * y[j] * K_j[l];
	}
	/*
	vec[i] = f(kernels[i], y, a, b);
	vec[j] = f(kernels[j], y, a, b);
	const size_t k1 = std::min(i, j), k2 = std::max(i, j);
	for (size_t l = 0; l < k1; l++) {
		vec[l] += db + di * y[i] * kernels[i][l] + dj * y[j] * kernels[j][l];
	}
	for (size_t l = k1 + 1; l < k2; l++) {
		vec[l] += db + di * y[i] * kernels[i][l] + dj * y[j] * kernels[j][l];
	}
	for (size_t l = k2 + 1; l < y.size(); l++) {
		vec[l] += db + di * y[i] * kernels[i][l] + dj * y[j] * kernels[j][l];
	}
	*/
}

double binSVM::get_error(const size_t i, const double fxi) {
	return fxi - static_cast<double>(y[i]);
}

double binSVM::get_delta(const size_t i, const size_t j, const std::vector<double> &fx) {
	if (i == j) return 0.0;
	const double L = y[i] != y[j] ? std::max(0.0, a[j] - a[i]) : std::max(0.0, a[j] + a[i] - c);
	const double H = y[i] != y[j] ? std::min(c, c + a[j] - a[i]) : std::min(c, a[j] + a[i]);
	if (L == H) return 0.0;
	const double nu = 2.0 * kernels[i][j] - kernels[i][i] - kernels[j][j];
	if (nu >= 0.0) return 0.0;
	const double Ei = get_error(i, fx[i]), Ej = get_error(j, fx[j]);
	double a_j_new = a[j] - y[j] * (Ei - Ej) / nu;
	if (a_j_new > H) a_j_new = H;
	if (a_j_new < L) a_j_new = L;
	const double delta = std::abs(a_j_new - a[j]);
	return delta < acc ? 0.0 : delta;
}

size_t binSVM::get_L2(const size_t i, const std::vector<double>& fx) {
	const auto n = y.size();
	size_t jmax = 0;
	while (jmax < n && get_delta(i, jmax, fx) == 0.0) ++jmax;
	if (jmax == n) return n;
	double gr_max = get_delta(i, jmax, fx);
	for (size_t j = jmax + 1; j < n; ++j) {
		const double gr_j = get_delta(i, j, fx);
		if (gr_j > gr_max) {
			gr_max = gr_j;
			jmax = j;
		}
	}
	return jmax;
}

size_t binSVM::get_L2(const size_t i, const std::vector<double>& fx, const std::vector<size_t> &vec) {
	if (vec.empty()) return get_L2(i, fx);
	const auto n = vec.size();
	size_t jmax = 0;
	while (jmax < n && get_delta(i, vec[jmax], fx) == 0) ++jmax;
	if (jmax == n) return get_L2(i, fx);
	double gr_max = get_delta(i, vec[jmax], fx);
	for (size_t j = jmax + 1; j < n; ++j) {
		const double gr_j = get_delta(i, vec[j], fx);
		if (gr_j > gr_max) {
			gr_max = gr_j;
			jmax = j;
		}
	}
	return vec[jmax];
}

std::pair<size_t, size_t> binSVM::get_Ls(const std::vector<double>& fx) {
	const auto &&[non_kkt, non_boundary_non_kkt] = get_candidates(fx);
	const size_t i = get_L1(fx, non_kkt, non_boundary_non_kkt);
	const auto n = y.size();
	if (i == n) return std::make_pair(n, n);
	size_t j = get_L2(i, fx);//, non_boundary
	if (j == n) return std::make_pair(n, n);
	return std::make_pair(i, j);
}

void binSVM::calcSMO(const size_t i, const size_t j, const std::vector<double>& fx) {
	const double Ei = get_error(i, fx[i]), Ej = get_error(j, fx[j]);
	const double ai_old = a[i], aj_old = a[j];
	const double L = y[i] != y[j] ? std::max(0.0, a[j] - a[i]) : std::max(0.0, a[j] + a[i] - c);
	const double H = y[i] != y[j] ? std::min(c, c + a[j] - a[i]) : std::min(c, a[j] + a[i]);
	const double nu = 2.0 * kernels[i][j] - kernels[i][i] - kernels[j][j];
	a[j] -= y[j] * (Ei - Ej) / nu;
	if (a[j] > H) a[j] = H;
	if (a[j] < L) a[j] = L;
	const double delta_j = a[j] - aj_old;
	a[i] -= static_cast<double>(y[i]) * static_cast<double>(y[j]) * (delta_j);
	const double b1 = b - Ei - static_cast<double>(y[i]) * (a[i] - ai_old) * kernels[i][i] - static_cast<double>(y[j]) * delta_j * kernels[i][j];
	const double b2 = b - Ej - static_cast<double>(y[i]) * (a[i] - ai_old) * kernels[i][j] - static_cast<double>(y[j]) * delta_j * kernels[j][j];
	if (0.0 < a[i] && a[i] < c) b = b1;
	else if (0.0 < a[j] && a[j] < c) b = b2;
	else b = (b1 + b2) / 2.0;
}

void binSVM::fit() {
	kernelCaching();
	a.resize(x.size());
	std::fill(a.begin(), a.end(), 0.0);
	std::vector<double> fx(y.size(), 0.0);
	bool flag = true;
	for (int count = 1; count <= maxIt && flag; ++count) {
		const auto &[i, j] = get_Ls(fx);
		if (i == y.size()) flag = false;
		else {
			const double ai_old = a[i], aj_old = a[j], b_old = b;
			calcSMO(i, j, fx);
			recalc_fx(fx, i, j, a[i] - ai_old, a[j] - aj_old, b - b_old);
		}
	}
	nums.reserve(a.size());
	for (size_t i = 0; i < a.size(); ++i) {
		if (std::abs(a[i]) > acc)
			nums.push_back(i);
	}
	nums.shrink_to_fit();
	if (nums.empty()) {
		nums.resize(a.size());
		std::iota(nums.begin(), nums.end(), 0);
	}
}

double binSVM::getKerlenPr(const std::vector<double>& _x, const size_t number) {
	if (kernelType.empty()) return scalar_product(x[number], _x);
	to_lower(kernelType);
	if (kernelType == "rbf") {
		if (gamma == 0.0 ) return scalar_product(x[number], _x);
		else {
			const double d = euclidean_distance(x[number], _x);
			return std::exp(-gamma * d * d);
		}
	} else if (kernelType == "poly") {
		if (degree == 0.0) return scalar_product(x[number], _x);
		else {
			const double s = scalar_product(x[number], _x);
			return std::pow(s + coef0, degree);
		}
	} else {
		return scalar_product(x[number], _x);
	}
	return 0.0;
}

double binSVM::finalF(const std::vector<double>& _x) {
	double res = 0.0;
	for (const auto e: nums) {
		res += a[e] * y[e] * getKerlenPr(_x, e);
	}
	return res + b;
}

int binSVM::predictInt(const std::vector<double>& _x) {
	const double v = finalF(_x);
	if (v >= 0.0) return 1;
	else return -1;
}

std::string binSVM::predict(const std::vector<double>& _x) {
	const double v = finalF(_x);
	return (v >= 0.0) ? labels.first : labels.second;
}

std::pair<std::string, double> binSVM::getPredictPair(const std::vector<double>& _x) {
	const double v = finalF(_x);
	const std::string key = (v >= 0.0) ? labels.first : labels.second;
	const double w = 1.0 / (1.0 + std::exp(-std::abs(v)));
	return std::make_pair(key, w);
}

std::vector<int> binSVM::predictInt(const std::vector<std::vector<double>>& _x) {
	std::vector<int> res;
	res.reserve(_x.size());
	for (const auto &xi: _x) {
		res.push_back(predictInt(xi));
	}
	return res;
}

std::vector<std::string> binSVM::predict(const std::vector<std::vector<double>>& _x) {
	std::vector<std::string> res;
	res.reserve(_x.size());
	for (const auto& xi: _x) {
		res.push_back(predict(xi));
	}
	return res;
}

int binSVM::getNumOfAttribytes() {
	return x[0].size();
}

void binSVM::setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY) {
	x = dataX;
	if (!y.empty()) y.clear();
	std::set<std::string> str_set = std::set(dataY.begin(), dataY.end());
	const auto n = str_set.size();
	if (n != 2) throw std::invalid_argument("binary SVM works with only 2 classes");
	labels.first = *str_set.begin();
	labels.second = *str_set.rbegin();
	str_set.clear();
	y.reserve(dataY.size());
	for (const auto &i: dataY) {
		if (i == labels.first) y.push_back(1);
		else y.push_back(-1);
	}
}

void binSVM::setData(std::vector<std::vector<double>> &dataX, std::vector<std::string> &dataY, const std::string &first, const std::string &second) {
	if (dataX.size() != dataY.size()) throw std::invalid_argument("Points and their labels must have equal size");
	if (!x.empty()) x.resize(0);
	if (!y.empty()) y.resize(0);
	x.reserve(dataY.size());
	y.reserve(dataY.size());
	labels.first = first;
	labels.second = second;
	for (std::vector<std::string>::size_type i = 0; i < dataY.size(); ++i) {
		if (dataY[i] == first) {
			y.push_back(1);
			x.push_back(dataX[i]);
		} else if (dataY[i] == second) {
			y.push_back(-1);
			x.push_back(dataX[i]);
		}
	}
	x.shrink_to_fit();
	y.shrink_to_fit();
}

void binSVM::setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY, const std::string &f, const std::string &s, const std::vector<std::size_t>& index1, const std::vector<std::size_t>& index2) {
	if (dataX.size() != dataY.size()) throw std::invalid_argument("Points and their labels must have equal size");
	if (!x.empty()) x.resize(0);
	if (!y.empty()) y.resize(0);
	x.reserve(index1.size() + index2.size());
	y.reserve(index1.size() + index2.size());
	labels.first = f;
	labels.second = s;
	for (const auto &i: index1) {
		y.push_back(1);
		x.push_back(dataX[i]);
	}
	for (const auto &i: index2) {
		y.push_back(-1);
		x.push_back(dataX[i]);
	}
}

void binSVM::setC(const double inC) {
	c = inC;
}

void binSVM::setKernelType(const std::string &kT) {
	kernelType = kT;
}

void binSVM::setB(const double inB) {
	b = inB;
}

void binSVM::setAcc(const double inAcc) {
	acc = inAcc;
}

void binSVM::setMaxIt(const int _maxIt) {
	maxIt = _maxIt;
}

void binSVM::setGamma(const double gammaIn) {
	gamma = gammaIn;
}

void binSVM::setPolyParam(const double degreeIn, const double coef0In) {
	degree = degreeIn;
	coef0 = coef0In;
}

void binSVM::setParameters(const std::string &kT, const double inC, const double inB, const double inAcc, const int inMaxIt) {
	kernelType = kT;
	c = inC;
	b = inB;
	acc = inAcc <= 0.0 ? 0.0001 : inAcc;
	maxIt = inMaxIt < 1 ? 10000 : inMaxIt;	
	degree = 0.0;	
	coef0 = 0.0;	
	gamma = 0.0;
}

binSVM& binSVM::operator = (const binSVM& other) {
	if (!other.kernelType.empty()) kernelType = other.kernelType;
	if (!other.kernels.empty()) kernels = other.kernels;
	if (!other.x.empty()) x = other.x;
	if (!other.y.empty()) y = other.y;
	if (!other.a.empty()) a = other.a;
	if (!other.nums.empty()) nums = other.nums;
	c = other.c;
	b = other.b;
	acc = other.acc;
	maxIt = other.maxIt;
	labels = other.labels;
	degree = other.degree;
	coef0 = other.coef0;
	gamma = other.gamma;
	return *this;
}
