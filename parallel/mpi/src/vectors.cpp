#include "vectors.hpp"


double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) [[unlikely]] throw std::invalid_argument("Vectors must have the same dimension");
	return std::sqrt(
		std::inner_product(
			a.begin(),
			a.end(),
			b.begin(),
			0.0,
			std::plus<double>(),
			[](const double x, const double y) { return (x - y) * (x - y); }
		)
	);
}

double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) [[unlikely]] throw std::invalid_argument("Vectors must have the same dimension");
	return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

bool all_equal_size(const std::vector<std::vector<double>>& vec) {
	if (vec.empty()) return true;
	const std::vector<double>::size_type expected = vec.front().size();
	return std::all_of(
		vec.begin() + 1,
		vec.end(),
		[expected](const std::vector<double>& v) { return v.size() == expected; }
	);
}

bool all_equal_size(const std::vector<std::vector<std::string>>& vec) {
	if (vec.empty()) return true;
	const std::vector<std::string>::size_type expected = vec.front().size();
	return std::all_of(
		vec.begin() + 1,
		vec.end(),
		[expected](const std::vector<std::string>& v) { return v.size() == expected; }
	);
}

// Liner Kernel
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points) {
	if (points.empty()) [[unlikely]] return {};
	if (!all_equal_size(points)) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const std::vector<std::vector<double>>::size_type m = points.size();
	std::vector<std::vector<double>> matrix(m, std::vector<double>(m)); 
	for (std::vector<double>::size_type i = 0; i < m; ++i) {
		matrix[i][i] = scalar_product(points[i], points[i]);
		for (std::vector<double>::size_type j = 0; j < i; ++j) {
			const double d = scalar_product(points[i], points[j]);
			matrix[i][j] = d;
			matrix[j][i] = d;
		}
	}
	return matrix;
}

// RBF Kernel
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double gamma) {
	if (points.empty()) [[unlikely]] return {};
	if (!all_equal_size(points)) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const std::vector<std::vector<double>>::size_type m = points.size();
	std::vector<std::vector<double>> matrix(m, std::vector<double>(m)); 
	for (std::vector<double>::size_type i = 0; i < m; ++i) {
		matrix[i][i] = 1.0;
		for (std::vector<double>::size_type j = 0; j < i; ++j) {
			const double ed = euclidean_distance(points[i], points[j]);
			const double d = std::exp(-gamma * ed * ed);
			matrix[i][j] = d;
			matrix[j][i] = d;
		}
	}
	return matrix;
}

// Poly Kernel
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double coef0, const double degree) {
	if (points.empty()) [[unlikely]] return {};
	if (!all_equal_size(points)) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const std::vector<std::vector<double>>::size_type m = points.size();
	std::vector<std::vector<double>> matrix(m, std::vector<double>(m)); 
	for (std::vector<std::vector<double>>::size_type i = 0; i < m; ++i) {
		for (std::vector<std::vector<double>>::size_type j = 0; j < i; ++j) {
			const double sp = scalar_product(points[i], points[j]);
			const double d = std::pow(sp + coef0, degree);
			matrix[i][j] = d;
			matrix[j][i] = d;
		}
		const double sp = scalar_product(points[i], points[i]);
		const double d = std::pow(sp + coef0, degree);
		matrix[i][i] = d;
	}
	return matrix;
}

size_t genInt(const size_t n, const size_t k) {
	std::mt19937 gen((std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()) % 1000000000).count());
	if (k == 0) {
		std::uniform_int_distribution<size_t> distrib(1, n - 1);
		return distrib(gen);
	}
	if (k == n - 1) {
		std::uniform_int_distribution<size_t> distrib(0, n - 2);
		return distrib(gen);
	}
	std::uniform_int_distribution<size_t> distrib(0, n - 1);
	const size_t r = distrib(gen);
	return r == k ? r + 1 : r;
}

double f(const std::vector<double>& d, const std::vector<int>& y, const std::vector<double>& a, const double b) {
	const size_t n = d.size();
	if (n != y.size() || n != a.size()) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const double* __restrict d_ptr = d.data();
	const int* __restrict y_ptr = y.data();
	const double* __restrict a_ptr = a.data();
	double res = 0.0;
	for (size_t i = 0; i < n; ++i) {
		res += y_ptr[i] * a_ptr[i] * d_ptr[i];
	}
	return res + b;
}

std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
	std::vector<std::vector<std::string>> result;
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
		return {};
	}
	std::string line;
	while (std::getline(file, line)) {
		std::vector<std::string> row;
		std::stringstream ss(line);
		std::string cell;
		while (std::getline(ss, cell, ',')) row.push_back(cell);
		result.push_back(row);
	}
	file.close();
	return result;
}

bool validateAllButLastAsDouble(const std::vector<std::string>& vec) {
	if (vec.empty()) return false;
	if (vec.size() == 1) return true;
	for (size_t i = 0; i < vec.size() - 1; ++i) {
		std::istringstream iss(vec[i]);
		double value;
		if (!(iss >> value) || (iss >> std::ws && !iss.eof())) {
			return false;
		}
	}
	return true;
}

double getAccuracy(const std::vector<std::string>& a, const std::vector<std::string>& b) {
	const auto n = a.size();
	if (n != b.size())[[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	if (n == 0) return 0.0;
	const auto count = std::inner_product(
		a.begin(), 
		a.end(),
		b.begin(),
		0ULL,
		std::plus<>(),
		[](const std::string& x, const std::string& y) { return x == y ? 1ULL : 0ULL; }
	);
	return static_cast<double>(count) / static_cast<double>(n);
}

double getAccuracy(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
	const auto n = a.size();
	if (n != b.size()) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	if (n == 0) return 0.0;
	const auto count = std::inner_product(
		a.begin(), 
		a.end(),
		b.begin(),
		0ULL,
		std::plus<>(),
		[](const std::size_t& x, const std::size_t& y) { return x == y ? 1ULL : 0ULL; }
	);
	return static_cast<double>(count) / static_cast<double>(n);
}

void trim_whitespace_inplace(std::string& s) {
	static constexpr const char* whitespace = " \t\n\r\f\v";
	if (s.empty()) return;
	const size_t first = s.find_first_not_of(whitespace);
	if (first == std::string::npos) {
		s.clear();
		return;
	}
	const size_t last = s.find_last_not_of(whitespace);
	const size_t len = last - first + 1;
	if (first > 0 || last < s.size() - 1) {
		s.assign(s, first, len);
	}
}

void trimStringVectorSpaces(std::vector<std::string>& vec) {
	std::for_each(vec.begin(), vec.end(), [](std::string& s) { trim_whitespace_inplace(s); });
}

void trimStringMatrixSpaces(std::vector<std::vector<std::string>> &matr) {
	std::for_each(matr.begin(), matr.end(), [] (std::vector<std::string>& vec) { trimStringVectorSpaces(vec); });
}

std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec) {
	std::vector<double> result;
	result.reserve(strVec.size());
	try {
		for (const auto& str : strVec) {
			std::size_t pos = 0;
			double value = std::stod(str, &pos);
			if (pos != str.size()) {
				throw std::invalid_argument("Invalid characters in element: '" + str + "'");
			}
			result.push_back(value);
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	return result;
}

std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec, const size_t pos) {
	if (pos == 0 || strVec.empty()) return {};
	std::vector<double> result;
	result.reserve(pos - 1);
	try {
		for (size_t i = 0; i < pos; ++i) {
			std::size_t p = 0;
			double value = std::stod(strVec[i], &p);
			if (p != strVec[i].size()) {
				throw std::invalid_argument("Invalid characters in element '" + strVec[i] + "'");
			}
			result.push_back(value);
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	return result;
}


std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec) {
	std::vector<std::vector<double>> res;
	res.reserve(strVec.size());
	std::transform(strVec.begin(), strVec.end(), std::back_inserter(res), [](const std::vector<std::string>& s) { return convertToDoubleVector(s); });
	return res;
}

std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec, const size_t pos) {
	std::vector<std::vector<double>> res;
	res.reserve(strVec.size());
	std::transform(strVec.begin(), strVec.end(), std::back_inserter(res), [pos](const std::vector<std::string>& s) { return convertToDoubleVector(s, pos); });
	return res;
}

void removeFirstElement(std::vector<std::string>& vec) {
	if (!vec.empty()) vec.erase(vec.begin());
}

void removeFirstElement(std::vector<std::vector<std::string>>& vec) {
	if (!vec.empty()) vec.erase(vec.begin());
}

std::vector<std::string> getLastTable(const std::vector<std::vector<std::string>>& matr) {
	std::vector<std::string> res;
	res.reserve(matr.size());
	std::ranges::transform(
		matr,
		std::back_inserter(res),
		[](const auto& vec) -> std::string { return vec.empty() ? std::string{} : vec.back(); }
	);
	return res;
}

void autoscaling(std::vector<std::vector<double>>& matr) {
	const size_t n = matr.size(), m = matr[0].size();
	std::vector<double> ex(m), exx(m);
	for (size_t i = 0; i < n; ++i) {
		const auto& row = matr[i];
		for (size_t j = 0; j < m; ++j) {
			ex[j] += row[j];
			exx[j] += row[j] * row[j];
		}
	}
	for (size_t j = 0; j < m; ++j) {
		ex[j] /= n;
		exx[j] /= n;
		double d = std::sqrt(exx[j] - ex[j] * ex[j]);
		if (d == 0.0) d = 1.0;
		exx[j] = 1.0 / d;
	}
	for (size_t i = 0; i < n; ++i) {
		auto& row = matr[i];
		for (size_t j = 0; j < m; ++j) {
			row[j] = (row[j] - ex[j]) * exx[j];
		}
	}
}

std::vector<int> convertToIntVec(const std::vector<std::string>& vec, std::string & first) {
	if (vec.empty()) return {};
	std::vector<int> res;
	res.reserve(vec.size());
	std::ranges::transform(
		vec,
		std::back_inserter(res),
		[first](const std::string& e) { return e == first ? 1 : -1; }
	);
	return res;
}

void to_lower(std::string& str) {
	std::transform(
		str.begin(),
		str.end(),
		str.begin(),
		[](unsigned char c) { return std::tolower(c); }
	);
}

void printResults(const std::vector<std::string>& vec) {
	if (vec.empty()) {
		std::cout << "Result std::vector<std::string> is empty\n";
		return;
	}
	std::cout << "Object: Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		std::cout << "Object[" << i << "]: " << vec[i] << '\n';
	}
	std::cout << '\n';
}

void printResults(const std::vector<std::string>& vec, const double acc) {
	if (vec.empty()) {
		std::cout << "Result std::vector<std::string> is empty\n";
		return;
	}
	std::cout << "Accuracy of SVM-classification = " << acc << '\n';
	std::cout << "Object: Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		std::cout << "Object[" << i << "]: " << vec[i] << '\n';
	}
	std::cout << '\n';
}

void printResults(const std::string &filename, const std::vector<std::string>& vec) {
	std::fstream file;
	file.open(filename, std::ios_base::out);
	if (!file.is_open()) {
		std::runtime_error("Can't open '" + filename + "' file for writing\n");
	}
	if (vec.empty()) {
		file << "Result std::vector<std::string> is empty\n";
		file.close();
		return;
	}
	file << "Object,Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		file << "Object[" << i << "]," << vec[i] << '\n';
	}
	file << '\n';
	file.close();
}

void printResults(const std::string &filename, const std::vector<std::string>& vec, const double acc) {
	std::fstream file;
	file.open(filename, std::ios_base::out);
	if (!file.is_open()) {
		std::runtime_error("Can't open '" + filename + "' file for writing\n");
	}
	if (vec.empty()) {
		file << "Result std::vector<std::string> is empty\n";
		file.close();
		return;
	}
	file << "Accuracy of SVM-classification," << acc << '\n';
	file << "Object,Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		file << "Object[" << i << "]," << vec[i] << '\n';
	}
	file << '\n';
	file.close();
}

std::string mostFrequentString(const std::vector<std::string>& input) {
	if (input.empty()) return "";
	std::unordered_map<std::string, int> frequency_map;
	for (const auto &s: input) {
		++frequency_map[s];
	}
	std::string result;
	int max_count = 0;
	for (const auto& pair: frequency_map) {
		const std::string& s = pair.first;
		int count = pair.second;
		if (count >= max_count) {
			max_count = count;
			result = s;
		}
	}
	return result;
}

std::string mostFrequentString(const std::vector<std::string>& input, const size_t n) {
	if (input.empty()) return "";
	std::unordered_map<std::string, int> frequency_map;
	frequency_map.reserve(n);
	for (const auto &s: input) {
		++frequency_map[s];
	}
	std::string result;
	int max_count = 0;
	for (const auto& pair: frequency_map) {
		const std::string& s = pair.first;
		int count = pair.second;
		if (count >= max_count) {
			max_count = count;
			result = s;
		}
	}
	return result;
}


double getBalancedAccuracy(const std::vector<std::string>& refenced, const std::vector<std::string>& obtained) {
	const auto n = refenced.size();
	if (obtained.size() != n) throw std::invalid_argument("All points must have the same dimension");
	if (n == 0) return 0.0;
	std::unordered_map<std::string, size_t> denominator, numerator; // numerator - числитель
	for (std::vector<std::string>::size_type i = 0; i < n; ++i) {
		++denominator[refenced[i]];
		if (refenced[i] == obtained[i]) ++numerator[refenced[i]];
	}
	double res = 0.0;
	for (const auto &pair: denominator) {
		if (auto search = numerator.find(pair.first); search != numerator.end()) {
			res += static_cast<double>(search->second) / static_cast<double>(pair.second);
		}
	}
	const auto s = denominator.size();
	return s == 0 ? 0.0 : res / static_cast<double>(s);
}

std::string getMostBalancedStr(const std::vector<std::pair<std::string, double>>& vec) {
	std::unordered_map<std::string, double> map;
	for (const auto &v: vec) {
		map[v.first] += v.second;
	}
	std::string max_key = map.begin()->first;
	double max_val = map.begin()->second;
	for (const auto &pair: map) {
		const double cur_val = pair.second;
		if (cur_val > max_val) {
			max_val = cur_val;
			max_key = pair.first;
		}
	}
	return max_key;
}

std::unordered_map<std::string, std::vector<size_t>> get_data_map(const std::vector<std::vector<double>> &x, const std::vector<std::string> y) {
	const auto n = y.size();
	if (n != x.size()) [[unlikely]] throw std::invalid_argument("X and Y vectors must have the same dimension");
	std::unordered_map<std::string, std::vector<size_t>> res;
	for (size_t i = 0; i < n; ++i) {
		if (auto search = res.find(y[i]); search != res.end()) {
			search->second.push_back(i);
		} else {
			res[y[i]].reserve(n);
			res[y[i]].push_back(i);
		}
	}
	for (auto &i: res) {
		i.second.shrink_to_fit();
	}
	return res;
}

size_t choose_random(const std::vector<size_t>& data) {
	if (data.empty()) [[unlikely]] throw std::invalid_argument("Can't select element from empty array");
	std::mt19937 gen((std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()) % 1000000000).count());
	std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
	return data[dist(gen)];
}
