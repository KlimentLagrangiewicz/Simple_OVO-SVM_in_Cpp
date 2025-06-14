#include <numeric>
#include <vector>
#include <math.h>
#include <stdexcept>
//#include <functional>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>
#include <random>
#include <ranges>
#include <chrono>
#include <limits>
#include <iterator>
#include <iostream>
#include <unordered_map>


double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);
double scalar_product(const std::vector<double>& a, const std::vector<double>& b);
bool all_equal_size(const std::vector<std::vector<double>>& vec);
bool all_equal_size(const std::vector<std::vector<std::string>>& vec);
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points);
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double gamma);
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double coef0, const double degree);
size_t genInt(const size_t n, const size_t k);
double f(const std::vector<double>& d, const std::vector<int>& y, const std::vector<double>& a, const double b);
std::vector<std::vector<std::string>> readCSV(const std::string& filename);
bool validateAllButLastAsDouble(const std::vector<std::string>& vec);
double getAccuracy(const std::vector<std::string>& a, const std::vector<std::string>& b);
double getAccuracy(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b);
void trim_whitespace_inplace(std::string& s);
void trimStringVectorSpaces(std::vector<std::string>& vec);
void trimStringMatrixSpaces(std::vector<std::vector<std::string>> &matr);
std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec);
std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec, const size_t pos);
std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec);
std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec, const size_t pos);
void removeFirstElement(std::vector<std::string>& vec);
void removeFirstElement(std::vector<std::vector<std::string>>& vec);
std::vector<std::string> getLastTable(const std::vector<std::vector<std::string>>& matr);
void autoscaling(std::vector<std::vector<double>>& matr);
std::vector<int> convertToIntVec(const std::vector<std::string>& vec, std::string& first);
void to_lower(std::string& str);
void printResults(const std::vector<std::string>& vec);
void printResults(const std::vector<std::string>& vec, const double acc);
void printResults(const std::string &filename, const std::vector<std::string>& vec);
void printResults(const std::string &filename, const std::vector<std::string>& vec, const double acc);
std::string mostFrequentString(const std::vector<std::string>& input);
std::string mostFrequentString(const std::vector<std::string>& input, const size_t n);
double getBalancedAccuracy(const std::vector<std::string>& refenced, const std::vector<std::string>& obtained);
std::string getMostBalancedStr(const std::vector<std::pair<std::string, double>>& vec);
std::unordered_map<std::string, std::vector<size_t>> get_data_map(const std::vector<std::vector<double>> &x, const std::vector<std::string> y);
size_t choose_random(const std::vector<size_t>& data);
