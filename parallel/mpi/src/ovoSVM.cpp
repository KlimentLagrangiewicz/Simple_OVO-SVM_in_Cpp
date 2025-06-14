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

static void bcast_binSVM(binSVM &b, const size_t rank) {
	boost::mpi::communicator world;
	std::vector<size_t> nums;
	std::vector<double> a;
	double b1;
	if (rank == 0) {
		nums = b.getNums();
		a = b.getA();
		b1 = b.getB();
	}
	boost::mpi::broadcast(world, nums, 0);
	boost::mpi::broadcast(world, a, 0);
	boost::mpi::broadcast(world, b1, 0);
	if (rank != 0) {
		b.setNums(nums);
		b.setA(a);
		b.setB(b1);
	}
}


static void send_binSVM(binSVM &b, size_t sp, size_t root) {
	boost::mpi::communicator world;	
	const size_t rank = world.rank();
	if (rank == sp) {
		world.send(0, 0, b.getNums());
		world.send(0, 1, b.getA());
		world.send(0, 2, b.getB());
	} else if (rank == root) {
		std::vector<size_t> nums;
		std::vector<double> a;
		double b1;
		world.recv(sp, 0, nums);
		world.recv(sp, 1, a);
		world.recv(sp, 2, b1);
		b.setNums(nums);
		b.setA(a);
		b.setB(b1);
	}

}

static void send_block_binSVM(std::vector<binSVM> &vec, size_t sp, size_t rank, size_t id1, size_t id2) {
	boost::mpi::communicator world;	
	if (rank == sp) {
		std::vector<std::vector<size_t>> nums(id2 - id1);
		std::vector<std::vector<double>> vecA(id2 - id1);
		std::vector<double> vecB(id2 - id1);
		for (size_t i = id1; i < id2; i++) {
			nums[i - id1] = vec[i].getNums();
			vecA[i - id1] = vec[i].getA();
			vecB[i - id1] = vec[i].getB();
		}
		world.send(0, 1, nums);
		world.send(0, 2, vecA);
		world.send(0, 3, vecB);
	} else if (rank == 0) {
		std::vector<std::vector<size_t>> nums(id2 - id1);
		std::vector<std::vector<double>> vecA(id2 - id1);
		std::vector<double> vecB(id2 - id1);
		world.recv(sp, 1, nums);
		world.recv(sp, 2, vecA);
		world.recv(sp, 3, vecB);
		for (size_t i = id1; i < id2; i++) {
			vec[i].setNums(nums[i - id1]);
			vec[i].setA(vecA[i - id1]);
			vec[i].setB(vecB[i - id1]);
		}
	}

}

static void bcast_block_binSVM(std::vector<binSVM> &vec, size_t rank, size_t id1, size_t id2) {
	boost::mpi::communicator world;
	std::vector<std::vector<size_t>> nums;
	std::vector<std::vector<double>> a;
	std::vector<double> b1;
	if (rank == 0) {
		nums.resize(id2 - id1);
		a.resize(id2 - id1);
		b1.resize(id2 - id1);
		for (size_t i = id1; i < id2; i++) {
			nums[i] = vec[i].getNums();
			a[i] = vec[i].getA();
			b1[i] = vec[i].getB();
		}
	}
	boost::mpi::broadcast(world, nums, 0);
	boost::mpi::broadcast(world, a, 0);
	boost::mpi::broadcast(world, b1, 0);
	if (rank != 0) {
		for (size_t i = id1; i < id2; i++) {
			vec[i].setNums(nums[i]);
			vec[i].setA(a[i]);
			vec[i].setB(b1[i]);
		}
	}
}


void ovoSVM::mpi_fit() {
	boost::mpi::communicator world;
	const size_t rank = world.rank();
	const size_t size = world.size();
	const size_t perProc = binClassifiers.size() / size;
	if (size == 1 || perProc == 0) {
		// Обрабатываем все главным процессом, а после раскидываем всем процессам
		if (rank == 0) {
			for (auto &a: binClassifiers) {
				a.fit();
			}
		}
		bcast_block_binSVM(binClassifiers, rank, 0, binClassifiers.size());
	} else {
		for (size_t i = rank * perProc; i < rank * perProc + perProc; i++) {
			binClassifiers[i].fit();
		}
		for (size_t r = 1; r < size; r++) {
			send_block_binSVM(binClassifiers, r, rank, r * perProc, r * perProc + perProc);
		}
		if (rank == 0 && binClassifiers.size() % size){
			for (size_t i = perProc * size; i < binClassifiers.size(); i++) {
				binClassifiers[i].fit();
			}
		}
		bcast_block_binSVM(binClassifiers, rank, 0, binClassifiers.size());
	}
}

std::string ovoSVM::predict(const std::vector<double>& _x) {
	if (binClassifiers.empty()) return "";
	std::vector<std::string> vec;
	vec.reserve(binClassifiers.size());
	for (auto &a: binClassifiers) {
		vec.push_back(a.predict(_x));
	}
	return mostFrequentString(vec);
}

std::vector<std::string> ovoSVM::predict(const std::vector<std::vector<double>>& _x) {
	std::vector<std::string> res;
	res.reserve(_x.size());
	for (const auto &xi: _x) {
		res.push_back(predict(xi));
	}
	return res;
}

ovoSVM& ovoSVM::operator = (const ovoSVM& other) {
	if (!other.binClassifiers.empty()) binClassifiers = other.binClassifiers;
	if (!other.labels.empty()) labels = other.labels;
	return *this;
}
