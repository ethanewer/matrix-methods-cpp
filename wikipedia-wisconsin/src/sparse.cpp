#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <eigen3/Eigen/Sparse>

int main() {
	std::ifstream nodes_csv("../../data/wikipedia-wisconsin/wisconsin_nodes.csv");
	std::ifstream edges_csv("../../data/wikipedia-wisconsin/wisconsin_edges.csv");
	std::string line;
	
	std::vector<std::string> names;
	while (std::getline(nodes_csv, line)) {
		int l = line.find('\"'), r = line.rfind('\"');
		names.push_back(line.substr(l, r - l));
	}

	int n = names.size();
	Eigen::SparseMatrix<double> A(n, n);
	std::vector<Eigen::Triplet<double>> triplets;
	while (std::getline(edges_csv, line)) {
		std::istringstream ss(line);
		std::string s;
		std::getline(ss, s, ',');
		int j = stoi(s);
		std::getline(ss, s, ',');
		int i = stoi(s);
		triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
	}
	A.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::VectorXd modified_col_sums = Eigen::VectorXd(n);
	for (int j = 0; j < n; j++) {
		modified_col_sums(j) = A.col(j).sum() + n * 0.001;
		A.col(j) /= modified_col_sums(j);
	}

	auto start = std::chrono::high_resolution_clock::now();
	
	Eigen::VectorXd c = Eigen::VectorXd::Ones(n) / sqrt(n);
	for (int i = 0; i < 1000; i++) {
		double x = 0;
		for (int j = 0; j < n; j++) {
			x += c(j) / modified_col_sums(j);
		}

		c = A * c;
		c.array() += 0.001 * x;
		c /= c.norm();
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "Time: " << duration.count() << " seconds" << std::endl;

	std::vector<std::pair<int, double>> c_enumerated;
	for (int i = 0; i < n; i++) {
		c_enumerated.push_back({i, c(i)});
	}
	std::sort(c_enumerated.begin(), c_enumerated.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

	std::cout << '\n';
	for (int i = 0; i < 10; i++) {
		std::cout << c_enumerated[i].first << ' ' << names[c_enumerated[i].first] << ' ' << c_enumerated[i].second << '\n';
	}
}

// dense time: 187.112 seconds
// sparse time: 0.942936 seconds
// dense -O3 time: 6.029 seconds
// sparse -O3 time: 0.0670919 seconds
// python time: 3.93289