#ifndef load_data_hpp
#define load_data_hpp

#include <MatrixMethods>

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_credit_card_fraud_data() {
	std::ifstream file("../../../data/credit-card-fraud/creditcard_2023.csv");
	std::string line;
	std::getline(file, line); // skip header
	std::vector<std::vector<double>> data;
	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string cell;
		std::vector<double> row;
		while (std::getline(ss, cell, ',')) {
			row.push_back(std::stod(cell));
		}
		data.push_back(row);
	}
	file.close();
	if (data.empty()) throw std::runtime_error("empty CSV file");

	std::random_device rd;
  std::mt19937 gen(rd());
	std::shuffle(data.begin(), data.end(), gen);

	int m = data.size(), n = data[0].size() - 2;
	
	MatrixXd X = MatrixXd(m, n);
	VectorXd y(m);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			X(i, j) = data[i][j + 1];
		}
		y(i) = data[i].back();
	}

	X = mm::normalize(X);

	return mm::split_data(X, y, 0.5, 0.5);
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_digits_data() {
	MatrixXd X_train = mm::csv2matrix("../../../data/mnist-digits/X_train.csv");
	MatrixXd X_test = mm::csv2matrix("../../../data/mnist-digits/X_test.csv");
	MatrixXd Y_train = mm::csv2matrix("../../../data/mnist-digits/Y_train.csv");
	MatrixXd Y_test = mm::csv2matrix("../../../data/mnist-digits/Y_test.csv");
	
	return {X_train, X_test, Y_train, Y_test};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_digits_data(int max_rows) {
	MatrixXd X_train = mm::csv2matrix("../../../data/mnist-digits/X_train.csv", max_rows);
	MatrixXd X_test = mm::csv2matrix("../../../data/mnist-digits/X_test.csv", max_rows);
	MatrixXd Y_train = mm::csv2matrix("../../../data/mnist-digits/Y_train.csv", max_rows);
	MatrixXd Y_test = mm::csv2matrix("../../../data/mnist-digits/Y_test.csv", max_rows);
	
	return {X_train, X_test, Y_train, Y_test};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_fashion_data() {
	MatrixXd X_train = mm::csv2matrix("../../../data/mnist-fashion/X_train.csv");
	MatrixXd X_valid = mm::csv2matrix("../../../data/mnist-fashion/X_test.csv");
	MatrixXd Y_train = mm::csv2matrix("../../../data/mnist-fashion/y_train.csv");
	MatrixXd Y_valid = mm::csv2matrix("../../../data/mnist-fashion/y_test.csv");
	
	return {X_train, X_valid, Y_train, Y_valid};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_fashion_data(int max_rows) {
	MatrixXd X_train = mm::csv2matrix("../../../data/mnist-fashion/X_train.csv", max_rows);
	MatrixXd X_test = mm::csv2matrix("../../../data/mnist-fashion/X_test.csv", max_rows);
	MatrixXd Y_train = mm::csv2matrix("../../../data/mnist-fashion/Y_train.csv", max_rows);
	MatrixXd Y_test = mm::csv2matrix("../../../data/mnist-fashion/Y_test.csv", max_rows);
	
	return {X_train, X_test, Y_train, Y_test};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_face_emotion_data() {
	MatrixXd X = mm::csv2matrix("../../../data/face-emotion/X.csv");
	VectorXd y = mm::csv2vector("../../../data/face-emotion/y.csv");
	return mm::split_data(X, y, 0.75, 0.25);
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_easy_data() {
	return {
		mm::csv2matrix("../../../data/gd-data/X_train.csv"),
		mm::csv2matrix("../../../data/gd-data/X_test.csv"),
		mm::csv2vector("../../../data/gd-data/y_train.csv"),
		mm::csv2vector("../../../data/gd-data/y_test.csv"),
	};
}

#endif