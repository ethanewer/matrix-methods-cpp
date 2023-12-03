#ifndef util_hpp
#define util_hpp

#include <common.hpp>
#include <ANN.hpp>
#include <CNN.hpp>
#include <DataLoader.hpp>

double test_binary_classifier(ANN& model, const MatrixXd& X, const VectorXd& y) {
	int m = X.rows();
	double error_count = 0;
	for (int i = 0; i < m; i++) {
		double pred = model.predict(X.row(i))(0) < 0.5 ? 0 : 1;
		if (pred != y(i)) {
			error_count++;
		}
	}
	return error_count / m;
}

double test_binary_classifier(CNN& model, MatrixXd X, const VectorXd& y) {
	int m = X.rows();
	double error_count = 0;
	for (int i = 0; i < m; i++) {
		Tensor3d t = Eigen::TensorMap<Tensor3d>(X.row(i).data(), 1, 28, 28);
		double pred = model.predict(t)(0) < 0.5 ? 0 : 1;
		if (pred != y(i)) {
			error_count++;
		}
	}
	return error_count / m;
}

double test_binary_classifier(ANN& model, MatrixDataLoader& data) {
	double error_count = 0;
	auto [X, Y] = data.get_batch();
	int batch_size = Y.rows();
	for (int i = 0; i < batch_size; i++) {
		double pred = model.predict(X.row(i))(0) < 0.5 ? 0 : 1;
		if (pred != Y(i)) {
			error_count++;
		}
	}
	return error_count / batch_size;
}

double test_binary_classifier(CNN& model, TensorDataLoader& data) {
	double error_count = 0;
	auto [X, Y] = data.get_batch();
	int batch_size = Y.rows();
	for (int i = 0; i < batch_size; i++) {
		double pred = model.predict(X.chip(i, 0))(0) < 0.5 ? 0 : 1;
		if (pred != Y(i)) {
			error_count++;
		}
	}
	return error_count / batch_size;
}

double test_multi_classifier(ANN& model, const MatrixXd& X, const MatrixXd& Y) {
	int m = X.rows(), n = Y.cols();
	double error_count = 0;
	for (int i = 0; i < m; i++) {
		VectorXd pred = model.predict(X.row(i));
		
		int max_pred_idx = 0, max_y_idx = 0;
		double max_pred_val = pred(0), max_y_val = Y(i, 0);
		for (int j = 0; j < n; j++) {
			if (pred(j) > max_pred_val) {
				max_pred_idx = j;
				max_pred_val = pred(j);
			}
			if (Y(i, j) > max_y_val) {
				max_y_idx = j;
				max_y_val = Y(i, j);
			}
		}

		if (max_pred_idx != max_y_idx) {
			error_count++;
		}
	}
	return error_count / m;
}

double test_multi_classifier(CNN& model, MatrixXd X, const MatrixXd& Y) {
	int m = X.rows(), n = Y.cols();
	double error_count = 0;
	for (int i = 0; i < m; i++) {
		VectorXd row = X.row(i);
		Tensor3d t = Eigen::TensorMap<Tensor3d>(row.data(), 1, 28, 28);
		VectorXd pred = model.predict(t);
		if (pred.hasNaN() || !pred.allFinite()) {
			std::cout << "BAD PRED: " << pred.transpose() << "\n\n";
		}
		int max_pred_idx = 0, max_y_idx = 0;
		double max_pred_val = pred(0), max_y_val = Y(i, 0);
		for (int j = 0; j < n; j++) {
			if (pred(j) > max_pred_val) {
				max_pred_idx = j;
				max_pred_val = pred(j);
			}
			if (Y(i, j) > max_y_val) {
				max_y_idx = j;
				max_y_val = Y(i, j);
			}
		}

		if (max_pred_idx != max_y_idx) {
			error_count++;
		}
	}
	return error_count / m;
}

double test_multi_classifier(ANN& model, MatrixDataLoader& data) {
	double error_count = 0;
	auto [X, Y] = data.get_batch();
	int batch_size = Y.rows();
	for (int i = 0; i < batch_size; i++) {
		VectorXd pred = model.predict(X.row(i));
		
		int max_pred_idx = 0, max_y_idx = 0;
		double max_pred_val = pred(0), max_y_val = Y(i, 0);
		for (int j = 0; j < pred.size(); j++) {
			if (pred(j) > max_pred_val) {
				max_pred_idx = j;
				max_pred_val = pred(j);
			}
			if (Y(i, j) > max_y_val) {
				max_y_idx = j;
				max_y_val = Y(i, j);
			}
		}

		if (max_pred_idx != max_y_idx) {
			error_count++;
		}
	}
	return error_count / batch_size;
}

double test_multi_classifier(CNN& model, TensorDataLoader& data) {
	double error_count = 0;
	auto [X, Y] = data.get_batch();
	int batch_size = Y.rows();
	for (int i = 0; i < batch_size; i++) {
		VectorXd pred = model.predict(X.chip(i, 0));
		
		int max_pred_idx = 0, max_y_idx = 0;
		double max_pred_val = pred(0), max_y_val = Y(i, 0);
		for (int j = 0; j < pred.size(); j++) {
			if (pred(j) > max_pred_val) {
				max_pred_idx = j;
				max_pred_val = pred(j);
			}
			if (Y(i, j) > max_y_val) {
				max_y_idx = j;
				max_y_val = Y(i, j);
			}
		}

		if (max_pred_idx != max_y_idx) {
			error_count++;
		}
	}
	return error_count / batch_size;
}

template<typename Fn, typename... Args>
auto time_fn(Fn&& fn, Args&&... args) {
	auto start = std::chrono::high_resolution_clock::now();
	auto result = std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	double seconds = duration.count();
	std::cout << "time: " << seconds << " seconds\n";
	return result;
}

double categorical_cross_entropy(ANN& model, const MatrixXd& X, const MatrixXd& Y) {
	int m = Y.rows(), n = Y.cols();
	double res = 0;
	for (int i = 0; i < m; i++) {
		VectorXd pred = model.predict(X.row(i));
		for (int j = 0; j < n; j++) {
			res -= Y(i, j) * log(pred(j));
		}
	}
	return res;
}

#endif