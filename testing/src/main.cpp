#include <MatrixMethods>
#include <util.hpp>
#include <load_data.hpp>

int main() {
	auto [X_train, X_test, Y_train, Y_test] = load_mnist_digits_data(30000);

	Y_train = 2 * Y_train.array() - 1;
	Y_test = 2 * Y_test.array() - 1;

	double sig = 1;
	double lam = 1;
	int num_iters = 200;
	double lr = 0.01;

	auto kernel_fn = [sig](const VectorXd& u, const VectorXd& v) { 
		return exp(-(u - v).squaredNorm() / (2 * sig * sig)); 
	};

	std::vector<mm::KernelSVM> model;
	for (int i = 0; i < 10; i++) {
		model.push_back(mm::KernelSVM(kernel_fn));
	}

	MatrixXd K_train = mm::make_kernel(X_train, X_train, kernel_fn);
	for (int i = 0; i < 10; i++) {
		model[i].fit_with_kernel_matrix(X_train, K_train, Y_train.col(i), lam, num_iters, lr);
		std::cout << "training " << i + 1 << "/10\n";
	}

	MatrixXd preds(Y_test.rows(), Y_test.cols());
	MatrixXd K_test = mm::make_kernel(X_test, X_train, kernel_fn);
	for (int i = 0; i < 10; i++) {
		preds.col(i) = model[i].predict_with_kernel_matrix(K_test);
		std::cout << "testing " << i + 1 << "/10\n";
	}
	
	double error_rate = test_multi_preds(preds, Y_test);
	std::cout << "lam: " << lam << ", sig: " << sig << ", error rate: " << error_rate << '\n';
}

// [mnist-digits] lam: 1, sig: 1, error rate: 0.0428
// [mnist-fashion] lam: 1, sig: 1, error rate: 0.1664