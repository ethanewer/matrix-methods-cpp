#include <MatrixMethods>
#include <util.hpp>
#include <load_data.hpp>

int main() {
	auto [X_train, X_test, Y_train, Y_test] = load_credit_card_fraud_data();

	double lam = 1;
	int num_iters = 1000;
	double lr = 0.01;


	auto start = std::chrono::high_resolution_clock::now();

	mm::LinearRegressionL2 model;
	model.fit(X_train, Y_train, lam, num_iters, lr);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "time: " << duration.count() << " seconds\n";

	MatrixXd preds = model.predict(X_test);
	std::cout << test_binary_preds(preds, Y_test);
}
