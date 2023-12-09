#include <MatrixMethods>
#include <util.hpp>
#include <load_data.hpp>

int main() {
	auto [X_train, X_test, y_train, y_test] = load_face_emotion_data();
	

	double sig = 1.5;
	double lam = 0.5;

	mm::KernelRegressionL2 model1([sig](const VectorXd& u, const VectorXd& v) { 
		return exp(-(u - v).squaredNorm() / (2 * sig * sig)); 
	});

	model1.fit(X_train, y_train, lam);
	std::cout << "error rate: " << test_binary_preds(model1.predict(X_test), y_test) << '\n';

	model1.fit(X_train, y_train, lam, 10000, 1e-3);
	std::cout << "error rate: " << test_binary_preds(model1.predict(X_test), y_test) << '\n';

	mm::KernelSVM model2([sig](const VectorXd& u, const VectorXd& v) { 
		return exp(-(u - v).squaredNorm() / (2 * sig * sig)); 
	});

	model2.fit(X_train, y_train, lam, 10000, 1e-3);
	std::cout << "error rate: " << test_binary_preds(model2.predict(X_test), y_test) << '\n';

	model2.fit(X_train, y_train, lam, 1e-3, 1e-3);
	std::cout << "error rate: " << test_binary_preds(model2.predict(X_test), y_test) << '\n';
}