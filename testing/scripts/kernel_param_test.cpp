#include <MatrixMethods>
#include <util.hpp>
#include <load_data.hpp>

int main() {
	auto [X_train, X_test, y_train, y_test] = load_face_emotion_data();
	

	std::vector<double> sigs;
	std::vector<double> lams;

	for (double i = 1e-3; i < 1e3; i *= 10) {
		for (double j = i; j < i * 10; j += i) {
			sigs.push_back(j);
			lams.push_back(j);
		}
	}

	double min_error_rate = 1;
	double best_sig = 0;
	double best_lam = 0;

	for (double lam : lams) {
		for (double sig : sigs) {
				mm::KernelRegressionL2 model1([sig](const VectorXd& u, const VectorXd& v) { 
				return exp(-(u - v).squaredNorm() / (2 * sig * sig)); 
			});
			model1.fit(X_train, y_train, lam);
			double error_rate = test_binary_preds(model1.predict(X_test), y_test);
			if (error_rate < min_error_rate) {
				min_error_rate = error_rate;
				best_sig = sig;
				best_lam = lam;
			}
		}
	}

	std::cout << "sig = " << best_sig << ", lam = " << best_lam << ", error rate: " << min_error_rate << '\n';
}