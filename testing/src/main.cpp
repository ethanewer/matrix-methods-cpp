#include <MatrixMethods>
#include <util.hpp>

int main() {
	double lr = 0.5;
	
	MatrixXd X = mm::csv2matrix_with_ones("../../data/face-emotion/X.csv");
	VectorXd y = mm::csv2vector("../../data/face-emotion/y.csv");
	y = y.array().max(0);

	int m = X.rows(), n = X.cols();
	
	mm::ANN model(
		{
			new mm::Dense(10, 32),
			new mm::Sigmoid(),
			new mm::Dense(32, 1),
		},
		new mm::SigmoidBinaryCrossentropy()
	);
	
	double error_sum = 0;
	int num_groups = 8;
	int test_size = m / num_groups;

	for (int group = 0; group < num_groups; group++) {
		int test_begin = group * test_size;
		int test_end = test_begin + test_size;
		MatrixXd X_test = X.block(test_begin, 0, test_size, n);
		VectorXd y_test = y.segment(test_begin, test_size);

		for (int epoch = 0; epoch < 200; epoch++) {
			for (int i = 0; i < X.rows(); i++) {
				if (test_begin <= i && i < test_end) {
					continue;
				}
				model.predict(X.row(i));
				model.update(y.row(i), lr);
			}
		}

		error_sum += test_binary_classifier(model, X_test, y_test);
	}
	
	std::cout << "Error rate: " << error_sum / num_groups << '\n';
}