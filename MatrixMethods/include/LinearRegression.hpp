#ifndef linear_regression_hpp
#define linear_regression_hpp

#include <eigen3/Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace mm {

    struct LinearRegression {
        void fit(const MatrixXd& X, const VectorXd& y);
        void fit(const MatrixXd& X, const VectorXd& y, int num_iters, double lr);
        void fit(const MatrixXd& X, const VectorXd& y, double tol, double lr);
        VectorXd predict(const MatrixXd& X);

        VectorXd w;
    };

    struct LinearRegressionL1 {
        void fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr);
        VectorXd predict(const MatrixXd& X);

        VectorXd w;
    };

    struct LinearRegressionL2 {
        void fit(const MatrixXd& X, const VectorXd& y, double lam);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr);
        VectorXd predict(const MatrixXd& X);

        VectorXd w;
    };

}

#endif