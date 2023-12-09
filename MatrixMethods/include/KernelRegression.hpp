#ifndef kernel_regression_hpp
#define kernel_regression_hpp

#include <eigen3/Eigen/Dense>
#include <functional>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace mm {

    struct KernelRegressionL2 {
        KernelRegressionL2(std::function<double(const VectorXd&, const VectorXd&)> kernel_fn);
        void fit(const MatrixXd& X, const VectorXd& y, double lam);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr);
        VectorXd predict(const MatrixXd& X_pred);

        std::function<double(const VectorXd&, const VectorXd&)> kernel_fn;
        MatrixXd X_train;
        VectorXd a;
    };

    struct KernelSVM {
        KernelSVM(std::function<double(const VectorXd&, const VectorXd&)> kernel_fn);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr);
        void fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr);
        VectorXd predict(const MatrixXd& X_pred);

        std::function<double(const VectorXd&, const VectorXd&)> kernel_fn;
        MatrixXd X_train;
        VectorXd a;
    };

    static MatrixXd make_kernel(const MatrixXd& X1, const MatrixXd& X2, std::function<double(const VectorXd&, const VectorXd&)> kernel_fn);

}

#endif