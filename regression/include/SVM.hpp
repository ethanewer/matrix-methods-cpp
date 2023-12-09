#ifndef svm_hpp
#define svm_hpp

#include <eigen3/Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct SVM {
    void fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr);
    void fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr);
    VectorXd predict(const MatrixXd& X);

    VectorXd w;
};

#endif