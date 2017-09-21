#ifndef NIWHYPERPARAM_H
#define NIWHYPERPARAM_H
#include <eigen3/Eigen/Core>

// Normal-inverse-Wishart hyperparameters
struct NIWHyperParam
{
    NIWHyperParam(Eigen::VectorXd mu,
                   Eigen::MatrixXd S,
                   double k,
                   double v)
        : mu_(mu),
          S_(S),
          k_(k),
          v_(v)
    {}

    Eigen::VectorXd mu_;
    Eigen::MatrixXd S_;
    double k_;
    double v_;

};

#endif
