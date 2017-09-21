#ifndef MULTIVARIATENORMAL_H
#define MULTIVARIATENORMAL_H
#include "LikelihoodFcn.h"
#include "NIWHyperParam.h"

class MultivariateNormal : public LikelihoodFcn
{
public:
    MultivariateNormal(const Eigen::MatrixXd &data,
            const Eigen::VectorXd &mu0,
            const Eigen::MatrixXd &S0,
            double k0,
            double v0);

private:
    virtual double compute_marginal_log_likelihood(const std::set<std::size_t>& members) const;
    NIWHyperParam get_posterior_hyperparameters(const std::set<std::size_t>& members) const;
    static double multivariate_log_gamma_ratio(double a, double b, std::size_t d);

    NIWHyperParam phyper_;
};

#endif
