#ifndef DONTCARELIKELIHOOD_H
#define DONTCARELIKELIHOOD_H
#include "LikelihoodFcn.h"
#include "NIWHyperParam.h"

class DontcareLikelihood : public LikelihoodFcn
{
public:
    DontcareLikelihood(const Eigen::MatrixXd &data,
            const Eigen::VectorXd &mu0,
            const Eigen::MatrixXd &S0,
            double k0,
            double v0);

private:
    virtual double compute_marginal_log_likelihood(const std::set<std::size_t>& members) const;
};

#endif
