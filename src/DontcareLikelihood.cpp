#include "DontcareLikelihood.h"

DontcareLikelihood::DontcareLikelihood(
        const Eigen::MatrixXd &data,
        const Eigen::VectorXd &mu0,
        const Eigen::MatrixXd &S0,
        double k0,
        double v0)
    : LikelihoodFcn(data)
{

}

double DontcareLikelihood::compute_marginal_log_likelihood(const std::set<std::size_t>& members) const
{
    return 0.0;
}
