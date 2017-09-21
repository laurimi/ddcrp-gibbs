#include "MultivariateNormal.h"
#include <eigen3/Eigen/Dense>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <iostream>

MultivariateNormal::MultivariateNormal(const Eigen::MatrixXd& data,
                                       const Eigen::VectorXd& mu0,
                                       const Eigen::MatrixXd& S0,
                                       double k0,
                                       double v0)
    : LikelihoodFcn(data),
      phyper_(mu0, S0, k0, v0)
{
}

// Ref. https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf page 21: Marginal likelihood
double MultivariateNormal::compute_marginal_log_likelihood(const std::set<std::size_t>& members) const
{
    NIWHyperParam hpost = get_posterior_hyperparameters(members);

    double marginal_ll = - ( static_cast<double>(members.size() * data_dimension()) / 2.0) * std::log( boost::math::constants::pi<double>() );
    marginal_ll += multivariate_log_gamma_ratio(hpost.v_/2.0, phyper_.v_/2.0, data_dimension() );
    marginal_ll += (phyper_.v_ / 2.0) * std::log( phyper_.S_.determinant() ) - (hpost.v_ / 2.0) * std::log( hpost.S_.determinant() );
    marginal_ll += (static_cast<double>( data_dimension() ) / 2.0) * ( std::log(phyper_.k_) - std::log(hpost.k_)  );

    return marginal_ll;
}

NIWHyperParam MultivariateNormal::get_posterior_hyperparameters(const std::set<std::size_t>& members) const
{
    // Murphy: Machine learning - a probabilistic perspective
    // Sect. 4.6.3.3
    const double num_data = static_cast<double>( members.size() );
    NIWHyperParam hpost(phyper_);
    hpost.mu_ = ( phyper_.k_ / ( phyper_.k_ + num_data ) ) * phyper_.mu_ + (num_data / ( phyper_.k_ + num_data ) ) * sample_mean(members);
    hpost.k_ += num_data;
    hpost.v_ += num_data;
    hpost.S_ += sample_uncentered_sum_of_squares_matrix(members) + phyper_.k_ * ( phyper_.mu_ * phyper_.mu_.transpose() ) - hpost.k_ * ( hpost.mu_ * hpost.mu_.transpose() );

    return hpost;
}

double MultivariateNormal::multivariate_log_gamma_ratio(double a, double b, std::size_t d)
{
    double y = 0.0;
    for ( std::size_t i = 1; i <= d; ++i)
    {
        const double d = (1.0 - static_cast<double>(i)) / 2.0;
        try
        {
            //y += std::log( boost::math::tgamma_ratio(a+d, b+d) );
            y += std::lgamma(a+d) - std::lgamma(b+d);
        }
        catch (...)
        {
            std::cout << "Failure with a+d = " << a+d << " and b+d = " << b + d << " (d = " << d << ")\n";
            throw;
        }
    }

    return y;
}
