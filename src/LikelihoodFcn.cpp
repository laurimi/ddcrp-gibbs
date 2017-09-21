#include "LikelihoodFcn.h"

LikelihoodFcn::LikelihoodFcn(const Eigen::MatrixXd &data)
    : lh_(),
      data_(data)
{
}

double LikelihoodFcn::get_marginal_log_likelihood(const std::set<std::size_t> &members)
{
    auto it = lh_.find(members);
    if ( it != lh_.end() )
    {
        return it->second;
    }
    else
    {
        const double l = compute_marginal_log_likelihood(members);      
        lh_.insert( std::make_pair(members, l) );
        return l;
    }
}

Eigen::MatrixXd LikelihoodFcn::sample_uncentered_sum_of_squares_matrix(const std::set<std::size_t>& members) const
{
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero( data_dimension(), data_dimension() );
    for (const auto& i : members)
        S += ( data_.row(i).transpose() * data_.row(i) );

    return S;
}

Eigen::VectorXd LikelihoodFcn::sample_mean(const std::set<std::size_t>& members) const
{
    return (sum_data(members) / members.size());
}

Eigen::VectorXd LikelihoodFcn::sum_data(const std::set<std::size_t>& members) const
{
    Eigen::VectorXd sum = Eigen::VectorXd::Zero( data_dimension() );
    for (const auto& i : members)
        sum += data_.row(i);
    return sum;
}

int LikelihoodFcn::data_dimension() const
{
    return data_.cols();
}
