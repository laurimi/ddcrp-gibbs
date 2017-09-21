#ifndef LIKELIHOODFCN_H
#define LIKELIHOODFCN_H
#include <eigen3/Eigen/Core>
#include <boost/functional/hash.hpp>
#include <memory>
#include <unordered_map>
#include <cstddef>
#include <vector>

// Abstract class for computing marginal log likelihoods of data sets
class LikelihoodFcn
{
public:
    LikelihoodFcn(const Eigen::MatrixXd& data);
    virtual ~LikelihoodFcn() {}
    double get_marginal_log_likelihood(const std::set<std::size_t>& members);

    Eigen::MatrixXd sample_uncentered_sum_of_squares_matrix(const std::set<std::size_t>& members) const;
    Eigen::VectorXd sample_mean(const std::set<std::size_t>& members) const;
    Eigen::VectorXd sum_data(const std::set<std::size_t>& members) const;
    int data_dimension() const;

private:
    struct SubsetHasher
    {
        std::size_t operator()(const std::set<std::size_t>& s) const
        {
            return boost::hash_range(s.begin(), s.end());
        }
    };

    typedef std::unordered_map< std::set<std::size_t>, double, SubsetHasher> LHMap;
    LHMap lh_;
    Eigen::MatrixXd data_;

    // concrete classes implement how to compute the likelihood of members
    virtual double compute_marginal_log_likelihood(const std::set<std::size_t>& members) const = 0;
};

#endif
