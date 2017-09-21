#ifndef DDCRP_H
#define DDCRP_H
#include "CustomerAssignment.h"
#include "LikelihoodFcn.h"
#include <eigen3/Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <memory>
#include <iostream>

class ddCRP
{
public:
    ddCRP(const Eigen::MatrixXd& link_probabilities, unsigned int seed);
    void iterate();
    void setLikelihood(const std::shared_ptr<LikelihoodFcn>& l);
    void print_tables(std::ostream &os) const;

    std::size_t get_table(std::size_t customer) const;
    std::size_t num_tables() const;

private:
    void get_link_likelihoods(std::size_t source, std::vector<double>& p) const;
    double get_link_likelihood(std::size_t source, std::size_t target) const;

    CustomerAssignment c_;
    Eigen::MatrixXd log_decay_values_;

    std::shared_ptr<LikelihoodFcn> likelihood_;

    boost::random::mt19937 rng_;
};

#endif
