#include "ddCRP.h"
#include <boost/random/discrete_distribution.hpp>
#include <vector>
#include <iostream>

ddCRP::ddCRP(const Eigen::MatrixXd &log_decay_values, unsigned int seed)
    : c_(log_decay_values.rows()),
      log_decay_values_(log_decay_values),
      likelihood_(NULL),
      rng_( seed )
{
}

void ddCRP::setLikelihood(const std::shared_ptr<LikelihoodFcn> &l)
{
    likelihood_ = l;
}

void ddCRP::iterate()
{
    std::vector<double> p_link(c_.num_customers(), 0.0);
    for ( std::size_t source = 0; source < c_.num_customers(); ++source)
    {
        c_.unlink(source);
        get_link_likelihoods(source, p_link);
        std::transform(p_link.begin(), p_link.end(), p_link.begin(), [](double p){return std::exp(p); } );
        boost::random::discrete_distribution<std::size_t> d(p_link.begin(), p_link.end());
        c_.link(source, d(rng_));
    }
}

void ddCRP::get_link_likelihoods(std::size_t source, std::vector<double>& p) const
{
    p.resize(c_.num_customers());
    for (std::size_t target = 0; target < p.size(); ++target)
    {
        p[target] = get_link_likelihood(source, target);
    }
}

double ddCRP::get_link_likelihood(std::size_t source, std::size_t target) const
{
    double p = log_decay_values_(source, target);
    if ((source != target) && !std::isinf(p))
    {
        // If this is NOT a self link and the link is possible, we examine if tables are connected
        std::size_t k(0), l(0);
        if (c_.joins_tables(source, target, k, l))
        {
            std::set<std::size_t> table_l = c_.get_table_members(l);
            p -= likelihood_->get_marginal_log_likelihood(table_l);

            std::set<std::size_t> table_k = c_.get_table_members(k);
            p -= likelihood_->get_marginal_log_likelihood(table_k);

            // insert the smaller set to the larger one (less insert calls) and get joint table likelihood.
            if ( table_l.size() >= table_k.size() )
            {
                table_l.insert(table_k.begin(), table_k.end());
                p += likelihood_->get_marginal_log_likelihood(table_l);
            }
            else
            {
                table_k.insert(table_l.begin(), table_l.end());
                p += likelihood_->get_marginal_log_likelihood(table_k);
            }
        }
    }
    return p;
}

void ddCRP::print_tables(std::ostream& os) const
{
    c_.print_tables(os);
}

std::size_t ddCRP::get_table(std::size_t customer) const
{
    return c_.get_table(customer);
}

std::size_t ddCRP::num_tables() const
{
    return c_.num_tables();
}
