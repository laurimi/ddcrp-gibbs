#include "CustomerAssignment.h"
#include <boost/graph/connected_components.hpp>
#include <iostream>

CustomerAssignment::CustomerAssignment(std::size_t num_customers)
    : g_(num_customers),
      tables_(num_customers, 0),
      n_tables_(num_customers)
{
    for (auto vit = boost::vertices(g_); vit.first != vit.second; ++vit.first)
        boost::add_edge(*vit.first, *vit.first, g_);
    tables_ = std::vector<std::size_t>( boost::num_vertices(g_) );
    n_tables_ = boost::connected_components(g_, &tables_[0]);
}

void CustomerAssignment::print_tables(std::ostream& os) const
{
    for (std::size_t t = 0; t < n_tables_; ++t)
    {
        auto vit = vertices(g_);
        do
        {
            if (is_in_table(*vit.first, t))
            {
                os << *vit.first;
                ++vit.first;
                if ( vit.first != vit.second)
                    os << ", ";
            }
            else
            {
                ++vit.first;
            }
        }
        while ( vit.first != vit.second);

        os << std::endl;
    }
}

void CustomerAssignment::unlink(std::size_t source)
{
    boost::remove_edge_if([source,this](Edge e){ return boost::source(e, g_) == source; }, g_);

    const std::size_t c_table = tables_[source];
    ComponentGraph f(g_,
                     [source, c_table, this](Edge e){ return tables_[boost::source(e, g_)] == c_table; },
                     [c_table, this](Vertex v){ return tables_[v] == c_table; });

    std::vector<std::size_t> newtables( boost::num_vertices(g_), 0 );
    std::size_t n = boost::connected_components(f, &newtables[0] );
    if (n > 1)
    {
        // newtables is filled from 0 to how many vertices left in filtered graph,
        // while we must assign to table according to the original graphs' vertex indices.
        for (auto fit = vertices(f); fit.first != fit.second; ++fit.first)
        {
            if ( newtables[*fit.first] == 1 )
                tables_[*fit.first] = n_tables_;
        }
        ++n_tables_;
    }
}

void CustomerAssignment::link(std::size_t source, std::size_t target)
{
    boost::add_edge(source, target, g_);
    std::size_t k(0), l(0);
    if (joins_tables(source, target, k, l))
    {
        // assign to minimal table, deduct others to keep numbering consistent
        const std::size_t tnew = std::min(k, l);
        const std::size_t tmax = std::max(k, l);
        for (std::size_t i = 0; i < tables_.size(); ++i)
        {
            if ((tables_[i] == k) || (tables_[i] == l))
            {
                tables_[i] = tnew;
            }
            else if ( tables_[i] > tmax )
            {
                --tables_[i];
            }
        }
        --n_tables_;
    }
}

bool CustomerAssignment::joins_tables(std::size_t source,
                                      std::size_t target,
                                      std::size_t& k,
                                      std::size_t& l) const
{
    k = tables_[source];
    l = tables_[target];
    return (k != l);
}

bool CustomerAssignment::is_in_table(std::size_t customer, std::size_t table) const
{
    return (tables_[customer] == table);
}

std::size_t CustomerAssignment::num_customers() const
{
    return boost::num_vertices(g_);
}

std::set<std::size_t> CustomerAssignment::get_table_members(std::size_t table) const
{
    std::set<std::size_t> v;
    for (std::size_t c = 0; c < num_customers(); ++c)
    {
        if (is_in_table(c, table))
            v.insert(c);
    }
    return v;
}

std::size_t CustomerAssignment::num_tables() const
{
    return n_tables_;
}

std::size_t CustomerAssignment::get_table(std::size_t customer) const
{
    return tables_[customer];
}
