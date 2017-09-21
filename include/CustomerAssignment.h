#ifndef CUSTOMERASSIGNMENT_H
#define CUSTOMERASSIGNMENT_H
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/function.hpp>

class CustomerAssignment
{
public:
    CustomerAssignment(std::size_t num_customers);
    void print_tables(std::ostream &os) const;
    void unlink(std::size_t source);
    void link(std::size_t source, std::size_t target);
    bool joins_tables(std::size_t source,
                      std::size_t target,
                      std::size_t& k,
                      std::size_t& l) const;

    std::size_t num_customers() const;
    std::size_t num_tables() const;
    std::size_t get_table(std::size_t customer) const;
    std::set<std::size_t> get_table_members(std::size_t table) const;


private:
    typedef boost::adjacency_list<
    boost::listS,
    boost::vecS,
    boost::undirectedS
    > UndirectedGraph;

    typedef typename boost::graph_traits<UndirectedGraph>::vertex_descriptor Vertex;
    typedef typename boost::graph_traits<UndirectedGraph>::edge_descriptor Edge;
    typedef boost::filtered_graph<UndirectedGraph, boost::function<bool(Edge)>, boost::function<bool(Vertex)> > ComponentGraph;

    bool is_in_table(std::size_t customer, std::size_t table) const;


    UndirectedGraph g_; // links from customer to customer
    std::vector<std::size_t> tables_; // which table each customer sits at
    std::size_t n_tables_; // how many tables are there
};

#endif
