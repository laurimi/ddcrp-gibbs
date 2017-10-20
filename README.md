# A Gibbs sampler for the Distance Dependent Chinese Restaurant Process (ddCRP)
This is a C++ implementation of a Gibbs sampler for the Distance Dependent Chinese Restaurant Process (ddCRP), originally introduced in:
[Blei, D.M., Frazier, P.I.: "Distance Dependent Chinese Restaurant Process", Journal of Machine Learning Research 12 (2011):2383-2410](http://www.cs.columbia.edu/~blei/papers/BleiFrazier2011.pdf).

This implementation was used to obtain the results presented in: [Lauri, M., Frintrop, S.: "Object Proposal Generation Applying the Distance Dependent Chinese Restaurant Process", in Proc. 20th Scandinavian Conference on Image Analysis, Tromsö, Norway, June 12--14, 2017](https://doi.org/10.1007/978-3-319-59126-1_22).

For now, the code supports a multivariate normal cluster likelihood model.

## Building
You need a compiler with C++11 support.
This software also requires the Boost libraries and Eigen3.
```
mkdir build && cd build
cmake ..
make
```
The executable will be placed in the `bin` folder.

## Running
Executing `./bin/ddcrp_clustering_example` will yield the help message.

The feature (or data) file contains the `N` data points in `d`-dimensional space to feed into the ddCRP, as a `N`-by-`d` matrix.

The log decay file contains a `N`-by-`N` matrix. Informally, entry `(i,j)` quantifies the relative likelihood that the `i`th and `j`th data points will form a link (and thus be in the same cluster). More formally, entry `(i,j)` is equal to `log( f(d(i,j)) )`, where `d(i,j)` is a distance measure between the `i`th and `j`th data point, `f` is a decay function (see the papers). An entry `-Inf` here corresponds to an impossible link.

The prior covariance file sets the prior cluster covariance matrix, and is a `d`-by-`d` matrix.
The prior mean file sets the prior cluster mean vector, a `d`-by-`1` vector.
The strengths of these priors are determined by the input parameters `v` and `k`.

`n` specifies how many samples of clusterings to draw from the ddCRP, and `b` sets the number of burn-in samples before outputting the samples.

You can also draw samples from the ddCRP prior (ignoring the likelihood model) by setting the switch `--p`.

## Output
The output will be written to files called `clustering_0000.csv` with a running numbering.
The `i`th row in the file has a comma separated list of data point indices belonging to the `i`th cluster.
The number of rows in the file indicates the number of clusters.
For example, for 5 data points an output
```
0, 1, 2
3
4, 5
```
would mean that there are 3 clusters, with data points corresponding to the indices `{0,1,2}`, `{3}`, and `{4,5}`, respectively.

## Demo
There is some test data provided in the folder `data`. The file `data.csv` contains 100 samples drawn from two bivariate Gaussian distributions.

The file `log_decay.csv` contains a 100-by-100 matrix of log of the decay function values obtained as follows.
We compute the Euclidean distance `d(i,j)` between each pair of data points.
We apply a windowed exponential decay function
```
		exp(-d/a),		if d <= d_max
f(d) = 
		0				otherwise.
```
Here we have set `a=0.3`, `d_max=1.5`. 
The values on the off-diagonals of `log_decay.csv` are obtained by `log(f(d(i,j)))`.
For the diagonal entries we want the self-link likelihoods of each data point. Here we set a constant value `-0.8/a`.

We can run the clustering by
```
cd data
./../bin/ddcrp-gibbs-example -f data.csv -S covar.csv -m mean.csv -l log_decay.csv
```
The output will be written to the current folder.
The figure below compares the true clusters (left) and one of the clusterings drawn from the ddCRP (right).

![clustering example](/fig/example.png?raw=true)
