#include <cmath>
#include <random>
#include <algorithm>
#include <list>
//#include <armadillo>
#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp17)]]

using namespace arma;
using namespace Rcpp;

class return_eigs_list {
    size_t step, n_final;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 generator{ seed };
    
    std::normal_distribution<double> normal_dist{ 0, 1 };
    std::uniform_real_distribution<double> unif_dist{ 0, 1 };
    std::cauchy_distribution<double> cauchy_dist{ 0, 1 };
    std::exponential_distribution<double> exp_dist{ 1 };
    
public:
    //Parameter Initialization
    return_eigs_list(size_t step_set, size_t n_final_set) {
        step = step_set, n_final = n_final_set;
    };
    
    
    std::vector<mat> solve(std::string distribution) {
        std::vector<mat> results;
        
		for (size_t i = 1; i <= n_final; ++i) {
			size_t n = i * step;
			mat random_matrix(n, n);
			
			if (distribution == "normal") {
				std::generate(random_matrix.begin(), random_matrix.end(), [=]() { return normal_dist(generator); } );
			  random_matrix -= 0; //Mean Centered
			  random_matrix /= sqrt(n); //Variance Standardized
			}
			
			else if (distribution == "uniform") {
				std::generate(random_matrix.begin(), random_matrix.end(), [=]() { return unif_dist(generator); } );
				random_matrix -= 1/2; //Mean Centered
				random_matrix /= sqrt(n / 12); //Variance Standardized
			}
			
			else if (distribution == "exponential") {
				std::generate(random_matrix.begin(), random_matrix.end(), [=]() { return exp_dist(generator); } );
				random_matrix -= 1; //Mean Centered
				random_matrix /= sqrt(n); //Variance Standardized
			}
			
			else if (distribution == "cauchy") {
				std::generate(random_matrix.begin(), random_matrix.end(), [=]() { return cauchy_dist(generator); } );
				random_matrix -= mean(vectorise(random_matrix)); //Mean Centered
				random_matrix /= stddev(vectorise(random_matrix)) * sqrt(n); //Variance Standardized
			}
			
			cx_vec eigval = eig_gen(random_matrix);
			vec vec_n(n); vec_n.fill(n);
			
			mat results_eig(n, 3);
			results_eig.col(0) = real(eigval);
			results_eig.col(1) = imag(eigval);
			results_eig.col(2) = vec_n;
			results.push_back(results_eig);
		}
        return results;
    }
};

//[[Rcpp::export]]
Rcpp::List random_matrix_solve(size_t step, size_t n_final, std::string distribution) {
    
    return_eigs_list random_eigs(step, n_final);
    auto results = random_eigs.solve(distribution);
    
    std::vector<Rcpp::DataFrame> R_results;
    for (const auto& data_frame : results) {
        Rcpp::DataFrame temp = Rcpp::DataFrame::create(Named("Real") = as<Rcpp::NumericVector>(wrap(data_frame.col(0))),
                                                     Named("Imaginary") = as<Rcpp::NumericVector>(wrap(data_frame.col(1))),
                                                     Named("n") = as<Rcpp::NumericVector>(wrap(data_frame.col(2))));
        R_results.push_back(temp);
    }
    return as<Rcpp::List>(wrap(R_results));
}