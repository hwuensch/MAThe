function y = MCMC_2DNormal_Posterior(x,mu_mvn,kovmat_type,scaling,blocksize)

dimension = length(x);
rho = 0.9;
sigma_mvn = MCMC_2DNormal_KovMat(kovmat_type,rho,scaling,dimension,blocksize);
% y = mvnpdf(theta_curr,mu_mvn,sigma_mvn);

p1 = -0.5 * (x-mu_mvn)' * (sigma_mvn \ (x-mu_mvn));
p2 = sqrt(det(sigma_mvn)*(2*pi)^dimension);

y = exp(p1) / p2;
end