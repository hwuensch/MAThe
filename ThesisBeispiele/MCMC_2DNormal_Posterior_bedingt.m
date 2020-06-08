function y = MCMC_2DNormal_Posterior_bedingt(block_j,x,kovmat_type,scaling,blocksize)
%%
% x = (x(1),...,x(j-1),x(j),x(j+1),...,x(J))
% mu_mvn = -"-
% kovmat_type = 0, 1, 2
% scaling = Zahl
% blocksize = k_j <= dimension

dimension = size(x,1);
idx_block_j = (blocksize*(block_j-1)+1):(blocksize*block_j);
idx_block_other = 1:dimension;
idx_block_other(idx_block_j) = [];

rho = 0.9;
mu_mvn = zeros(dimension,1); % gegebene Zielverteilung
sigma_mvn = MCMC_2DNormal_KovMat(kovmat_type,rho,scaling,dimension,blocksize); % gegebene Zielverteilung

mu_bedingt    = mu_mvn(idx_block_j) + sigma_mvn(idx_block_j,idx_block_other) * (sigma_mvn(idx_block_other,idx_block_other) \ (x(idx_block_other) - mu_mvn(idx_block_other)));
sigma_bedingt = sigma_mvn(idx_block_j,idx_block_j) - sigma_mvn(idx_block_j,idx_block_other) * (sigma_mvn(idx_block_other,idx_block_other) \ sigma_mvn(idx_block_other,idx_block_j));

y = mvnpdf(x(idx_block_j),mu_bedingt,sigma_bedingt);

% p1 = -0.5 * (x-mu_mvn)' * (sigma_mvn \ (x-mu_mvn));
% p2 = sqrt(det(sigma_mvn)*(2*pi)^dimension);
% 
% y = exp(p1) / p2;
end

%%
