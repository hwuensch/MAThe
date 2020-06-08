function [candidate,candidate_pdf] = MCMC_2DNormal_Proposal_bedingt(block_j,theta_curr,proposal_type,sigma_proposal,blocksize)

dimension = size(theta_curr,1);
idx_block_j = (blocksize*(block_j-1)+1):(blocksize*block_j);
idx_block_other = 1:dimension;
idx_block_other(idx_block_j) = [];

rho = 0.9;

% Random-Walk von aktueller Position aus
mu_mvn     = zeros(dimension,1); % gegebene Zielverteilung
sigma_ziel = MCMC_2DNormal_KovMat(1,rho,1,dimension,blocksize); % gegebene Zielverteilung
mu_bedingt = mu_mvn(idx_block_j) + sigma_ziel(idx_block_j,idx_block_other) * (sigma_ziel(idx_block_other,idx_block_other) \ (theta_curr(idx_block_other) - mu_mvn(idx_block_other)));
% sigma_bedingt = sigma_mvn(idx_block_j,idx_block_j) - sigma_mvn(idx_block_j,idx_block_other) * (sigma_mvn(idx_block_other,idx_block_other) \ sigma_mvn(idx_block_other,idx_block_j));

sigma_mvn  = MCMC_2DNormal_KovMat(proposal_type,rho,sigma_proposal^2,dimension,blocksize); % KovMat für Proposal

candidate = theta_curr;
candidate(idx_block_j) = mvnrnd(mu_bedingt,sigma_mvn(idx_block_j,idx_block_j),1)';
candidate_pdf = mvnpdf(candidate(idx_block_j),mu_bedingt,sigma_mvn(idx_block_j,idx_block_j));
end