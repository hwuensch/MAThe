function [candidate,candidate_pdf] = MCMC_2DNormal_Proposal(theta_curr,proposal_type,sigma_proposal,blocksize)

dimension = length(theta_curr);
rho = 0.9;

% Random-Walk von aktueller Position aus
sigma_mvn = MCMC_2DNormal_KovMat(proposal_type,rho,sigma_proposal^2,dimension,blocksize);

candidate = mvnrnd(theta_curr,sigma_mvn,1)';
candidate_pdf = mvnpdf(candidate,theta_curr,sigma_mvn);
end