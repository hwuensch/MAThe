function [theta_curr,theta_can] = MCMC_2DNormalblockweise(iterAll,dimension,proposal_type,sigma_proposal,sigma_post,startpunkt,blocksize)
%MCMC_2DNORMAL Summary of this function goes here
%   Detailed explanation goes here
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
if nargin<1
    iterAll = 5000;
end
if nargin<2
    dimension = 2;
end
if nargin<3
    proposal_type = [-1 0 1]; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
end
if nargin<4
    sigma_proposal = 0.1;
end
if nargin<5
    sigma_post = 1;
end
if nargin<6
    startpunkt = 1; % 0=nah am Ziel, 1 = weit weg.
end
if nargin<7
    blocksize = dimension;
end
block = [1:blocksize:dimension dimension+1];

for proposal_type_i = proposal_type
%%% setup
theta_curr = zeros(dimension,iterAll+1); theta_can = zeros(dimension,iterAll);
theta_curr(:,1) = MCMC_2DNormal_Startpunkt(dimension,startpunkt);
posterior_curr = zeros(1,(length(block)-1)); q_curr = zeros(1,(length(block)-1));
for b = 1:(length(block)-1)
    posterior_curr(1,b)  = MCMC_2DNormal_Posterior_bedingt(b,theta_curr(:,1),1,sigma_post,blocksize);
    [~,q_curr(1,b)]      = MCMC_2DNormal_Proposal_bedingt(b,theta_curr(:,1),proposal_type_i,sigma_proposal,blocksize);
end
accept = 0;

for iterJ = 1:iterAll
    for b = 1:(length(block)-1)
        %%% new candidate
        [theta_ca, q_can] = ...
            MCMC_2DNormal_Proposal_bedingt(b,theta_curr(:,iterJ),proposal_type_i,sigma_proposal,blocksize);
        theta_can(block(b):(block(b+1)-1),iterJ) = theta_ca(block(b):(block(b+1)-1),1);
        %%% posterior of candidate
        posterior_can = MCMC_2DNormal_Posterior_bedingt(b,theta_can(:,iterJ),1,sigma_post,blocksize);
        %%% accept-reject
        %     alpha = posterior_can / posterior_curr;
        alpha = posterior_can * q_can / posterior_curr(1,b) / q_curr(1,b);
        if alpha > rand
            theta_curr(block(b):(block(b+1)-1),iterJ+1) = theta_can(block(b):(block(b+1)-1),iterJ);
            posterior_curr(1,b)                         = posterior_can;
            q_curr(1,b)                                 = q_can;
            accept = accept + blocksize;
        else
            theta_curr(block(b):(block(b+1)-1),iterJ+1) = theta_curr(block(b):(block(b+1)-1),iterJ);
        end
    end
end

%%
MCMC_2DNormal_Plot(iterAll,proposal_type_i,dimension,theta_curr,theta_can,sigma_proposal,startpunkt,sigma_post,blocksize,accept);
end
end
