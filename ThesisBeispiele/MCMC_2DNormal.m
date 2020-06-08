function [Kette,theta_can,me_Kette_k] = MCMC_2DNormal(iterAll,dimension,sigma_wahl,sigma_proposal,sigma_posterior,startpunkt,swapscheme,swaptype,Kettennr)
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
    sigma_wahl = [-1 0 1]; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
end
if nargin<4
    sigma_proposal = 0.1;
end
if nargin<5
    startpunkt = 1;
end

mu = zeros(dimension,1);
blocksize = dimension;

me_Kette_k = zeros(1,iterAll+1); me_Kette_k(1) = Kettennr;

for sigma_wahl_i = sigma_wahl
    %%% setup
    Kette = zeros(dimension+1,iterAll+1); theta_can = zeros(dimension+1,iterAll);
    alpha = zeros(1,iterAll);
    Kette(1:end-1,1) = MCMC_2DNormal_Startpunkt(dimension,startpunkt);
    theta_curr       = Kette(1:end-1,1);
    posterior_curr   = MCMC_2DNormal_Posterior(theta_curr,mu,1,sigma_posterior,blocksize); Kette(end,1) = posterior_curr;
    [~,q_curr]       = MCMC_2DNormal_Proposal(theta_curr,sigma_wahl_i,sigma_proposal,blocksize);
    
    accept = 0;
    for iterJ = 1:iterAll
        if swaptype==1 % gucken, ob sPHS
            nextSwap = swapscheme(:,iterJ);
            Tauschkette = nextSwap==Kettennr;
        else
            Tauschkette = false;
        end
        if any(Tauschkette) % gucken, ob die aktuelle Kette tauschen muss
            % original sPHS
            Kette(:,iterJ+1) = Kette(:,iterJ); % "zwischenspeichern", obwohl diese Position nicht zu der aktuellen Kette gehört. Wird im Nachhinein umsortiert.
            Kettennr = nextSwap(~Tauschkette); % Bin jetzt neue Kette
            accept = accept + dimension;
        else
            %%% new candidate
            [theta_can(1:end-1,iterJ), q_can] = MCMC_2DNormal_Proposal(theta_curr,sigma_wahl_i,sigma_proposal,blocksize);
            %%% posterior of candidate
            posterior_can = MCMC_2DNormal_Posterior(theta_can(1:end-1,iterJ),mu,1,sigma_posterior,blocksize); theta_can(end,iterJ) = posterior_can;
            %%% accept-reject
            %     alpha = posterior_can / posterior_curr;
            alpha(1,iterJ) = posterior_can * q_can / posterior_curr / q_curr;
            if alpha(1,iterJ) > rand
                theta_curr      = theta_can(1:end-1,iterJ); Kette(1:end-1,iterJ+1)  = theta_curr;
                posterior_curr  = posterior_can; Kette(end,iterJ+1) = posterior_curr;
                q_curr          = q_can;
                accept          = accept + dimension;
            else
                Kette(:,iterJ+1) = Kette(:,iterJ);
            end
        end
        me_Kette_k(iterJ+1) = Kettennr;
    end
    
    Akzeptanzrate = accept/iterAll/dimension;
    % figure('Name','Akzeptanzlevel');
    % semilogy(alpha);
    % xlabel('Iteration'); ylabel('Akzeptanzlevel')
    %%
    MCMC_2DNormal_Plot(iterAll,sigma_wahl_i,dimension,Kette,theta_can,sigma_proposal,startpunkt,sigma_posterior,blocksize,accept);
end
end