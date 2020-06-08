rng('default')
%% unterschiedliche Proposals vergleichen
iterAll         = 10;%[100 500 1000 5000 20000];
dimension       = 2;
proposal        = 0;%[-1 0 1]; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
sigma_prop      = 0.75; %0.01; %0.05; %0.07; %0.1;
sigma_post      = 1;% 1 - (0.9)^2;
startpunkt      = 2; % 0=nah am Ziel, 1 = weit weg, 2 = halb weit, sonst = x * ones(dimension,1)
blocksize       = 1;
for nIterAll=iterAll
    for prop=proposal
        [theta_curr,theta_can] = MCMC_2DNormalblockweise(nIterAll,dimension,prop,sigma_prop,sigma_post,startpunkt,blocksize);
        %     figure;
        %     [S,AX,BigAx,H,HAx] = plotmatrix(theta_can(1:2,:)');
    end
end

%% unterschiedliche Dimensionen miteinander vergleichen - startpunkt fern
% iterAll         = 5000;
% dimension       = [2:5 8 10];
% proposal        = 0; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
% sigma_prop      = 1;
% startpunkt      = 1; % 0=nah am Ziel, 1 = weit weg.
% 
% for dim=dimension
%     [theta_curr,theta_can] = MCMC_2DNormalblockweise(iterAll,dim,proposal,sigma_prop,sigma_post,startpunkt);
% end
% 
% %% unterschiedliche Dimensionen miteinander vergleichen - startpunkt nah
% iterAll         = 5000;
% dimension       = 10;
% proposal        = 0; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
% sigma_prop      = 1;
% startpunkt      = 0; % 0=nah am Ziel, 1 = weit weg.
% 
% for i=1:5
%     [theta_curr,theta_can] = MCMC_2DNormalblockweise(iterAll,i*dimension,proposal,sigma_prop,sigma_post,startpunkt);
% end
% 
%%