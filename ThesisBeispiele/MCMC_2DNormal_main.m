
%% unterschiedliche Proposals vergleichen
iterAll         = 100000 %[100 500 1000 5000 20000]
dimension       = 20;
proposal        = 0;%[-1 0 1]; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
sigma_prop      = 0.04; %0.4; %0.05; %0.07; %0.1;
sigma_post      = 1;
start           = 0; % 21,....,29 = 0.01,...,0.09; 99 = (-5,-10,...)
nChains         = 1
swaptype        = 0 % 0 = nicht, 1 = original sPHS, 2 = alle Ketten tauschen partnerweise, 3 = alle Ketten tauschen permutationsmaessig

laufe = 10;
ESS_all = -ones(2,laufe);
for lauf=1:laufe
    close all
    
    KettenOriginal  = cell(nChains,1);
    chain_curr = 1; % aktuelle Kette
    if swaptype == 1
        swapscheme = MCMC_2DNormal_Swapscheme_sPHS(iterAll,nChains);
    else
        swapscheme = [];%zeros(dimension,iterAll);
    end
    for startpunkt = start*ones(1,nChains)%-10:5:10%proposal=[1 2 5 10]%21:29
        for nIterAll = iterAll
            for prop=proposal
                [theta_curr,theta_can,me_Kette_k] = MCMC_2DNormal(nIterAll,dimension,prop,sigma_prop,sigma_post,startpunkt,swapscheme,swaptype,chain_curr);
                KettenOriginal{chain_curr} = theta_curr;
                me_Kette{chain_curr} = me_Kette_k; % Indexarray zu welcher Kette ein jeder Parametervektor gehört
                %     figure;
                %     [S,AX,BigAx,H,HAx] = plotmatrix(theta_can(1:2,:)');
                chain_curr = chain_curr+1;
            end
        end
    end
    %% Swap der Ketten abschließen und Ketten aneinander kleben
    if nChains > 1
        if swaptype==2
            % swap: alle Ketten tauschen partnerweise
            KetteGeklebt_Swap = MCMC_2DNormal_Swap(KettenOriginal,nChains,iterAll);
        elseif swaptype==1
            % swap original sPHS
            KetteGeklebt_Swap = MCMC_2DNormal_Swap_sPHS(me_Kette, KettenOriginal, nChains, iterAll, dimension);
        end
    end
    % Ketten aneinander kleben
    KetteGeklebt_Original = zeros(dimension+1,nChains*(iterAll+1));
    for chain_curr=1:nChains
        idx = ((chain_curr-1)*(iterAll+1)+1):(chain_curr*(iterAll+1));
        KetteGeklebt_Original(:,idx) = KettenOriginal{chain_curr};
    end
    
    %% ESS berechnen
    
    ESS = -ones(2,1); % 2, wegen: Originalkette und Swapkette
    Kette_orig = KetteGeklebt_Original(1:dimension,:);
    ESS(1,1) = MCMC_2DNormal_ESS(Kette_orig,iterAll,dimension);
    if swaptype > 0
        Kette_swap = KetteGeklebt_Swap(1:dimension,:);
        ESS(2,1) = MCMC_2DNormal_ESS(Kette_swap,iterAll,dimension);
    end
    mean_est = mean(KetteGeklebt_Original(1:dimension,:),2)
    var_est = var(KetteGeklebt_Original(1:dimension,:),0,2)
    ESS
    ESS_all(:,lauf) = ESS;
end

[min(ESS_all,[],2) median(ESS_all,2) max(ESS_all,[],2)]
%% plot der original/swapped Ketten
figure('Name','swap. Kette Parameter 1.');
plot((KetteGeklebt_Swap(1,:))')
xlabel('Iteration'); ylabel('x_1')
hold on
plot((KetteGeklebt_Original(1,:))','.')
hold off
legend('swap','original')


%% unterschiedliche Dimensionen miteinander vergleichen - startpunkt fern
% iterAll         = 5000;
% dimension       = [2:5 8 10];
% proposal        = 0; % 0 = standardnormal, 1=ausrichtung nach Posterior, -1=gegenausrichtung zur posterior
% sigma_prop      = 1;
% startpunkt      = 1; % 0=nah am Ziel, 1 = weit weg.
%
% for dim=dimension
%     [theta_curr,theta_can] = MCMC_2DNormal(iterAll,dim,proposal,sigma_prop,sigma_post,startpunkt);
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
%     [theta_curr,theta_can] = MCMC_2DNormal(iterAll,i*dimension,proposal,sigma_prop,sigma_post,startpunkt);
% end

%%