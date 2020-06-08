function KetteGeklebt = MCMC_2DNormal_Swap_sPHS(me_Kette, Kette, nChains, iterAll, dimension)
    % sortiert die Ketten im Nachhinein entsprechend me_Kette und klebt die mehreren
    % Ketten aneinander
    Idx_Iter = 1:(iterAll+1);
    
    KetteGeklebt = NaN(dimension+1,(iterAll+1)*nChains);
    for kk = 1:nChains
        for k = 1:nChains
            Idx_Iter_Kette = Idx_Iter(me_Kette{k} == kk);
            KetteGeklebt(:,(kk-1)*(iterAll+1) + Idx_Iter_Kette) = Kette{k}(:,Idx_Iter_Kette);
        end
    end
end

%% Test
% Kette = {[1 8 15 10 11 6; 0.1 0.1 0.1 0.1 0.1 0.1] [7 2 3 4 17 18; 0.1 0.1 0.1 0.1 0.1 0.1] [13 14 9 16 5 12; 0.1 0.1 0.1 0.1 0.1 0.1]};
% me_Kette = {[1 2 3 2 2 1] [2 1 1 1 3 3] [3 3 2 3 1 2]};
% nChains = 3;
% iterAll = 5;
% dimension = 1;
% 
% KetteGeklebt = MCMC_2DNormal_Swap_sPHS(me_Kette, Kette, nChains, iterAll, dimension)