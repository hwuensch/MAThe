function KetteGeklebt_Swap = MCMC_2DNormal_Swap(KettenOriginal,nChains,iterAll)
%% führt den Positionswechsel (swap) der Kette im Nachhinein durch
% und zwar nach dem Schema: alle Ketten tauschen partnerweise
if nChains>1
    KettenSwap = KettenOriginal;
    swaps = zeros(nChains,iterAll);
    for i=1:iterAll
        %     swaps(:,i) = randperm(nChains);
        swaps(:,i) = MCMC_2DNormal_Swapscheme(nChains);
        for j=1:nChains
            KettenSwap{j}(:,i+1) = KettenOriginal{swaps(j,i)}(:,i+1);
        end
    end
    
    % Ketten aneinander kleben
    KetteGeklebt_Swap = zeros(dimension+1,nChains*(iterAll+1));
    for chain_curr=1:nChains
        idx = ((chain_curr-1)*(iterAll+1)+1):(chain_curr*(iterAll+1));
        KetteGeklebt_Swap(:,idx) = KettenSwap{chain_curr};
    end
end
end