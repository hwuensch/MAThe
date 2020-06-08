function swapscheme = MCMC_2DNormal_Swapscheme(nChains)
    %% Tausschema für nChains Ketten:
    % immer zwei Ketten tauschen miteinander.
    % Wenn nChains grade ist, tauschen jede Kette mit einer anderen. (keine
    % tauscht nicht! das passiert hier nur, wenn nChains ungrade)
    
    %%
    even = mod(nChains,2)==0;
    if even
        nSwaps = nChains/2;
    else
        nSwaps = (nChains-1)/2;
    end
    ctr = nSwaps*2;
    swapscheme = 1:nChains;
    swapscheme_remain = 1:nChains;
    
    for i = 1:nSwaps
        % Zufallsinteger aus verbleibenden #Ketten ziehen...
        P1_roh = randi(ctr); ctr = ctr-1;
        P2_roh = randi(ctr); ctr = ctr-1;
        % ...umrechnen auf die Kettennr. ...
        P1 = swapscheme_remain(P1_roh);
        swapscheme_remain(P1_roh) = [];
        P2 = swapscheme_remain(P2_roh);
        swapscheme_remain(P2_roh) = [];
        % ...tauschen
        swapP1 = swapscheme(P1);
        swapscheme(P1) = swapscheme(P2);
        swapscheme(P2) = swapP1;
    end
end