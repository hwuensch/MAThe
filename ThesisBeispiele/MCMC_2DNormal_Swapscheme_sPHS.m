function swapscheme = MCMC_2DNormal_Swapscheme_sPHS(iterAll,nChains)
    %% Tausschema für nChains Ketten:
    % genau zwei Ketten tauschen miteinander.
    
    %%
    if nChains > 2
        swapscheme = zeros(2,iterAll);
        for i=1:iterAll
            swapscheme(:,i) = randperm(nChains,2)';
        end
    end
end