function autocorrfct_est = MCMC_2DNormal_AutocorrEst(Kette,iterAll,dimension,MW_est,Var_est,tau)
%% nach Schmidl et al., 2013, S. 9
%
% Kette und MW_est können mehrdimensional sein:
% Kette = dimension X iterAll+1
% MW_est = dimension X 1
%
% autocorrfct_est ist dann auch mehrdimensional:
% autocorrfct_est = dimension X 1 = für jede dimension die geschaetzte
% Autokorrelationsfunktion

%%
% Summe = 0;
% for j = (tau:iterAll)+1
%     Summe = Summe + (Kette(1:dimension,j) - MW_est(1:dimension,1)) .* (Kette(1:dimension,j-tau) - MW_est(1:dimension,1));
% end
Summe = sum((Kette(1:dimension,(tau:iterAll)+1) - MW_est(1:dimension,1)) .* (Kette(1:dimension,1:(iterAll+1-tau)) - MW_est(1:dimension,1)),2);

autocorrfct_est = Summe ./ (Var_est .* (iterAll + 1 - tau));
end