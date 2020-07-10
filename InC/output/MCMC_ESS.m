function ESS = MCMC_ESS(Kette,iterAll,dimension)
%% nach Schmidl et. al, 2013, S. 9
% 
% Kette kann mehrdimensional sein:
% Kette = dimension X iterAll+1,
%   d.h. die Kettenglieder sind Spalte f�r Spalte sortiert

%% inputs pruefen
if size(Kette,1)~=dimension
    if size(Kette,2)==dimension
        Kette = Kette';
    else
        ESS = -1;
        return
    end
end
%%

MW_est  = zeros(dimension,1); % mean(Kette(1:dimension,:),2);
Var_est = ones(dimension,1);  % var(Kette(1:dimension,:),0,2);

autocorrfct_est = zeros(dimension,iterAll+1);
ESS_dim         = zeros(dimension,1);
T_stop          = zeros(dimension,1);
zugriff         = 1:dimension;
nBreak          = 0;

for tau = 1:iterAll % lag \tau
    autocorrfct_est(zugriff,tau) = MCMC_AutocorrEst(Kette(zugriff,:),iterAll,dimension-nBreak,MW_est(zugriff,1),Var_est(zugriff,1),tau);
    p1 = (1 - tau/(iterAll+1)) .* autocorrfct_est(zugriff,tau);
    ESS_dim(zugriff,1) = ESS_dim(zugriff,1) + p1;
    
    argmin = autocorrfct_est(zugriff,tau) < 0.05; % die 0.05 sind willk�rlich von Hoffman and Gelman, 2011, Anhang A gew�hlt
    if any(argmin) 
        T_stop(zugriff(argmin),1) = tau;
        nBreak = nBreak + sum(argmin);
        zugriff = zugriff(~argmin);
    end
    if nBreak == dimension
        break;
    end
end
ESS_dim = 1 + 2 * ESS_dim;
ESS_dim = (iterAll + 1) ./ ESS_dim;

ESS = max(ESS_dim);
end