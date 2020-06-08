function kovarianzmatrix = MCMC_2DNormal_KovMat(Wahl,rho,scaling,dimension,blocksize)
%%
% Korrelationskoeffizient: $ \rho(x,y) = \frac{Cov(x,y)}{\sigma(x) \sigma(y)} $, 
% d.h. Zielverteilung hat $ \rho = 0,9 $. Wenn ich Proposal danach ausrichten will,
% d.h. gleiche Korrelation aber kleinere Varianzen, und Varianzen $ \sigma^2 $ vorgebe, 
% dann sind die Kovarianzen durch $ Cov(x,y) = \rho(x,y) \cdot \sigma_x^{Proposal} \sigma_y^{Proposal} = 0,9 \cdot \dots $ gegeben.
%%
switch Wahl
    case 0
        kovarianzmatrix = scaling * eye(dimension);
    case 1 % Zielverteilung
        rho = rho * scaling; % scaling ist varianz, also bereits quadriert.
        kovarianzmatrix = rho * ones(dimension);
        kovarianzmatrix = kovarianzmatrix - diag(diag(kovarianzmatrix));
        kovarianzmatrix = ( kovarianzmatrix + scaling * eye(dimension) ); % [1 0.9; 0.9 1];
%         kovarianzmatrix = scaling * ( kovarianzmatrix + eye(dimension) .* [1:dimension].^2 ); % [1 0.9; 0.9 4];
    case -1
        rho = rho * scaling;
        kovarianzmatrix = -rho * ones(dimension);
        kovarianzmatrix = kovarianzmatrix - diag(diag(kovarianzmatrix));
        kovarianzmatrix = ( kovarianzmatrix + scaling * eye(dimension) );
%         kovarianzmatrix = scaling * ( kovarianzmatrix + eye(dimension) .* [dimension:-1:1].^2 ); % [4 -0.9; -0.9 1];
    otherwise
        warning('falsche Zahl für Wahl der Kovarianzmatrix');
        kovarianzmatrix = NaN(2,2);
end
end