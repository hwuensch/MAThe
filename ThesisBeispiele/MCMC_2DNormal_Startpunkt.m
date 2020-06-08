function theta_0 = MCMC_2DNormal_Startpunkt(dimension,wahl)
switch wahl
    case 21
        theta_0 = 0.01 * ones(dimension,1);
    case 22
        theta_0 = 0.02 * ones(dimension,1);
    case 23
        theta_0 = 0.03 * ones(dimension,1);
    case 24
        theta_0 = 0.04 * ones(dimension,1);
    case 25
        theta_0 = 0.05 * ones(dimension,1);
    case 26
        theta_0 = 0.06 * ones(dimension,1);
    case 27
        theta_0 = 0.07 * ones(dimension,1);
    case 28
        theta_0 = 0.08 * ones(dimension,1);
    case 29
        theta_0 = 0.09 * ones(dimension,1);
    case 99
        theta_0 = -5 * [1:dimension]; %[-5; -10; ...];
    otherwise
        theta_0 = wahl * ones(dimension,1);
end
end