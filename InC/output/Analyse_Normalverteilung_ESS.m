
dimension  = 20;
startvalue = 0;
proptype   = 40;

ESSmeanALL = zeros(20,10,3);

for run=1:10
    folder = sprintf('schwarz/output%d/',run)
    
    %%% lange Kette
    iterAll    = 10000;
    swap       = 0;
    world_rank = 0;
    %
    filename  = sprintf('%siter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',folder,iterAll,dimension,startvalue,proptype,swap,world_rank)
    fileChain = importfileChain(filename, dimension);
    
    % Diagnosen
    Kette = fileChain(:,1:dimension);
    ESS = round(MCMC_ESS(Kette', iterAll, dimension))
    ESSmeanALL(:,run,1) = ESS;
    
    %%% kurze Ketten
    iterAll    = 1000;
    world_rank = 0;
    world_size = 10;
    %
    Ketten  = zeros(iterAll+1,dimension,world_size);
    Ketten2 = zeros(world_size*(iterAll+1),dimension);
    Accrate = zeros(1,world_size);
    ESS     = zeros(dimension,world_size);
    pValue  = zeros(world_size, dimension);
    Zeiten  = zeros(2*iterAll+2,world_size);
    nSwaps  = zeros(1,world_size);
    for swap = [0 1]
        for rank=0:world_size-1
            filename           = sprintf('%siter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',folder,iterAll,dimension,startvalue,proptype,swap,rank)
            fileChain          = importfileChain(filename, dimension);
            Kette              = fileChain(:,1:dimension);
            Ketten(:,:,rank+1) = Kette;
            Ketten2(rank*(iterAll+1)+1:(rank+1)*(iterAll+1),:) = Kette;
            nSwaps(1,rank+1)   = sum(fileChain(:,end)==2);
            
            filename         = sprintf('%siter%d_dim%d_start%d_prop%d_swap%d_rank%d_times.txt',folder,iterAll,dimension,startvalue,proptype,swap,rank)
            Zeiten(:,rank+1) = importfileTimes(filename);
            
            %%% Akzeptanzrate
            Accrate(1,rank+1)  = fileChain(end,end);
            
            %%% ESS
            ESS(:,rank+1)      = MCMC_ESS(Kette', iterAll, dimension);
            
            %%% Geweke
            % [zz,pp]=gewekeplot(Kette);
            [~,pValue(rank+1,:)] = geweke(Kette);
        end
        %%% ESS
        meanESS = round(MCMC_ESS(Ketten2,(iterAll+1)*world_size,dimension))
        ESSmeanALL(:,run,swap+2) = meanESS;
    end
    
end
%% plot