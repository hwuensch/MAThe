
dimension  = 20;
startvalue = 0;
proptype   = 40;

ESSmeanALL = zeros(20,10,4); % 4, weil: 10, 8, 5, 4 # Ketten
worlds     = [10 8 5 4];
ctr = 1;
for iterAll    = [1000 1250 2000 2500]
    world_size = worlds(ctr);
    for run=1:10
        folder = sprintf('schwarz/output/')
        
        %%% kurze Ketten
        
        world_rank = 0;
        swap       = 1;
        %
        Ketten  = zeros(iterAll+1,dimension,world_size);
        Ketten2 = zeros(world_size*(iterAll+1),dimension);
        Accrate = zeros(1,world_size);
        ESS     = zeros(dimension,world_size);
        pValue  = zeros(world_size, dimension);
        Zeiten  = zeros(2*iterAll+2,world_size);
        nSwaps  = zeros(1,world_size);
        for rank=0:world_size-1
            filename           = sprintf('%s%d_iter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',folder,run,iterAll,dimension,startvalue,proptype,swap,rank)
            fileChain          = importfileChain(filename, dimension);
            Kette              = fileChain(:,1:dimension);
            Ketten(:,:,rank+1) = Kette;
            Ketten2(rank*(iterAll+1)+1:(rank+1)*(iterAll+1),:) = Kette;
            nSwaps(1,rank+1)   = sum(fileChain(:,end)==2);
            
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
        ESSmeanALL(:,run,ctr) = meanESS;
    end
    ctr = ctr+1;
end
ESSmean = reshape(reshape(round(mean(ESSmeanALL,2)),[],1),20,4)
%% plot
figure;
plot(ESSmean,'LineWidth',2)
grid on
ax = gca;
ax.FontSize = 20; ax.FontWeight = 'bold';
xlabel('Parameter'); ylabel('ESS')
legend('10 Ketten','8 Ketten','5 Ketten','4 Ketten','Location','best','Orientation','horizontal')
%%