iterAll    = 1000;
dimension  = 20;
startvalue = 0;
proptype   = 40;
swap       = 0;
world_rank = 0;
world_size = 10;
%%
Ketten  = zeros(iterAll+1,dimension,world_size);
Accrate = zeros(1,world_size);
ESS     = zeros(dimension,world_size);
pValue  = zeros(world_size, dimension);
for rank=0:world_size-1
    filename           = sprintf('iter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',iterAll,dimension,startvalue,proptype,swap,rank)
    fileChain          = importfileInfo(filename, dimension);
    Kette              = fileChain(:,1:dimension);
    Ketten(:,:,rank+1) = Kette;
    
    %%% Akzeptanzrate
    Accrate(1,rank+1)  = fileChain(end,end);
    
    %%% ESS
    ESS(:,rank+1)      = MCMC_ESS(Kette', iterAll, dimension);
    
    %%% Geweke
    % [zz,pp]=gewekeplot(Kette);
    [~,pValue(rank+1,:)] = geweke(Kette);
end

%% Diagnosen
%%% Akzeptanzraten
Accrate

%%% ESS
minmeanmaxESS = round([min(ESS,[],2) mean(ESS,2) max(ESS,[],2)])

%%% Geweke
pValue

%%% Gelman-Rubin Factor (Potential Scale Reduction Factor)
Rhat = psrf(Ketten)

%% #Iterationen bis Konvergenz also R<1.1 ist
Rhat=nan(iterAll,dimension);
tstart = 10; tend = 1000;
tStop=tstart;
for t=tstart:tend+1
    Rhat(t,:) = psrf(Ketten(1:t,:,:));
    if all(Rhat(t,:)<1.1)
        tStop = t
        break;
    end
end
if tStop==tstart
    tStop = t;
end
figure;
semilogy(Rhat(tstart:tStop,:));
hold on
line([tstart tStop],[1.1 1.1],'Color','#000000','DisplayName','Grenze','LineWidth',1.5)
grid on
ax = gca;
ax.LineWidth = 1.5; ax.FontSize = 20; ax.FontWeight = 'bold';
for i=2:dimension+1
    ax.Children(i).DisplayName = sprintf('x %d',dimension+2-i);
    ax.Children(i).LineWidth   = 1.5;
end
legend
%%