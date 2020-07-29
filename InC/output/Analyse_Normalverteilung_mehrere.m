iterAll    = 1000;
dimension  = 20;
startvalue = 0;
proptype   = 40;
swap       = 1;
world_rank = 0;
world_size = 10;
%%
Ketten  = zeros(iterAll+1,dimension,world_size);
Ketten2 = zeros(world_size*(iterAll+1),dimension);
Accrate = zeros(1,world_size);
ESS     = zeros(dimension,world_size);
pValue  = zeros(world_size, dimension);
Zeiten  = zeros(2*iterAll+2,world_size);
nSwaps  = zeros(1,world_size);
for rank=0:world_size-1
    filename           = sprintf('iter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',iterAll,dimension,startvalue,proptype,swap,rank)
    fileChain          = importfileChain(filename, dimension);
    Kette              = fileChain(:,1:dimension);
    Ketten(:,:,rank+1) = Kette;
    Ketten2(rank*(iterAll+1)+1:(rank+1)*(iterAll+1),:) = Kette;
    nSwaps(1,rank+1)   = sum(fileChain(:,end)==2);
    
    filename         = sprintf('iter%d_dim%d_start%d_prop%d_swap%d_rank%d_times.txt',iterAll,dimension,startvalue,proptype,swap,rank)
    Zeiten(:,rank+1) = importfileTimes(filename);
    
    %%% Akzeptanzrate
    Accrate(1,rank+1)  = fileChain(end,end);
    
    %%% ESS
    ESS(:,rank+1)      = MCMC_ESS(Kette', iterAll, dimension);
    
    %%% Geweke
    % [zz,pp]=gewekeplot(Kette);
    [~,pValue(rank+1,:)] = geweke(Kette);
end
%% Diagnosen
%%% Anzahl Beteiligung swaps
nSwaps 

%%% Akzeptanzraten
Accrate

%%% ESS
meanESS = round(MCMC_ESS(Ketten2,(iterAll+1)*world_size,dimension))

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
    tStop = t
end

figure;
semilogy(Rhat(tstart:tStop,:));
hold on
line([tstart tStop],[1.1 1.1],'Color','#000000','DisplayName','Grenze','LineWidth',1.5)
grid on
ax = gca;
ax.XLim = [tstart tStop];
ax.LineWidth = 1.5; ax.FontSize = 20; ax.FontWeight = 'bold';
for i=2:dimension+1
    ax.Children(i).DisplayName = sprintf('x %d',dimension+2-i);
    ax.Children(i).LineWidth   = 1.5;
end
legend
%% Zeiten
overall   = Zeiten(end,:)
setup     = Zeiten(1,:)
mcmciters = overall - setup

swaptime  = sum(Zeiten(2:2:end-1,:))
MHtime    = sum(Zeiten(3:2:end-1,:))
Rest      = mcmciters - swaptime - MHtime

y = [MHtime; swaptime; setup; Rest]'
%%
% figure
% ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
bplot = bar(y,'stacked');
grid on
xlabel('Kette')
ylabel('Sekunden')
legend('MH','sPHS','setup','Rest','Location','south','Orientation','horizontal')
ax = gca;
ax.FontSize = 20;
ax.FontWeight = 'bold';
if swap
    text(1:length(nSwaps),y(:,1),num2str(round(nSwaps')),'FontSize',15,'FontWeight','bold','vert','bottom','horiz','center');
    % box off
end
%%
linkaxes([ax1 ax2],'y')

%% Analyse bzgl mehrerer Durchläufe
% close 
% i=10;
% if swap
%     yysPHS(:,:,i) = y
%     nSwapssPHS(i,:) = nSwaps
% else
%     yy(:,:,i) = y
% end

% y = mean(yy,3)
% y = mean(yysPHS,3)
% nSwaps = mean(nSwapssPHS,1)
