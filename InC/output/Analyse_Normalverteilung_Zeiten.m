iterAll    = 1000;
dimension  = 20;
startvalue = 0;
proptype   = 40;
swap       = 1;
world_rank = 0;
world_size = 10;
runs       = 10;
%%
Ketten  = zeros(iterAll+1,dimension,world_size);
Ketten2 = zeros(world_size*(iterAll+1),dimension);
Zeiten  = zeros(3*iterAll+2,world_size);
nSwaps  = zeros(1,world_size);
yysPHS  = zeros(world_size,5,runs);
yy      = zeros(world_size,5,runs);
%% Analyse bzgl mehrerer Durchläufe
for swap = [0 1]
    for run=1:runs
        for rank=0:world_size-1
            filename           = sprintf('%d_iter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',run,iterAll,dimension,startvalue,proptype,swap,rank)
            fileChain          = importfileChain(filename, dimension);
            Kette              = fileChain(:,1:dimension);
            Ketten(:,:,rank+1) = Kette;
            Ketten2(rank*(iterAll+1)+1:(rank+1)*(iterAll+1),:) = Kette;
            nSwaps(1,rank+1)   = sum(fileChain(:,end)==2);
            
            filename         = sprintf('%d_iter%d_dim%d_start%d_prop%d_swap%d_rank%d_times.txt',run,iterAll,dimension,startvalue,proptype,swap,rank)
            Zeiten(:,rank+1) = importfileTimes(filename);
            
        end
        %% Zeiten
        overall   = Zeiten(end,:)
        setup     = Zeiten(1,:)
        mcmciters = overall - setup
        
        broadcast = sum(Zeiten(2:3:end-1,:))
        swaptime  = sum(Zeiten(3:3:end-1,:))
        MHtime    = sum(Zeiten(4:3:end-1,:))
        Rest      = mcmciters - broadcast - MHtime
        
        y = [MHtime; broadcast; swaptime; setup; Rest]';
        
        if swap
            yysPHS(:,:,run) = y;
            nSwapssPHS(run,:) = nSwaps;
        else
            yy(:,:,run) = y;
        end
        
    end
    
    
    if swap
        y = mean(yysPHS,3);
        nSwaps = mean(nSwapssPHS,1);
        ax2 = subplot(1,2,2);
    else
        y = mean(yy,3);
        figure
        ax1 = subplot(1,2,1);
    end
    bplot = bar(y,'stacked');
    grid on
    xlabel('Kette')
    ylabel('Sekunden')
    legend('MH','Bcast', 'swap','setup','Rest','Location','northwest','Orientation','horizontal')
    ax = gca;
    ax.FontSize = 20;
    ax.FontWeight = 'bold';
    if swap
        text(1:length(nSwaps),sum(y(:,1:2),2),num2str(round(nSwaps')),'FontSize',15,'FontWeight','bold','vert','bottom','horiz','center');
        % box off
    end
end
%
linkaxes([ax1 ax2],'y')

