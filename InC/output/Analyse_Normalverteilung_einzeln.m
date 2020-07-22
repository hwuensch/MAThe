iterAll    = 1000;
dimension  = 20;
startvalue = 0;
proptype   = 40;
swap       = 0;
world_rank = 0;
%
filename  = sprintf('iter%d_dim%d_start%d_prop%d_swap%d_rank%d.txt',iterAll,dimension,startvalue,proptype,swap,world_rank)
fileChain = importfileInfo(filename, dimension);

% Diagnosen
Kette = fileChain(:,1:dimension);

%% Akzeptanzrate
Accrate = fileChain(end,end)

%% ESS
ESS = MCMC_ESS(Kette', iterAll, dimension)

%% Geweke
[zz,pp]=gewekeplot(Kette);
[z,p] = geweke(Kette)

%% Gelman-Rubin Factor (Potential Scale Reduction Factor)
[R,neff,V,W,B] = psrf(Kette);
R
neff

% #Iterationen bis Konvergenz also R<1.1 ist
Rhat=nan(iterAll,dimension);
tstart = 10; tend = iterAll;
tStop=tstart;
for t=tstart:tend+1
    Rhat(t,:) = psrf(Ketten(1:t,:,:));
    if all(Rhat(t,:)<1.1) && tStop==tstart
        tStop = t
    end
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
%% plots
FontSize = 20;
%%% scatter der Kette + contour der Zielverteilung
% close all
if dimension > 1
    n = 100;
    x = linspace(min(fileChain(:,1)),max(fileChain(:,1)),n);
    y = linspace(min(fileChain(:,2)),max(fileChain(:,2)),n);
    [X,Y] = meshgrid(x,y);
    Z = reshape(mvnpdf([reshape(X,[],1) reshape(Y,[],1)]),n,n);
    
    ctr = 1;
    figure
    hold on
    contour3(X,Y,Z); contour2D=ctr; ctr=ctr+1;
    scatter(fileChain(:,1),fileChain(:,2)); scatterAll=ctr; ctr=ctr+1;
    scatter(fileChain(1,1),fileChain(1,2)); scatterStart=ctr; ctr=ctr+1; 
    grid on
    xlabel('x_1'); ylabel('x_2'); zlabel('p(x)');
    ax = gca;
    ax.FontWeight = 'bold'; ax.FontSize = FontSize;
    ax.Children(ctr-contour2D).LineWidth = 2.5; ax.Children(ctr-contour2D).DisplayName = 'Zielverteilung';
    ax.Children(ctr-scatterAll).Marker = '+'; ax.Children(ctr-scatterAll).LineWidth = 1.5; ax.Children(ctr-scatterAll).DisplayName = 'Markovkette';
    ax.Children(ctr-scatterStart).LineWidth = 2; ax.Children(ctr-scatterStart).DisplayName = 'Startpunkt';
    legend('Location','northwest')
end
%%% einzelne Markovketten als subplot
% fig = figure;
% % fig.WindowState = 'maximized';
% for d=1:dimension
%     subplot(dimension,1,d)
%     p = plot(fileLog(:,d));
%     grid on
%     ax = gca;
%     ax.XLim = [0 size(fileLog,1)];
%     ax.FontWeight = 'bold'; ax.FontSize = FontSize;
%     ax.Children.LineWidth = 2;
% end
% %% einzelne Histogramme
% nBins = 40;
% for d=1:dimension
%     fig = figure;
%     histfit(fileLog(:,d),nBins,'kernel');
%     xlabel(sprintf('x_%d',d));
%     grid on;
%     ax = gca;
%     ax.FontWeight = 'bold'; ax.FontSize = FontSize;
% end

%%% Markovketten und Histfit nebeneinander in einem figure
nBins = 40;
axAll = [];
fig = figure;
fig.WindowState = 'maximized';
for d=1:dimension
%     subplot(dimension,2,2*d-1)
%     p = plot(fileChain(:,d),'x-');
%     xlabel('Iteration'); ylabel(sprintf('x_%d',d));
%     grid on
%     ax = gca;
%     ax.XLim = [0 size(fileChain,1)];
%     ax.FontWeight = 'bold'; ax.FontSize = FontSize;
%     ax.Children.LineWidth = 2;
    
    subplot(dimension,2,2*d)
    histfit(fileChain(:,d),nBins,'kernel');
%     xlabel(sprintf('x%d',d));
    grid on;
    axh = gca; axAll = [axAll axh];
    axh.FontWeight = 'bold'; axh.FontSize = FontSize;
end
linkaxes(axAll)
subplot(dimension,2,1:2:2*dimension)
st = stackedplot(fileChain(:,1:dimension));
grid on
axst = gca;
axst.LineWidth = 2; axst.FontSize = FontSize;
axst.XLabel = 'Iteration';
% axst.DisplayLabels = {};

%%