iterAll    = 1000;
dimension  = 3;
startvalue = 3;
proptype   = 60;
world_rank = 0;
%%
filename = sprintf('iter%d_dim%d_start%d_prop%d_rank%d.txt',iterAll,dimension,startvalue,world_rank)
fileLog  = importfileInfo(filename, dimension);

%%
close all
if dimension > 2
    n = 100;
    x = linspace(min(fileLog(:,1)),max(fileLog(:,1)),n);
    y = linspace(min(fileLog(:,2)),max(fileLog(:,2)),n);
    [X,Y] = meshgrid(x,y);
    Z = reshape(mvnpdf([reshape(X,[],1) reshape(Y,[],1)]),n,n);
    
    figure
    hold on
    scatter(fileLog(:,1),fileLog(:,2))
    contour3(X,Y,Z)
    grid on
end
%%
fig = figure;
fig.WindowState = 'maximized';
for d=1:dimension
    subplot(dimension,1,d)
    p = plot(fileLog(:,d));
    grid on
    ax = gca;
    ax.XLim = [0 size(fileLog,1)];
    ax.FontWeight = 'bold'; ax.FontSize = 15;
    ax.Children.LineWidth = 2;
end
%%