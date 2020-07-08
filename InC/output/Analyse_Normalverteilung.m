iterAll    = 1000;
dimension  = 3;
startvalue = 3;
world_rank = 0;
%%
filename = sprintf('iter%d_dim%d_start%d_rank%d.txt',iterAll,dimension,startvalue,world_rank)
fileLog = importfileLog(filename, dimension);

%%
close all

n = 100;
x = linspace(min(fileLog(:,1)),max(fileLog(:,1)),n);
y = linspace(min(fileLog(:,2)),max(fileLog(:,2)),n);
[X,Y] = meshgrid(x,y);
xx = reshape(X,[],1);
yy = reshape(Y,[],1);
Z = reshape(mvnpdf([xx yy]),n,n);

figure
hold on
scatter(fileLog(:,1),fileLog(:,2))
contour3(X,Y,Z)

grid on

%%