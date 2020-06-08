function MCMC_2DNormal_Plot(iterAll,sigma_wahl_i,dimension,theta_curr,theta_can,sigma_proposal,startpunkt,sigma_posterior,blocksize,accept)

if blocksize == dimension
    %% plot markovkette x-dimensional
    fig_dimension = figure('Name',sprintf('Iterationen: %d, sigma: %d, dim: %d',iterAll,sigma_wahl_i,dimension),'visible','off');
    set(gcf,'Units','normalized','OuterPosition',[0 0 0.8 1])
    % subplot(2,2,[1 3])
    plot(theta_curr(1,:),theta_curr(2,:),'.-','LineWidth',3) % 2D Markovkette
    grid on
    hold on;
    %% contour der Proposal beim Startpunkt
    sigma_prop = MCMC_2DNormal_KovMat(sigma_wahl_i,0.9,sigma_proposal^2,dimension,blocksize);
    mu_prop = theta_curr(1:2,1);
    mesh_prop = 3*[-1:0.01:1].*[sigma_prop(1,1);sigma_prop(2,2)]+mu_prop;
    [X_prop,Y_prop] = meshgrid(mesh_prop(1,:),mesh_prop(2,:));
    x_prop = [reshape(X_prop,[],1) reshape(Y_prop,[],1)];
    z_prop = mvnpdf(x_prop,mu_prop',sigma_prop(1:2,1:2));
    z_prop = reshape(z_prop,length(X_prop),[]);
    [~,c] = contour(mesh_prop(1,:),mesh_prop(2,:),z_prop,2,'k--'); c.LineWidth = 3;
    xlabel('x_1');ylabel('x_2');zlabel('pdf');
    
    
    %% Zielverteilung in contour plotten
    rho = 0.9;
    plotBereich = -3:0.1:3;
    mu_mvn = [0 0];
    sigma_mvn = MCMC_2DNormal_KovMat(1,rho,1,dimension,blocksize);
    
    x_mvn = mvnrnd(mu_mvn,sigma_mvn(1:2,1:2),1000);
    z_mvn = mvnpdf(x_mvn,mu_mvn,sigma_mvn(1:2,1:2));
    
    % figure
    % scatter3(x_mvn(:,1),x_mvn(:,2),z_mvn);
    
    [X_mvn,Y_mvn] = meshgrid(plotBereich);
    x_mvn = [reshape(X_mvn,[],1) reshape(Y_mvn,[],1)];
    
    z_mvn = mvnpdf(x_mvn,mu_mvn,sigma_mvn(1:2,1:2));
    z_mvn = reshape(z_mvn,length(X_mvn),[]);
    
    % figure
    [~,c] = contour(plotBereich,plotBereich,z_mvn,3,'k-','ShowText','off'); c.LineWidth = 3; % Zielverteilung
    xlabel('x_1');ylabel('x_2');zlabel('pdf');
    daspect([1,1,1])
    % ylim([-10 10])
    
    ax = gca; ax.FontSize = 25; ax.FontWeight = 'bold';
    % end
    saveas(fig_dimension,sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_dimensional.eps",dimension,iterAll,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize),'epsc');
    close(fig_dimension);
else
    %% plot x-dimensionale Markovkette block per block
    
    plotBereich = -3:0.1:3;
    mu_mvn = [0 0];
    sigma_mvn = MCMC_2DNormal_KovMat(1,0.9,1,dimension,2);
    
    x_mvn = mvnrnd(mu_mvn,sigma_mvn(1:2,1:2),1000);
    z_mvn = mvnpdf(x_mvn,mu_mvn,sigma_mvn(1:2,1:2));
    
    % figure
    % scatter3(x_mvn(:,1),x_mvn(:,2),z_mvn);
    
    [X_mvn,Y_mvn] = meshgrid(plotBereich);
    x_mvn = [reshape(X_mvn,[],1) reshape(Y_mvn,[],1)];
    
    z_mvn = mvnpdf(x_mvn,mu_mvn,sigma_mvn(1:2,1:2));
    z_mvn = reshape(z_mvn,length(X_mvn),[]);
    
    rho = 0.9;
%     sigma_wahl_i = 0; %
%     sigma_proposal = 0.75; %
%     sigma_posterior = sqrt(1-rho^2); %
    sigma_prop = MCMC_2DNormal_KovMat(sigma_wahl_i,rho,sigma_proposal^2,dimension,2);
    sigma_post = MCMC_2DNormal_KovMat(1,1,sigma_posterior^2,dimension,2);
    nPoints = 201;
    for iter = 1:3 % jeden Parameter x-mal plotten
        for block = 1:dimension
            can = theta_can(block,iter);
            mu = rho * theta_curr(mod(block,dimension)+1,iter);
            sig_prp = sigma_prop(block,block);
            sig_pos = sigma_post(block,block);
            X      = mu + 4 * sig_prp * linspace(-1,1,nPoints);
            Y_prop = gauss_distribution(X,mu,sig_prp);
            Y_post = gauss_distribution(X,mu,sig_pos);
            faktor = max(Y_prop)/max(Y_post);
            Y_post = Y_post * faktor;
            Z = mu/rho * ones(1,nPoints);
            if block==1
                x_plot = X; y_plot = Z; z_plot_prop = Y_prop; z_plot_post = Y_post;
                x_dot = mu; y_dot = mu/rho; z_dot = 0; x_dotPro = can; y_dotPro = mu/rho; z_dotPro = 0;
            elseif block==2
                x_plot = Z; y_plot = X; z_plot_prop = Y_prop; z_plot_post = Y_post;
                x_dot = mu/rho; y_dot = mu; z_dot = 0; x_dotPro = mu/rho; y_dotPro = can; z_dotPro = 0;
            end
            fig_ketteperblock = figure('Name',sprintf('Iter %d, block %d',iter,block),'visible','off');
            plot3(x_plot,y_plot,z_plot_prop,':','Color','k','LineWidth',3);
            hold on
            plot3(x_plot,y_plot,z_plot_post,'Color','k','LineWidth',3);
            plot3(x_dot,y_dot,z_dot,'+','Color','k','LineWidth',3);
            plot3(x_dotPro,y_dotPro,z_dotPro,'*','Color','k','LineWidth',3);
            [~,c] = contour(plotBereich,plotBereich,z_mvn,3,'k-','ShowText','off'); c.LineWidth = 3; % Zielverteilung
            xlabel('x_1'); ylabel('x_2'); zticks([]);
            grid on
            view([1 -1 1])
            pbaspect([10 10 1])
            ax = gca; ax.FontSize = 18; ax.FontWeight = 'bold';
            
            legend({'Vorschlagsdichte','Bedingte Verteilung','Position','Kandidat','Zielverteilung'},'Location','northwest','FontSize',18)
            
            saveas(fig_ketteperblock,sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_Parameter%d_KettePerBlock.eps",dimension,iter,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize,block),'epsc');
            close(fig_ketteperblock);
        end
    end
end
%% einzelne Markovketten
for param = 1:min(2,dimension)
    fig_ketten = figure('Name',sprintf('Iterationen: %d, sigma: %d, dim: %d',iterAll,sigma_wahl_i,dimension),'visible','off');
    set(gcf,'Units','normalized','OuterPosition',[0 0 0.9 1])
    % figure('Name',sprintf('Markovketten, sigma: %d, dim: %d',sigma_wahl_i, dimension))
    % subplot(2,2,[2 4])
    % ylabels = cell(1,dimension);
    % for i=1:dimension
    %     ylabels(i) = {sprintf('x_%d',i)};
    % end
    % s = stackedplot(theta_curr(1:end-1,:)','DisplayLabels',ylabels);
    plot(theta_curr(param,:),'LineWidth',3);
    hold on
    xlim([0 iterAll])
    xlabel('Iteration'); ylabel(sprintf('x_{%d}',param));
    grid on
    
    ax = gca; ax.FontSize = 40; ax.FontWeight = 'bold';
    
    saveas(fig_ketten,sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_Parameter%d.eps",dimension,iterAll,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize,param),'epsc');
    close(fig_ketten);
end
%% scatter
fig_scatter = figure('Name',sprintf('Iterationen: %d, sigma: %d, dim: %d',iterAll,sigma_wahl_i,dimension),'visible','off');
set(gcf,'Units','normalized','OuterPosition',[0 0 0.8 1])
scatter(theta_curr(1,:),theta_curr(2,:),'.','SizeData',400);
xll = xlim; yll = ylim; limis = [min([xll(1),yll(1),-3]), max([xll(2),yll(2),3])];
xlim(limis);ylim(limis);

xlabel('x_1'); ylabel('x_2')
grid on
ax = gca; ax.FontSize = 25; ax.FontWeight = 'bold';

saveas(fig_scatter,sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_scatter.eps",dimension,iterAll,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize),'epsc');
close(fig_scatter);
%% histogram
for i=1:min(2,dimension)
    figure('Name','histfit','visible','off');
    h = histfit(theta_curr(i,:));
    nBins = size(h(1).XData,2);
    h = histfit(theta_curr(i,:), nBins,'kernel');%,'FaceColor','#0072BD');
    fig_hist = figure('Name',sprintf('Iterationen: %d, sigma: %d, dim: %d',iterAll,sigma_wahl_i,dimension),'visible','off');
    set(gcf,'Units','normalized','OuterPosition',[0 0 0.8 1])
    histogram(theta_curr(i,:),'FaceColor','#0072BD');
    hold on
    x_plot = -3.5:0.01:3.5;
    y_plot_prop = gauss_distribution(x_plot,zeros(1,length(x_plot)),1);
    faktor = max(h(2).YData) / max(y_plot_prop); % auf histogram skalieren
%     faktor = h(2).YData(floor(length(h(2).YData))/2) / y_plot_prop((length(y_plot_prop)-1)/2); % auf histogram an der stelle 0 skalieren
    y_plot_prop = y_plot_prop * faktor;
    plot(x_plot,y_plot_prop,'Color','#000000','LineWidth',3) % #0072BD = MATLAB blau
    
    xlabel(sprintf('x_{%d}',i)); yticks([])
    ax = gca; ax.FontSize = 40; ax.FontWeight = 'bold';
    legend({sprintf('T = %d',iterAll),'N(0,1)'},'Location','northeast')
    saveas(fig_hist,sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_hist%d.eps",dimension,iterAll,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize,i),'epsc');
    close(fig_hist);
end

%% Akzeptanzrate
fID = fopen(sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_absolut.tex",dimension,iterAll,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize),'w');
fprintf(fID,'%d',accept);
fclose(fID);
accrate = mat2str(floor(1000*accept/iterAll/dimension)/1000);
if accept == 0
    accrate = '0.0';
end
fID = fopen(sprintf("./output/%dD_Normalv_iter%d_prop%d_scal%d_start%d_block%d_rate.tex",dimension,iterAll,sigma_wahl_i,sigma_proposal*100,startpunkt,blocksize),'w');
fprintf(fID,'0,%s',accrate(3:end));
fclose(fID);

end