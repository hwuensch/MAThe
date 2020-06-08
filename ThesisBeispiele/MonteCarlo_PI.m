% function MonteCarlo_PI(iterAll)

path2save='/Users/hendrik_w/Documents/Studium/2Köln/Master/5.Sem-WS1819/Bayer/vorlage_arbeit/input/MCMCBeispiele/MonteCarloPI';

iterAll = 10.^[2:2:8];
fprintf(1,"Approximation von pi\n#Stichproben\t|\tApproximation\t|\tFehler\n")
for R=iterAll
samples_x = rand(1,R);
samples_y = rand(1,R);

pi_approx_n = 4 * sum(samples_x.^2 + samples_y.^2<=1)/R;
pi_error = abs(pi-pi_approx_n);
fprintf(1,"%d\t\t|\t%.6f\t|\t%.4e\n",R,pi_approx_n,pi_error);

% pi_approx_int = 4 .* sum(sqrt(1-samples_x.^2)) ./ R;
% pi_error = abs(pi-pi_approx_int);
% fprintf(1,"%d\t\t|\t%.6f\t|\t%.4e\n",R,pi_approx_int,pi_error);

% fig_R = figure(R);
% set(gcf,'Units','normalized','OuterPosition',[0 0 0.625 1])
% axis equal
% plot(samples_x,samples_y,'.','Color',[0.3010 0.7450 0.9330])
% hold on
% plot(sort(samples_x),sqrt(1-sort(samples_x).^2),'LineWidth',3,'Color',[0.4660 0.6740 0.1880])
% hold off
% xlabel('x'); ylabel('y');
% grid on
% ax = gca;
% ax.FontWeight = 'bold'; ax.FontSize = 50;
% ax.XTick = 0:0.5:1;
% ax.YTick = 0:0.5:1;
% % pbaspect([1 1 1]) % quadratische Achsen
% % daspect([1 1 1]) % relative Achsenbeschriftung gleich
% 
% saveas(fig_R,sprintf('%s/MonteCarloPI_%d',path2save,R),'epsc');
% close(fig_R);
end
%%
% end