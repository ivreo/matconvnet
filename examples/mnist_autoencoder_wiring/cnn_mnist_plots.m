close all; clear; clc;
addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');
addpath('~/proj/tools/matlab2tikz/src');


leg = {'LISTA', 'IT'};

% network options 
for expDir = {'./TTTlistaSTsgn', './IT_ST'} ;

expDir = expDir{1};
epoch = findLastCheckpoint(expDir) ;
file = fullfile(expDir, sprintf('net-epoch-%d.mat', epoch) ) ;
[net, state, stats] = loadState(file);

figure(1), hold on, plot([stats.train.psnr]) ;    
figure(2), hold on, plot([stats.val.psnr]) ;      
figure(3), hold on, plot([stats.val.Sparsity1]) ;



end

figure(1);
ylabel('psnr'); xlabel('epochs'); legend(leg, 'Location', 'southeast'); grid off; legend boxoff ;
name = '~/figures/training_lista_it_st_val.tikz' ;
matlab2tikz(name, 'height', '\figureheight', 'width', '\figurewidth');

figure(2);
ylabel('psnr'); xlabel('epochs'); legend(leg, 'Location', 'southeast'); grid off; legend boxoff ;
name = '~/figures/training_lista_it_st_train.tikz' ;
matlab2tikz(name, 'height', '\figureheight', 'width', '\figurewidth');

figure(3);
ylabel('sparsity'); xlabel('epochs'); legend(leg, 'Location', 'northeast'); grid off; legend boxoff ;
name = '~/figures/training_lista_it_st_train_sparsity.tikz' ;
matlab2tikz(name, 'height', '\figureheight', 'width', '\figurewidth');


