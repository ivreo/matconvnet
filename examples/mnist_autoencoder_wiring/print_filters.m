close all; clear; clc;
addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');


% LISTA

expDir = './TTTlistaSTsgn' ;
epoch = findLastCheckpoint(expDir) ;
file = fullfile(expDir, sprintf('net-epoch-%d.mat', epoch) ) ;
net = loadState(file);   

figure(1)
id = net.getParamIndex('e_IT1_Filters_we'); 
D = net.params(id).value;
vl_imarraysc(reshape(squeeze(D),7,7,[]),'spacing',1, 'layout', [4,16]);

figure(2)
id = net.getParamIndex('e_IT1_Filters_wd'); 
D = net.params(id).value;
vl_imarraysc(reshape(squeeze(D),7,7,[]),'spacing',1, 'layout', [4,16]);


% IT

expDir = './IT_ST' ;
epoch = findLastCheckpoint(expDir) ;
file = fullfile(expDir, sprintf('net-epoch-%d.mat', epoch) ) ;
net = loadState(file);   

figure(3),
id = net.getParamIndex('e_IT1_Filters'); 
D = net.params(id).value;
vl_imarraysc(reshape(squeeze(D),7,7,[]),'spacing',1, 'layout', [4,16]);


figure(1); colormap gray; axis equal; axis image off;
name = '~/figures/training_lista_it_filters_we.png' ;
saveas(gcf,name);

figure(2); colormap gray; axis equal; axis image off;
name = '~/figures/training_lista_it_filters_wd.png' ;
saveas(gcf,name);

figure(3); colormap gray; axis equal; axis image off;
name = '~/figures/training_lista_it_filters_D.png' ;
saveas(gcf,name);


