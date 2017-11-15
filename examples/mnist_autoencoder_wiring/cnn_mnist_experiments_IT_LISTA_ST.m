close all; clear; clc;

addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');

global noiseSD;
noiseSD = 0.2;

% solver options
train.learningRate = 1e-3;
train.numEpochs = 100 ;
train.batchSize = 3 ;
train.solver = @solver.adam;
train.weightDecay = 1e-4;
train.gpus = 4;

% network options 
% global layerType;
% layerType = 'IT';
global iters;
iters = 3;
global isFC;
isFC = 0;
net = cnn_mnist_init_csLISTA_ST();
%net = cnn_mnist_init_IT();
%net.meta.trainOpts.tiedFilters = 1;
networkType = 'dagnn' ;
expDir = 'TTT';
expDir = 'TTTlistaST';
expDir = 'TTTlistaSTsgn';
% Training
[net, info] = cnn_mnist( ...
    'train', train, ...
    'expDir', expDir, ...
    'networkType', networkType,...
    'network', net);



figure;
D = net.params(net.getParamIndex('e_IT1_Filters_we')).value;
vl_imarraysc(reshape(squeeze(D),7,7,[]),'spacing',2);
colormap gray;
axis equal;
axis image off;

figure;
D = net.params(net.getParamIndex('d_IT1_Filters')).value;
vl_imarraysc(reshape(squeeze(D),5,5,[]),'spacing',2);
colormap gray;
axis equal;
axis image off;