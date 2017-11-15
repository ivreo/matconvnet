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
train.gpus = 2;
% train.gpus = [];

% network options 
global layerType;
layerType = 'IT';
global iters;
iters = 3;
global isFC;
isFC = 0;
%net = cnn_mnist_init_csLISTA();
%net = cnn_mnist_init_IT_ST();
net = cnn_mnist_init_IT_ST();
%net.meta.trainOpts.tiedFilters = 1;
networkType = 'dagnn' ;
expDir = 'IT';
% Training
[net, info] = cnn_mnist( ...
    'train', train, ...
    'expDir', expDir, ...
    'networkType', networkType,...
    'network', net);

