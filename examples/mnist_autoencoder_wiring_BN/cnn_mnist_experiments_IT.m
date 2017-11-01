close all; clear; clc;

addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');


global noiseSD;
noiseSD = 0.2;

% solver options
train.learningRate = 1e-3;
train.numEpochs = 10 ;
train.batchSize = 128 ;
train.solver = @solver.adam;
train.weightDecay = 1e-4;
train.gpus = 2;

% network options 
global layerType;
layerType = 'IT';
global iters;
iters = 4;
global isFC;
isFC = 0;
net = cnn_mnist_init_ISTA();
%net.meta.trainOpts.tiedFilters = 1;
networkType = 'dagnn' ;
expDir = './delete/';

% Training
[net, info] = cnn_mnist( ...
    'train', train, ...
    'expDir', expDir, ...
    'networkType', networkType,...
    'network', net);

% figure;
% D = net.params(net.getParamIndex('e_IT1_Filters')).value;
% vl_imarraysc(reshape(squeeze(D),5,5,[]),'spacing',2);
% colormap gray;
% axis equal;
% axis image off;
% 
% figure;
% D = net.params(net.getParamIndex('d_IT1_Filters')).value;
% vl_imarraysc(reshape(squeeze(D),5,5,[]),'spacing',2);
% colormap gray;
% axis equal;
% axis image off;

