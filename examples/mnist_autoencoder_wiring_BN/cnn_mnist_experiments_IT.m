close all; clear; clc;

addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');

global initfn;
initfn = @cnn_mnist_init_ISTA;
%initfn = @cnn_mnist_init_ISTA_3BN;

global solverfn;
solverfn = @solver.adam;
solverfn = [];

global learningRate;
learningRate = 1e-4;

global numEpoch;
numEpoch = 200;

global noiseSD;
noiseSD = 0.2;

global iters;
iters = 0;

%expDir = ['/home/reyotero/exps/mnist_autoencoder_3_3_1_8_ADAM=1e-4_iters=' num2str(iters)];
expDir = './delete';

[net, info] = cnn_mnist( ...
    'train', struct('gpus', 4), ...
    'expDir', expDir);

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

