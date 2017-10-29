close all; clear; clc;

addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');

global isFC;
isFC = 0;
global initfn;
if isFC
    initfn = @cnn_fc_mnist_init;
else
    initfn = @cnn_conv_mnist_init;
end

global solverfn;
solverfn = [];
solverfn = @solver.adam;

global learningRate;
learningRate = 1e-4 * ones(1,1);

global noiseSD;
noiseSD = 0.2;

global numEpochs;
numEpochs = length(learningRate);
numEpochs = 10;

global layerType;

global iters;
iters = 64;
iters = 4;

expDir = [ '~/exps4/delete_projSGD' num2str(iters(end))];
expDir = ['./delete/'];

layerType = 'IT';
[net, info] = cnn_mnist( ...
    'train', struct('gpus', 4), ...
    'expDir', expDir);

figure;
D = net.layers{1}.weights{1};
vl_imarraysc(reshape(squeeze(D),28,28,[]),'spacing',2);
colormap gray;
axis equal;
axis image off;

figure;
D = net.layers{3}.weights{1};
vl_imarraysc(reshape(squeeze(D),28,28,[]),'spacing',2);
colormap gray;
axis equal;
axis image off;

