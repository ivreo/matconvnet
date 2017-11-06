close all; clear; clc;

addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../matlab/simplenn');
addpath('../../../vlfeat/toolbox/imop');

global noiseSD;
noiseSD = 0.2;

% solver options
train.learningRate = 1e-3;
train.numEpochs = 50 ;
train.batchSize = 128 ;
train.solver = @solver.adam;
%train.solver = [];
train.gpus = 4;
train.errorFunction = 'psnr';

% network options %
global layerType;
layerType = 'IT';
global iters;
iters = 4;
global isFC;
isFC = 0;
if isFC
    net = cnn_fc_mnist_init();
else
    net = cnn_conv_mnist_init();    
end
net.meta.trainOpts.tiedFilters = 1;
networkType = 'simplenn' ;
expDir = './delete/';

% Training
[net, info] = cnn_mnist( ...
    'train', train, ...
    'expDir', expDir, ...
    'networkType', networkType,...
    'network', net);

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

