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
train.gpus = 3;

% network options 
global layerType;
layerType = 'IT';
global iters;
iters = 10;
global isFC;
isFC = 0;
net = cnn_mnist_conv_init();
if strcmp(layerType,'IT')
    train.weightDecay = 0 ;                                                                                                                            
else
    train.weightDecay = 0.0001 ;
end
%net.meta.trainOpts.tiedFilters = 1;
networkType = 'dagnn' ;
expDir = './delete/';

% Training
[net, info] = cnn_mnist( ...
    'train', train, ...
    'expDir', expDir, ...
    'networkType', networkType,...
    'network', net);





time_axis = [info.train.time];
totalBatches = length(time_axis) / numEpoch;
for i = 2 : numEpoch
    curIdx = totalBatches*(i-1)+1;
    nextIdx = totalBatches*i;
    time_axis(curIdx:nextIdx) = time_axis(curIdx:nextIdx) + time_axis(curIdx-1);
end
time_axis = time_axis - time_axis(1);

figure;
hold on;
plot(time_axis,[info.train.psnr]) ;
grid on;

D = net.params(1).value;
figure;
vl_imarraysc(reshape(squeeze(D),size(D,1),size(D,2),[]),'spacing',2);
colormap gray;
axis equal;
axis image off ;

