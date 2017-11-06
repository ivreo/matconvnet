function net = cnn_mnist_fc_init(varargin)

rng('default');
rng(0) ;

global layerType;
global iters;

global stepsize ;
stepsize = 1 ;


net = dagnn.DagNN() ;

net.meta.inputSize = [1 1 28*28] ;
global learningRate;
net.meta.trainOpts.learningRate = learningRate ;
global numEpoch;
net.meta.trainOpts.numEpochs = numEpoch ;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

global DOrate;

net.addLayer('reshape1', dagnn.Reshape('size', [1 1 28*28]), {'input'}, {'reshape1'});

if strcmp(layerType,'conv')
    net.addLayer('conv1', dagnn.Conv('size', [1 1 28*28 1024], 'stride', 1, 'pad', 0,  'inputSize', net.meta.inputSize  ), {'reshape1'}, {'conv1'},  {'conv1f'  'conv1b'});
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'relu1'}, {});
    %last = 'relu1';
    net.addLayer('dropout1', dagnn.DropOut( 'rate', DOrate), {'conv1'}, {'dropout1'});
    last = 'dropout1';
elseif strcmp(layerType,'IT')
    net.addLayer('IT1', dagnn.IT('size', [1 1 28*28 1024], 'iters', iters, 'stepsize', stepsize, 'stride', 1, 'pad', 0, 'inputSize', net.meta.inputSize, 'rate', DOrate), {'reshape1'}, {'IT1'},  {'conv1f'  'conv1b'});
    net.addLayer('PickLast1', dagnn.PickLast('iters', iters), {'IT1'}, {'PickLast1'});
    last = 'PickLast1';
elseif strcmp(layerType,'Lasso')
    net.addLayer('Lasso1', dagnn.Lasso('size', [1 1 28*28 1024], 'iters', iters, 'stride', 1, 'pad', 0, 'eta', 1e-3, 'inputSize', net.meta.inputSize), {'reshape1'}, {'Lasso1'},  {'conv1f'  'conv1b'});
    last = 'Lasso1';
else
	error('Wrong layer type!');
end

net.addLayer('convt1', dagnn.ConvTranspose('size', [1 1 28*28 1024], 'hasBias', false, 'upsample', 1, 'crop', 0), {last}, {'convt1'},  {'conv1f'});

net.addLayer('reshape2', dagnn.Reshape('size', [28 28 1]), {'convt1'}, {'reshape2'});
net.addLayer('objective', dagnn.PDist('noRoot', true), {'reshape2','label'}, 'objective');

net.addLayer('psnr', dagnn.Loss('loss', 'psnr_not_averaged'), {'reshape2','label'}, 'psnr');
net.addLayer('sparsity', dagnn.Loss('loss', 'sparsity_not_averaged'), {last}, 'sparsity');

net.initParams();


if strcmp(layerType,'IT')
    p = net.getParamIndex('conv1f');
    net.params(p).stepsize = stepsize;
    net.params(p).stride = 1;
    net.params(p).pad = 0;
    net.params(p).dilate = 1;
    net.params(p).inputSize = net.meta.inputSize;
    net.params(p).eigvec = []; 
    net.params(p).opts = {'cuDNN'};
else
    p = net.getParamIndex('conv1f');
    net.params(p).stepsize = [];
end


