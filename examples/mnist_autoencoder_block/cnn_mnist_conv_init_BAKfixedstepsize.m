function net = cnn_mnist_conv_init(varargin)

rng('default');
rng(0) ;

stepsize = 2;

global layerType;
global iters;

net = dagnn.DagNN() ;
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts = struct() ;

switch layerType
    case 'conv'
        net.addLayer('conv1', dagnn.Conv('size', [5 5 1 16], 'stride', 1, 'pad', 4), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
        net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'relu1'}, {});
        last = 'relu1';
    case 'IT'
        net.addLayer('IT1', dagnn.IT('size', [5 5 1 16], 'stride', 1, 'pad', 4, 'iters', iters, 'stepsize', stepsize, 'inputSize', net.meta.inputSize, 'rate', 0.5), {'input'}, {'IT1'},  {'conv1f'  'conv1b'});
        net.addLayer('PickLast1', dagnn.PickLast('iters', iters), {'IT1'}, {'PickLast1'});
        last = 'PickLast1';
    case 'Lasso'
        net.addLayer('Lasso1', dagnn.Lasso('size', [5 5 1 16], 'iters', iters, 'stride', 1, 'pad', 4, 'eta', 1e-3, 'inputSize', net.meta.inputSize), {'input'}, {'Lasso1'},  {'conv2f'  'conv1b'});
        last = 'Lasso1';
    otherwise
        error('Wrong layer type!');
end

net.addLayer('convt1', dagnn.ConvTranspose('size', [5 5 1 16], 'hasBias', false, 'upsample', 1, 'crop', 4), {last}, {'convt1'},  {'conv1f'});
net.addLayer('objective', dagnn.PDist(), {'convt1','label'}, 'objective');
%net.addLayer('objective', dagnn.PDist('noRoot', true), {'convt1','label'}, 'objective');  %% CHECK older version

%net.addLayer('psnr', dagnn.Loss('loss', 'psnr_not_averaged'), {'convt1','label'}, 'psnr');
%net.addLayer('sparsity', dagnn.Loss('loss', 'sparsity_not_averaged'), {last}, 'sparsity');
net.addLayer('psnr', dagnn.Loss('loss', 'psnr'), {'convt1','label'}, 'psnr');
net.addLayer('sparsity', dagnn.Loss('loss', 'sparsity'), {last}, 'sparsity');

net.initParams();

net.params(net.getParamIndex('conv1b')).weightDecay = 0.001;


if strcmp(layerType,'IT')
    p = net.getParamIndex('conv1b');
    net.params(p).stepsize = stepsize;
    net.params(p).stride = 1;
    net.params(p).pad = 4;
    net.params(p).dilate = 1;
    net.params(p).inputSize = net.meta.inputSize;
    net.params(p).eigvec = [];
    net.params(p).opts = {'cuDNN'};
end
    
    
    


