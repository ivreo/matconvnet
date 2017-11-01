function net = cnn_conv_mnist_init()

global layerType;
global iters;
global decodingIdx;

rng('default');
rng(0) ;

% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts = struct();

% Architecture
sz = [5,5,1,32];

net.layers = {} ;
switch layerType
    case 'THR'
        f = sqrt(2 / prod(sz(1:3)));
        net.layers{end+1} = struct('type', 'conv', ...
                                   'weights', {{f*randn(sz, 'single'), zeros(sz(4),1,'single')}}, ...
                                   'stride', 1, ...
                                   'pad', 0) ;
        % temp - same initial projection
        [~,eigval] = vl_nnPowerMethod( ...
          [], ...
          net.layers{end}.weights{1}, ...
          net.layers{end}.pad, ...
          net.layers{end}.stride, ...
          1, ...
          net.meta.inputSize, ...
          {'CuDNN'});
        f = 0.3 / sqrt(eigval) * sqrt(1.95 / 2 ) ;
        net.layers{end}.weights{1} = f * net.layers{end}.weights{1} ;                       
        % end temp
        net.layers{end+1} = struct('type', 'relu') ;
    case 'IT'
        % temp - no init projection
        f = sqrt(2 / prod(sz(1:3)));
        % end temp
        net.layers{end+1} = struct('type', 'IT', ...
                                   'weights', {{randn(sz, 'single'), zeros(sz(4),1,'single')}}, ...
                                   'pad', 0, ...
                                   'stride', 1, ...
                                   'dilate', 1, ...
                                   'iters', iters, ...
                                   'stepsize', 1, ...
                                   'eigvec', [], ...
                                   'inputSize', net.meta.inputSize) ;                               
        encodingIdx = length(net.layers);
        [~,eigval] = vl_nnPowerMethod( ...
          [], ...
          net.layers{end}.weights{1}, ...
          net.layers{end}.pad, ...
          net.layers{end}.stride, ...
          net.layers{end}.dilate, ...
          net.layers{end}.inputSize, ...
          {'CuDNN'});
        f = 1 / sqrt(eigval) * sqrt(1.95 / net.layers{end}.stepsize) ;
        net.layers{end}.weights{1} = f * net.layers{end}.weights{1} ;
                               
                               
        net.layers{end+1} = struct('type', 'picklast', ...
                                   'iters', iters, ...
                                   'precious', true);
end
net.layers{end+1} = struct('type', 'convt', ...
                           'weights', {{f*randn(sz, 'single'), []}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
decodingIdx = length(net.layers);

net.layers{end+1} = struct('type', 'pdist', ...
                           'p', 2, ...
                           'noRoot', true, ...
                           'aggregate', true) ;

if strcmp(layerType,'IT')
    net.layers{encodingIdx}.tiedFilters = decodingIdx;
    net.layers{decodingIdx}.tiedFilters = encodingIdx;
end

% Fill in default values
net = vl_simplenn_tidy(net) ;
