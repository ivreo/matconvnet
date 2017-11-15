function net = cnn_mnist_init_IT()

rng('default');
rng(0) ;

e_filters = {'e_IT1_Filters'};
d_filters = {'d_IT1_Filters'};
sz = {[7 7 1 64]};

pad = {1};
stride = {1};

e_stepsizes = {'e_IT1_stepsizes'}; 
e_biases = {'e_IT1_biases'}; 
szl = {[1,1,1,64]};
padl = {0};
stridel = {1};

net = dagnn.DagNN();
%net.meta.inputSize = [28 28 1] ;
net.meta.inputSize = [512 512 1] ;
net.meta.trainOpts = struct();

k = 1;

% DTY
DTY = ['e_layer_' num2str(k) '_iter_0_DTY'];
block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'pad', pad{k}, 'stride', stride{k});
net.addLayer(DTY, block, {'input'}, {DTY}, {e_filters{k}});

% Stepsize and bias
lDTYb = ['e_layer_' num2str(k) '_iter_0_lDTYb'];
block = dagnn.Conv('size', szl{k}, 'hasBias', false, 'stride', stridel{k}, 'pad', padl{k});
net.addLayer(lDTYb, block, {DTY}, {lDTYb}, {e_stepsizes{k}});

% ST
ST = ['e_layer_' num2str(k) '_iter_0_ST'];
net.addLayer(ST, dagnn.ST('size', sz{k}), {lDTYb}, {ST}, {e_biases{k}});

global iters;

for t = 1:iters
    
    % DTDGamma
    DGamma = ['e_layer_' num2str(k) '_iter_' num2str(t) '_D'];
    block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'crop', pad{k}, 'upsample', stride{k});
    net.addLayer(DGamma, block, {ST}, {DGamma}, {e_filters{k}});

    DTDGamma = ['e_layer_' num2str(k) '_iter_' num2str(t) '_DT'];
    block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'stride', stride{k}, 'pad', pad{k});
    net.addLayer(DTDGamma, block, {DGamma}, {DTDGamma}, {e_filters{k}});

    % Diff 
    Diff = ['e_layer_' num2str(k) '_iter_' num2str(t) '_Diff'];
    block = dagnn.Diff();
    net.addLayer(Diff, block, {DTY, DTDGamma}, Diff);
    
    % Stepsize
    lDiff = ['e_layer_' num2str(k) '_iter_' num2str(t) '_lDiffb'];
    block = dagnn.Conv('size', szl{k}, 'hasBias', false, 'stride', 1, 'pad', 0 );
    net.addLayer(lDiff, block, {Diff}, {lDiff}, {e_stepsizes{k}});
    
    % Sum
    Sum = ['e_layer_' num2str(k) '_iter_' num2str(t) '_Sum'];
    block = dagnn.Sum();
    net.addLayer(Sum, block, {ST, lDiff}, Sum);
        
    % ST
    ST = ['e_layer_' num2str(k) '_iter_' num2str(t) '_ST'];
    net.addLayer(ST, dagnn.ST('size', sz{1}), {Sum}, {ST}, {e_biases{k}});
end

sparsity = ['Sparsity' num2str(k)];
block = dagnn.Loss('loss', 'sparsity');
net.addLayer(sparsity, block, {ST}, {sparsity});

DGamma = ['d_layer_' num2str(k) '_D'];
block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'upsample', stride{k}, 'crop', pad{k});
net.addLayer(DGamma, block, {ST}, {DGamma}, {d_filters{k}});

block = dagnn.PDist();
net.addLayer('objective', block, {DGamma, 'label'}, 'objective');

block = dagnn.Loss('loss', 'psnr');
net.addLayer('psnr', block, {DGamma,'label'}, 'psnr');

net.initParams();


% Modifying params initialization

% IT stepsizes
ss = 2;
id = getParamIndex(net,e_stepsizes);
net.params(id).value = ss*ones(size(net.params(id).value), 'single');

% biases
id = getParamIndex(net,e_biases);
net.params(id).value = 0*ones(size(net.params(id).value), 'single');

% D encoder
id = getParamIndex(net,e_filters);
D = net.params(id).value;
dilate = 1;
[~,eigval] = vl_nnPowerMethod([], ...
              D, ...
              pad{1}, ...
              stride{1}, ...
              dilate, ...
              net.meta.inputSize, ...
              {'CuDNN'});
factor = 1 / sqrt(eigval) * sqrt(2 / ss);
net.params(id).value = factor*D;

% D decoder
id = getParamIndex(net,d_filters);
net.params(id).value = factor*D;



