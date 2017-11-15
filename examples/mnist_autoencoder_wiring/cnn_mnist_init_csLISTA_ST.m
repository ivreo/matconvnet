function net = cnn_mnist_init_csLISTA()

rng('default');
rng(0) ;

e_filters_wd = {'e_IT1_Filters_wd'};
e_filters_we = {'e_IT1_Filters_we'};
d_filters = {'d_IT1_Filters'};
sz = {[7 7 1 64]};

pad = {1};
stride = {1};

e_stepsizes = {'e_IT1_stepsizes'}; 
e_biases = {'e_IT1_biases'}; 
szl = {[1,1,1,sz{1}(4)]};
padl = {0};
stridel = {1};

net = dagnn.DagNN();
%net.meta.inputSize = [28 28 1] ;
net.meta.inputSize = [512 512 1] ;
net.meta.trainOpts = struct();
k = 1;

% WeY
WeY = ['e_layer_' num2str(k) '_iter_0_DTY'];
block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'pad', pad{k}, 'stride', stride{k});
net.addLayer(WeY, block, {'input'}, {WeY}, {e_filters_we{k}});

% Stepsize and bias
lWeYb = ['e_layer_' num2str(k) '_iter_0_lDTYb'];
block = dagnn.Conv('size', szl{k}, 'hasBias', false, 'pad', padl{k}, 'stride', stridel{k});
net.addLayer(lWeYb, block, {WeY}, {lWeYb}, {e_stepsizes{k}});

% Soft Thresholding
ST = ['e_layer_' num2str(k) '_iter_0_ST'];
net.addLayer(ST, dagnn.ST('size', sz{k}), {lWeYb}, {ST}, {e_biases{k}});

global iters;

for t = 1:iters
    
    e_layer_iter = ['e_layer_' num2str(k) '_iter_' num2str(t)];
    
    % WdGamma
    WdGamma = [e_layer_iter '_D'];
    block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'crop', pad{k}, 'upsample', stride{k});
    net.addLayer(WdGamma, block, {ST}, {WdGamma}, {e_filters_wd{k}});

    % Diff 
    Diff = [e_layer_iter '_Diff'];
    block = dagnn.Diff();
    net.addLayer(Diff, block, {'input', WdGamma}, Diff);
    
    % WeDiff
    WeDiff = [e_layer_iter '_WeDiff'];
    block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'pad', pad{k}, 'stride', stride{k});
    net.addLayer(WeDiff, block, {Diff}, {WeDiff}, {e_filters_we{k}});
    
    % Sum
    Sum = [e_layer_iter '_Sum'];
    block = dagnn.Sum();
    net.addLayer(Sum, block, {ST, WeDiff}, Sum);
        
    % Soft Thresholding
    ST = [e_layer_iter '_ST'];
    net.addLayer(ST, dagnn.ST(), {Sum}, {ST}, {e_biases{k}});
end

sparsity = ['Sparsity' num2str(k)];
block = dagnn.Loss('loss', 'sparsity');
net.addLayer(sparsity, block, {ST}, {sparsity});

WdGamma = ['d_layer_' num2str(k) '_D'];
block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'upsample', stride{k}, 'crop', pad{k});
net.addLayer(WdGamma, block, {ST}, {WdGamma}, {d_filters{k}});

block = dagnn.PDist();
net.addLayer('objective', block, {WdGamma, 'label'}, 'objective');

block = dagnn.Loss('loss', 'psnr');
net.addLayer('psnr', block, {WdGamma,'label'}, 'psnr');

net.initParams();


%% Modifying params initialization (We, Wd, D, mu, lambda)

% IT stepsizes
%ss = 2;
L = 10;

% biases

id = getParamIndex(net,e_biases);
net.params(id).value = 1/L*ones(size(net.params(id).value), 'single');

% We
id = getParamIndex(net,e_filters_wd);
Wd = net.params(id).value;
% dilate = 1;
% [~,eigval] = vl_nnPowerMethod([], ...
%               Wd, ...
%               pad{1}, ...
%               stride{1}, ...
%               dilate, ...
%               net.meta.inputSize, ...
%               {'CuDNN'});
% factor = 1 / sqrt(eigval) * sqrt(2 / ss);
factor = 1;
net.params(id).value = factor*Wd;

% D
id = getParamIndex(net,d_filters);
net.params(id).value = factor*Wd;

% Wd
id = getParamIndex(net,e_filters_we);
net.params(id).value = 1/L*factor*Wd;

