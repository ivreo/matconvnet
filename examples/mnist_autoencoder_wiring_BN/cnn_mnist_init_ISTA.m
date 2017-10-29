function net = cnn_mnist_init_ISTA()

rng('default');
rng(0) ;

e_filters = {'e_IT1_Filters'};
d_filters = {'d_IT1_Filters'};
sz = {[3 3 1 8]};
pad = {1};
stride = {1};

net = dagnn.DagNN();

k = 1;

% DTY
DTY = ['e_layer_' num2str(k) '_iter_0_DTY'];
block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'pad', pad{k}, 'stride', stride{k});
net.addLayer(DTY, block, {'input'}, {DTY}, {e_filters{k}});

% BN
BN = ['e_layer_' num2str(k) '_iter_0_BN'];
block = dagnn.BatchNorm('numChannels',sz{k}(4));
net.addLayer(BN, block, {DTY}, {BN}, {[BN '_g'], [BN '_b'], [BN '_m']});

% ReLU
ReLU = ['e_layer_' num2str(k) '_iter_0_ReLU'];
net.addLayer(ReLU, dagnn.ReLU(), {BN}, {ReLU}, {});

global iters;

for t = 1:iters
    % DTDGamma
    DGamma = ['e_layer_' num2str(k) '_iter_' num2str(t) '_D'];
    block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'crop', pad{k}, 'upsample', stride{k});
    net.addLayer(DGamma, block, {ReLU}, {DGamma}, {e_filters{k}});

    DTDGamma = ['e_layer_' num2str(k) '_iter_' num2str(t) '_DT'];
    block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'stride', stride{k}, 'pad', pad{k});
    net.addLayer(DTDGamma, block, {DGamma}, {DTDGamma}, {e_filters{k}});

    % Diff
    Diff = ['e_layer_' num2str(k) '_iter_' num2str(t) '_Diff'];
    block = dagnn.Diff();
    net.addLayer(Diff, block, {DTY, DTDGamma}, Diff);
    
    % BN
    BN = ['e_layer_' num2str(k) '_iter_' num2str(t) '_BN'];
    block = dagnn.BatchNorm('numChannels',sz{k}(4));
    net.addLayer(BN, block, {Diff}, {BN}, {[BN '_g'], [BN '_b'], [BN '_m']});
    
    % Sum
    Sum = ['e_layer_' num2str(k) '_iter_' num2str(t) '_Sum'];
    block = dagnn.Sum();
    net.addLayer(Sum, block, {ReLU, BN}, Sum);
    
    % ReLU
    ReLU = ['e_layer_' num2str(k) '_iter_' num2str(t) '_ReLU'];
    net.addLayer(ReLU, dagnn.ReLU(), {Sum}, {ReLU}, {});
end

sparsity = ['Sparsity' num2str(k)];
block = dagnn.Loss('loss', 'sparsity');
net.addLayer(sparsity, block, {ReLU}, {sparsity});

DGamma = ['d_layer_' num2str(k) '_D'];
block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'upsample', stride{k}, 'crop', pad{k});
net.addLayer(DGamma, block, {ReLU}, {DGamma}, {d_filters{k}});

block = dagnn.PDist();
net.addLayer('objective', block, {DGamma, 'label'}, 'objective');

block = dagnn.Loss('loss', 'psnr');
net.addLayer('psnr', block, {DGamma,'label'}, 'psnr');

net.initParams();

net.meta.inputSize = [28 28 1] ;
global learningRate;
net.meta.trainOpts.learningRate = learningRate ;
global numEpoch;
net.meta.trainOpts.numEpochs = numEpoch ;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

