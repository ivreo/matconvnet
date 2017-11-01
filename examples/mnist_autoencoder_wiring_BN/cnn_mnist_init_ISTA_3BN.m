function net = cnn_mnist_init_ISTA_3BN()

rng('default');
rng(0) ;

e_filters = {'e_IT1_Filters'};
d_filters = {'d_IT1_Filters'};
sz = {[3 3 1 8]};
pad = {1};
stride = {1};
d_filters = e_filters;

net = dagnn.DagNN();
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts = struct();

name = 'input';

for k = 1 : length(e_filters)
    % DTY
	DTY = ['e_layer_' num2str(k) '_iter_0_DTY'];
	block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'pad', pad{k}, 'stride', stride{k});
	net.addLayer(DTY, block, {name}, {DTY}, e_filters{k});

	% BN3
	BN3 = ['e_layer_' num2str(k) '_iter_0_BN3'];
	block = dagnn.BatchNorm('numChannels',sz{k}(4));
	net.addLayer(BN3, block, {DTY}, {BN3}, {[BN3 '_g'],[BN3 '_b'],[BN3 '_m']});

	% ReLU
	ReLU = ['e_layer_' num2str(k) '_iter_0_ReLU'];
	net.addLayer(ReLU, dagnn.ReLU(), {BN3}, {ReLU}, {});

	global iters;

	for t = 1:iters
		% Gamma
		% BN1
		BN1 = ['e_layer_' num2str(k) '_iter_' num2str(t) '_BN1'];
		block = dagnn.BatchNorm('numChannels',sz{k}(4));
		net.addLayer(BN1, block, {ReLU}, {BN1}, {[BN1 '_g'],[BN1 '_b'],[BN1 '_m']});
		
		% DTDGamma
		DGamma = ['e_layer_' num2str(k) '_iter_' num2str(t) '_D'];
		block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'crop', pad{k}, 'upsample', stride{k});
		net.addLayer(DGamma, block, {ReLU}, {DGamma}, e_filters{k});

		DTDGamma = ['e_layer_' num2str(k) '_iter_' num2str(t) '_DT'];
		block = dagnn.Conv('size', sz{k}, 'hasBias', false, 'stride', stride{k}, 'pad', pad{k});
		net.addLayer(DTDGamma, block, {DGamma}, {DTDGamma}, e_filters{k});

		% BN2
		BN2 = ['e_layer_' num2str(k) '_iter_' num2str(t) '_BN2'];
		block = dagnn.BatchNorm('numChannels',sz{k}(4));
		net.addLayer(BN2, block, {DTDGamma}, {BN2}, {[BN2 '_g'],[BN2 '_b'],[BN2 '_m']});
			
		% DTY    
		% BN3
		BN3 = ['e_layer_' num2str(k) '_iter_' num2str(t) '_BN3'];
		block = dagnn.BatchNorm('numChannels',sz{k}(4));
		net.addLayer(BN3, block, {DTY}, {BN3}, {[BN3 '_g'],[BN3 '_b'],[BN3 '_m']});
		
		% Sum
		Sum = ['e_layer_' num2str(k) '_iter_' num2str(t) '_Sum'];
		block = dagnn.Sum();
		net.addLayer(Sum, block, {BN1, BN2, BN3}, Sum);
		
		% ReLU
		ReLU = ['e_layer_' num2str(k) '_iter_' num2str(t) '_ReLU'];
		net.addLayer(ReLU, dagnn.ReLU(), {Sum}, {ReLU}, {});
	end

    sparsity = ['Sparsity' num2str(k)];
    block = dagnn.Loss('loss', 'sparsity');
    net.addLayer(sparsity, block, {ReLU}, {sparsity});
    
    name = ReLU;
end

for k = length(d_filters) : -1 : 1
    % DGamma
    DGamma = ['d_layer_' num2str(k) '_D'];
    block = dagnn.ConvTranspose('size', sz{k}, 'hasBias', false, 'upsample', stride{k}, 'crop', pad{k});
    net.addLayer(DGamma, block, {name}, {DGamma}, {d_filters{k}});
    
    % ReLU
    if k ~= 1
        % BN
        BN = ['d_layer_' num2str(k) '_BN'];
        block = dagnn.BatchNorm('numChannels',sz{k}(4));
        net.addLayer(BN, block, {DGamma}, {BN}, {[BN '_g'], [BN '_b'], [BN '_m']});
        
        ReLU = ['d_layer_' num2str(k) '_ReLU'];
        net.addLayer(ReLU, dagnn.ReLU(), {BN}, {ReLU}, {});
        
        name = ReLU;
    else
        name = DGamma;
    end
end

block = dagnn.PDist();
net.addLayer('objective', block, {name, 'label'}, 'objective');

block = dagnn.Loss('loss', 'psnr');
net.addLayer('psnr', block, {name,'label'}, 'psnr');

net.initParams();

% 2 bias out of 3 are set to zero.
for i = 1 : length(net.params)
    if strcmp(net.params(i).name(end-4:end),'BN1_b') ...
            || strcmp(net.params(i).name(end-4:end),'BN2_b')
        net.params(i).learningRate = 0;
    end
end

