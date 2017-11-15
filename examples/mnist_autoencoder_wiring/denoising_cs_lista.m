clear all; close all; clc;
addpath('../../matlab');
addpath('../../matlab/mex');
addpath('../../../vlfeat/toolbox/imop');

% net
expDir = './TTTlistaSTsgn';
epoch = findLastCheckpoint(expDir) ;
net = loadState(fullfile(expDir, sprintf('net-epoch-%d.mat', epoch)));

% checking the atoms
id = getParamIndex(net, 'e_IT1_Filters_wd');
wd = net.params(id).value;
figure(100);
vl_imarraysc(reshape(squeeze(wd),7,7,[]),'spacing',1);
colormap gray;
axis equal;  axis image off;

% input
datadir = '/home/reyotero/proj/main/examples/mnist_autoencoder_wiring/test_images/';
list = dir([datadir '*.png']);
H = 300;
for i = 3;% i=1:length(list)
   
    data = [] ;
    tmp = imfinfo([datadir list(i).name] ) ;
    sz = [tmp.Width, tmp.Height] ;
    if min(sz)>H
        
        im = single(imread([datadir list(i).name]))  ;
        im = sum(im(1:H,1:H,:),3)/3 / 255;
    
        % eval
        noiseSD = 0 ; %20/255 ;
        label = im ;
        input = label + noiseSD* randn(size(label),'like', label) ;
        net.conserveMemory = false;
        net.eval({'label', label, 'input', input}) ;
        
        % psnr
        id = getVarIndex(net,'psnr') ;
        psnr = net.vars(id).value ;
        fprintf(1,' %s : %f \n', list(i).name, psnr);
        
        % image denoised
        DGamma = ['d_layer_' num2str(1) '_D'];
        id = getVarIndex(net, DGamma) ;
        figure(2*i) ;
        imagesc(net.vars(id).value) ;
        colormap gray; axis equal;  axis image off;
        
        % image original
        figure(2*i+1) ;
        imagesc(im);
        colormap gray; axis equal;  axis image off;
    
    end
end    









