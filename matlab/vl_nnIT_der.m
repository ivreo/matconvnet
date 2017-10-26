function [Y_der, D_der, lambda_der, mu_der] = vl_nnIT_der(Y, Gamma, Delta, D, lambda, mu, iters, pad, stride, dilate, opts)

Y_der = 0;
D_der = 0;
lambda_der = 0;
mu_der = 0;

mu = repmat(permute(mu,[2,3,1]),size(Gamma,1),size(Gamma,2),size(Gamma,3),size(Gamma,4));

for iter = iters : -1 : 1
    MDelta = vl_nnrelu(Gamma(:,:,:,:,iter+1),Delta);
    mu_MDelta = mu .* MDelta;
    
    % lambda
    [~, ~, MDelta_summed] = vl_nnconv(Y, D, lambda, MDelta, 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    lambda_der = lambda_der - MDelta_summed;
    
    % D
    D_Gamma = vl_nnconv(Y, D, [], Gamma(:,:,:,:,iter), 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    D_mu_MDelta = vl_nnconv(Y, D, lambda, mu_MDelta, 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    [~,D_mu_MDelta_Gamma] = vl_nnconv(D_mu_MDelta, D, [], Gamma(:,:,:,:,iter), 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    res = Y - D_Gamma;
    
    [~,res_mu_MDelta] = vl_nnconv(res, D, [], mu_MDelta, 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    D_der = D_der - D_mu_MDelta_Gamma + res_mu_MDelta;
    
    % mu
    DT_res = vl_nnconv(res, D, [], 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    DT_res_MDelta = DT_res .* MDelta;
    
    mu_der = mu_der + sum(DT_res_MDelta(:));
    
    % Y
    Y_der = Y_der + D_mu_MDelta;
    
    % Delta
    DT_D_mu_MDelta = vl_nnconv(D_mu_MDelta, D, [], 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    Delta = MDelta - DT_D_mu_MDelta;
end

MDelta = vl_nnrelu(Gamma(:,:,:,:,1),Delta);
mu_MDelta = mu .* MDelta;

% D
[~,Y_mu_MDelta] = vl_nnconv(Y, D, [], mu_MDelta, 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});

D_der = D_der + Y_mu_MDelta;

% lambda
[~, ~, MDelta_summed] = vl_nnconv(Y, D, lambda, MDelta, 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});

lambda_der = lambda_der - MDelta_summed;

% Y
D_mu_MDelta = vl_nnconv(Y, D, lambda, mu_MDelta, 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
Y_der = Y_der + D_mu_MDelta;

% mu
DT_Y = vl_nnconv(Y, D, [], 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});

DT_Y_MDelta = DT_Y .* MDelta;

mu_der = mu_der + sum(DT_Y_MDelta(:));

