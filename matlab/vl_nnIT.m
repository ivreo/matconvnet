function Gamma = vl_nnIT(Y, D, lambda, mu, iters, pad, stride, dilate, opts)

DTY = vl_nnconv(Y, D, [], 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});

fprintf(1,'\n mu %f \n', mu);
%std(mu)

Gamma = zeros([size(DTY),iters+1],'like',DTY);

lambda = repmat(permute(lambda,[2,3,1]),size(DTY,1),size(DTY,2),1,size(DTY,4));
mu = repmat(permute(mu,[2,3,1]),size(DTY,1),size(DTY,2),size(DTY,3),size(DTY,4)); % scalar mu
%mu = repmat(permute(mu,[2,3,1]),size(DTY,1),size(DTY,2),1,size(DTY,4)); % vector mu

Gamma(:,:,:,:,1) = mu .* DTY;

Gamma(:,:,:,:,1) = Gamma(:,:,:,:,1) - lambda;

Gamma(:,:,:,:,1) = vl_nnrelu(Gamma(:,:,:,:,1));

for iter = 1 : iters
    est = vl_nnconv(Y, D, [], Gamma(:,:,:,:,iter), 'Pad', pad, 'Stride', stride, 'Dilate', dilate, 'NoDerFilters', opts{:}); % values of first arg not used
    
    res = Y - est;
    
    DTres = vl_nnconv(res, D, [], 'Pad', pad, 'Stride', stride, 'Dilate', dilate, opts{:});
    
    Gamma(:,:,:,:,iter+1) = Gamma(:,:,:,:,iter) + mu .* DTres;
    
    Gamma(:,:,:,:,iter+1) = Gamma(:,:,:,:,iter+1) - lambda;
    
    Gamma(:,:,:,:,iter+1) = vl_nnrelu(Gamma(:,:,:,:,iter+1));
end

