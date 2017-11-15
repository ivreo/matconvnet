function y = vl_nnST(x,b)
%VL_NNRELU CNN Soft thresholding (forward pass)
%   Y = VL_NNST(X, B) applies the soft thresholding operator on X
%   and parameter B
%   if X has a size of [h,w,c,n], then B is expected to be a vector
%   of c values.
%
%
bb = repmat(permute(b,[2,3,1]),size(x,1),size(x,2),1,size(x,4));
y = sign(x).*max(abs(x)-bb, 0) ;

