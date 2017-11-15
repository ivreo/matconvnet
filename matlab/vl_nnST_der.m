function [dzdx, dzdb] = vl_nnST_der(x,b,dzdy)
%VL_NNRELU CNN Soft thresholding, backward pass.
%
%   [DZDX,DZDB] = VL_NNST(X, B, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
bb = repmat(permute(b,[2,3,1]),size(x,1),size(x,2),1,size(x,4));
dzdx = dzdy .* (abs(x)>bb) ;
dzdb = squeeze(sum(sum(sum(dzdx .* sign(-x),4),2),1));
%a = 0
