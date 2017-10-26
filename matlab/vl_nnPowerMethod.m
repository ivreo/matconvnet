function [eigvec,eigval] = vl_nnPowerMethod(init_eigvec,D,pad,stride,dilate,inputSize,opts)

if isempty(init_eigvec)
    iters = 1000;
else
    iters = 10;
end

zr = zeros(inputSize,'like',D);

if isempty(init_eigvec)
    tmp = vl_nnconv(zr, D, [], ...
      'pad', pad, ...
      'stride', stride, ...
      'dilate', dilate, ...
      opts{:}) ;
    eigvec = randn(size(tmp),'like',zr);
else
    eigvec = init_eigvec;
end

for iter = 1 : iters
  tmp = vl_nnconv(zr, D, [], eigvec, ...
      'pad', pad, ...
      'stride', stride, ...
      'dilate', dilate, ...
      'noderfilters', ...
      opts{:}) ; % first arg not needed
  tmp = vl_nnconv(tmp, D, [], ...
      'pad', pad, ...
      'stride', stride, ...
      'dilate', dilate, ...
      opts{:}) ;
  eigvec = tmp / norm(tmp(:));
end

tmp = vl_nnconv(zr, D, [], eigvec, ...
    'pad', pad, ...
    'stride', stride, ...
    'dilate', dilate, ...
    'noderfilters', ...
    opts{:}) ; % first arg not needed
tmp = vl_nnconv(tmp, D, [], ...
    'pad', pad, ...
    'stride', stride, ...
    'dilate', dilate, ...
    opts{:}) ;
eigval = sum(eigvec(:) .* tmp(:)) / norm(eigvec(:))^2;
