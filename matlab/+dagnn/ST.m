classdef ST < dagnn.ElementWise % soft thresholding
  properties
    %useShortCircuit = true
    %opts = {'cuDNN'}
    size = [0 0 0 64]
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnST(inputs{1}, params{1});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derParams{1}] = vl_nnST_der(inputs{1}, params{1}, derOutputs{1});
    end

    function set.size(obj, ksize) %???
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function params = initParams(obj)
       params{1} = 0*ones(obj.size(4), 1, 'single');
    end
    
    function obj = ST(varargin)
      obj.load(varargin) ;
      obj.size = obj.size ;
      % obj.inputSize = obj.inputSize;
    end
  end
end
