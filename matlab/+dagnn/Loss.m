classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
    ignoreAverage = false
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      switch obj.loss
          case 'psnr'
              sz = size(inputs{1},1) * size(inputs{1},2) * size(inputs{1},3);
              err = sum(sum(sum((inputs{1} - inputs{2}).^2,1),2),3);
              v = 10*log10(sz ./ err);
              outputs{1} = sum(v);

              if obj.ignoreAverage, return; end;
              n = obj.numAveraged ;
              m = n + size(inputs{1}, 4);
              obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
              obj.numAveraged = m ;

          case 'sparsity'
              outputs{1} = length(find(inputs{1}>0));
           
              if obj.ignoreAverage, return; end;
              n = obj.numAveraged ;
              m = n + numel(inputs{1});
              obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
              obj.numAveraged = m ;

          otherwise
              outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
              obj.accumulateAverage(inputs, outputs);
      end
    end

    function accumulateAverage(obj, inputs, outputs)
      if obj.ignoreAverage, return; end;
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 1) *  size(inputs{1}, 2) * size(inputs{1}, 4);
      obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
