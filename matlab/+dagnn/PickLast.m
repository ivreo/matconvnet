classdef PickLast < dagnn.Layer
  properties
      iters
  end
  
  methods
    function outputs = forward(obj, inputs, params) 
 %     outputs{1} = inputs{1}{end};
      outputs{1} = inputs{1}(:,:,:,:,end);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        %%global cur_epoch;
        %%%derInputs{1} = cell(obj.iters(cur_epoch)+1,1);
        %%global iters;
        %%derInputs{1} = cell(iters(cur_epoch)+1,1);
        %%derInputs{1}{end} = derOutputs{1};
        
%        % quick and dirty fix - iv
%        derInputs{1} = cell(obj.iters+1,1);
%        derInputs{1}{end} = derOutputs{1};
        % temp
        derInputs{1} = zeros([size(derOutputs{1}),obj.iters+1], 'like', derOutputs{1});
        derInputs{1}(:,:,:,:,end) = derOutputs{1};
        %derInputs{1} = derOutputs{1} ; 
        % end temp
        
        derParams = [];
    end

    function obj = PickLast(varargin)
      obj.load(varargin) ;
      obj.iters = obj.iters;
    end
  end
end
