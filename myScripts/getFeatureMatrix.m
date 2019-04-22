function [featureMat, predClass] = getFeatureMatrix(net,layerName)
%GETFEATUREMATRIX Return feature
%   Detailed explanation goes here
    global normalizedImages;
    global peiyunFeatures
    % http://www.vlfeat.org/matconvnet/mfiles/simplenn/vl_simplenn/
    % result(1) is input to net, result(i+1) is the result of ith layer
    layerIndex = find(strcmpi(cellfun(@(x) x.name,net.layers,'UniformOutput',false)',layerName))+1;
    
    temp = getFeatureVector(net,normalizedImages{1},layerIndex);
    featureMat = zeros(numel(temp),numel(normalizedImages));
    tic
    for ii = 1:numel(normalizedImages)
        [featureMat(:,ii), predClass(ii)] = getFeatureVector(net,normalizedImages{ii},layerIndex);
    end
    fprintf('%5.5f\n',toc);
end

function [featureVector, predClass] = getFeatureVector(net,im,layerIndex)
   global classes;
   global peiyunFeatures
   result = vl_simplenn(net,im);
   featureVector = squeeze(result(layerIndex).x);
   [prob,index] = sort(squeeze(result(end).x),1,'descend');

   predClass.prob = prob(1);
   predClass.index = index(1);
   predClass.name = classes.name{index(1)};
   predClass.description = classes.description{index(1)};
   
end

