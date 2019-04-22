

epochList = [1 2 4 8 14 20 30 40 50 52]; % use these epochs
stdVgg16Filename = 'net_imagenet_vgg_verydeep_16';
featureDir = '/Users/subravcr/Projects/lab-palmeri/trainedImagenet/novelObjects/features';
%colorFiles\
[fPaths, fNames] = getFilepaths(fullfile(featureDir,'color','net_epoch*.mat'));

% ending file numbers to sort on
% sortIndex = getSortIndex(fPaths);
% 
% fPaths = fPaths(sortIndex);
% fNames = fNames(sortIndex);
distMat = struct([]);
for i = 1:numel(epochList)
    epochNum = epochList(i);
    index = find(contains(fPaths,join({'_',num2str(epochNum),'.mat'},'')));
    distMat(i).epochNum = epochNum;
    distMat(i).epochName = fNames{index};
    temp = load(fPaths{i},'features');
    features = temp.features;
    distMat(i).dist = pdist2(features', features','euclidean');
    distMat(i).features = features;
   
end

% Augment distances and show
% gutter/border 
cMap = 'jet';
gutter = 10;
nRows = 2;
nCols = 5;
figure();
for p = 1:numel(distMat)
    h = subplot(nRows,nCols,p);
    %imagesc(distMat(p).dist); 
    imagesc(flipud(distMat(p).dist)); 
end





function [ sortIndex ] = getSortIndex(fList)
    pattern = '_(\d+)\.mat';
    [~,sortIndex] = sort(cell2mat(cellfun(@(x) str2double(x{1}),...
        regexp(fList,pattern,'tokens'),'UniformOutput',false)));
end