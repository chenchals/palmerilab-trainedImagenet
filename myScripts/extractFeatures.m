function [varargout] = extractFeatures(varargin)
%EXTRACTFEATURES Summary of this function goes here
% Need to run matconvnet's vl_setup so that vl_simplenn is in path
global classes
global peiyunFeatures
peiyunFeatures = load('~/Projects/lab-palmeri/trainedImagenet/peiyunStuff/features.mat');
run('~/apps/matconvnet/matlab/vl_setupnn');
outFile = '~/Projects/lab-palmeri/trainedImagenet/myModels/imagenet12-vgg-vd-16-bnorm-simplenn/deployed/vgg16-test-features.mat';
modelPathPattern = '~/Projects/lab-palmeri/trainedImagenet/myModels/imagenet12-vgg-vd-16-bnorm-simplenn/deployed/net-imagenet*.mat';
imdbPath = '~/Projects/lab-palmeri/trainedImagenet/myModels/imagenet12-vgg-vd-16-bnorm-simplenn/deployed/imdb.mat';
classes = load(imdbPath,'classes');
classes = classes.classes;
if nargin>=1
    modelPathPattern = varargin{1};
end
novelObjectPathPattern = '~/Projects/lab-palmeri/trainedImagenet/novelObjects/original/*.tif';
if nargin >=2
    novelObjectPathPattern = varargin{2};
end
% DNN model files
[modelFiles, modelNames, cleanModelNames] = getFilepaths(modelPathPattern);
% Novel object files
[novelObjectFiles, novelObjectNames] = getFilepaths(novelObjectPathPattern);
% Images to use for getting feature matrix
% populates global images var for reuse
fprintf('Loading novel object images from %s\n',novelObjectPathPattern);
loadImages(novelObjectFiles);
imageOpts.size = [0, 0]; 
imageOpts.normalizationAverage = [];
outModelFeatures = struct();
save(outFile,'-struct','outModelFeatures'); % empty save for append later
for index = 1:numel(modelNames) 
    cleanDnnName = cleanModelNames{index};
    fprintf('Extracting features for model [%s]:\n',modelNames{index});
    fprintf('Loading model...\n');
    model = load(modelFiles{index});%is a struct(net,state,stats)
    if isfield(model,'net')
        net = model.net;
    else
        net = model;
    end
    clearvars model;
    temp.size = net.meta.normalization.imageSize(1:2);
    temp.normalizationAverage = net.meta.normalization.averageImage;
    
    fprintf('Normalizing input images...\n');
    imageOpts = temp;
    normalizeImages(imageOpts);
    
    outModelFeatures.(cleanDnnName).files = novelObjectNames;
    fprintf('Getting feature matrix\n');
    [outModelFeatures.(cleanDnnName).features, outModelFeatures.(cleanDnnName).predClass] = getFeatureMatrix(net,'fc7');
    % save for each model -append
    save(outFile,'-append','-struct','outModelFeatures');
end

varargout ={outModelFeatures};
end

function normalizeImages(imageOpts)
%NORMALIZEIMAGES normalize images to image options into 
%global normalizedImages variable
    global normalizedImages images;

    normalizedImages = cellfun(@(x)  imresize(x, imageOpts.size),...
        images,'UniformOutput',false);
    avgImage = imageOpts.normalizationAverage;
    if numel(avgImage)~=3 % you may have saved pixel avg and not channel avg
        tmp(1,1,:) = arrayfun(@(x) mean(mean(avgImage(:,:,x))),1:3)';
        avgImage = tmp;
    end

    normalizedImages = cellfun(@(x)  x-avgImage,...
        normalizedImages,'UniformOutput',false);
end

