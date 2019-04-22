% Assumptions:
% Need to run matconvnet's vl_setup so that vl_simplenn is in path
run('~/apps/matconvnet/matlab/vl_setupnn');
% dir neural_nets is on matlab path
% dir stimuli_data is on matlab path
% get filenames of different neural nets
%modelPath = '~/Projects/lab-palmeri/trainedImagenet/modelZoo/imagenet-vgg-verydeep-*.mat';
modelPath = '~/Projects/lab-palmeri/trainedImagenet/myModels/imagenet12-vgg-vd-16-bnorm-simplenn/net-epoch-20.mat';
novelObjectPath = '~/Projects/lab-palmeri/trainedImagenet/novelObjects';
modelFiles = dir(fullfile(modelPath));
modelNames = {modelFiles.name}';
nNets = numel(modelNames);
outData = zeros(nNets,nNets);
nets = cell(nNets,nNets);

dissimilarityFx = @CosineDissimilarity;

for netIndex = 1:nNets 
    tempNet = load(modelNames{netIndex});
    fprintf('Computing distance matrix for net [%s]:\n',modelNames{netIndex})
    nets{netIndex} = images2dissmat(tempNet,dissimilarityFx);
end


for aa = 1:length(nets)
    for bb = 1:length(nets)
        outData(aa,bb) = CosineCompareDiss(nets{aa},nets{bb});
        fprintf('%d,%d\n',aa,bb);
    end
end


figure(2); imagesc(outData)
axis image
colorbar
title('Dissimilarity of Dissimilarity Matrices')

%% Find fc7 layer
function [outVal] = findLayerIndex(inNet,layerName)
% structure of inNet:
% inNet.layers
% Usage:
% findLayerIndex(inNet,'fc7')
outVal = find(strcmpi(cellfun(@(x) x.name,inNet.layers,'UniformOutput',false)',layerName));
end

function mat = images2dissmat(net,dissimilarityFx)
    stimFiles = dir('novelObjects/*.tif');
    raw_data = cell(1,numel(stimFiles));     
    fc7Index = findLayerIndex(net,'fc7');
    fprintf('  Stimuli...')

    for ii = 1:numel(stimFiles)
        im = imread(stimFiles(ii).name);
        im = single(im);
        im = imresize(im, net.meta.normalization.imageSize(1:2));
        im = im - net.meta.normalization.averageImage;

        res = vl_simplenn(net,im);
        vec = squeeze(res(fc7Index).x);
        raw_data{ii} = vec(:);        
    end
    mat = dissimilarityFx(raw_data);
    fprintf('  done\n')
end          