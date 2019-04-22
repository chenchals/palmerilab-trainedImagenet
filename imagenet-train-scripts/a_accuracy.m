%%
% Create ramdisk: on MAC use RAMDiskCreatorApp
% Creates /Volumes/RAMDISK --> 5 GB for test images
% Untar the imagenet12/images/test/ folder 
% tar -xvzmf ~/ImageNet/ramdisk-imagenet12.tar.gz ramdisk/imagenet12/images/test/ --strip-components=3 
%%

run('~/apps/matconvnet/matlab/vl_setupnn');
% test images must be @ /Volumes/RAMDISK/test folder
testFolder='/Volumes/RAMDISK';

imdbFile = '/Users/subravcr/Projects/lab-palmeri/trainedImagenet/myModels/xferLearning/net-epoch-52/imdb.mat';
if ~exist('imdb','var')
   imdb = load(imdbFile);
end
if ~exist('net','var')
   net = load('myModels/xferLearning/imagenet-vgg-verydeep-16/imagenet-vgg-verydeep-16.mat'); 
   net = load('myModels/xferLearning/imagenet-vgg-verydeep-16/imagenet-vgg-verydeep-16.mat'); 
end

test_indices = find(imdb.images.set==2) ; 
test_fnames = strcat(testFolder,filesep,imdb.images.name(test_indices));
%test_data = getImageBatch(imdb.images.name(test_indices), opts);

batchSize=100;

for i = 1:4 %batches
    batchStart = (i-1)*batchSize +1;
    batchEnd = i*batchSize;
    fprintf('doing batch %d - %d...%d\n',i,batchStart,batchEnd);
    fPaths = test_fnames(batchStart:batchEnd);
      data = vl_imreadjpeg(fPaths,'NumThreads',32,...
          'Resize', net.meta.normalization.imageSize(1:2),...
          'SubtractAverage', net.meta.normalization.averageImage,...
          'Pack','verbose');    
    
    ims = data{1};
    res = arrayfun(@(ind) vl_simplenn(net, ims(:,:,:,ind),[],[],'mode', 'test'),...
        1:batchSize,'UniformOutput',false);
    scores = cellfun(@(x) squeeze(gather(x(end).x)),res,'UniformOutput',false);
    [bestScore{i},best{i}] = cellfun(@(x) max(x),scores,'UniformOutput',false);
    
end
    
    resTable = table();
    resTable.bestScore = cell2mat([bestScore{:}])';
    resTable.bestPredClass = cell2mat([best{:}])';
    nSamples = size(resTable,1);
    % actual class labels
    resTable.actualClass=imdb.images.label(test_indices(1:nSamples))';
    resTable.correct = resTable.bestPredClass==resTable.actualClass;
     
    
%    correct = 0;
%    ncorrect = 0;
%     if (best(i,:) == imdb.images.label(test_indices(:,i)))
%         correct = correct + 1;
%     else
%         ncorrect = ncorrect + 1;
%     end