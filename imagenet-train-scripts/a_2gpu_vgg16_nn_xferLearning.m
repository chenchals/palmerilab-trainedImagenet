% Add dir to path
WORK_DIR = '/home/subravcr/Projects/lab-palmeri/trainedImagenet/imagenet-train-scripts';
DATA_DIR = '/home/subravcr/Projects/lab-palmeri/data';

MODEL_DIR = '/home/subravcr/Projects/lab-palmeri/trainedImagenet/myModels/imagenet12-vgg-vd-16-bnorm-simplenn/deployedAll';
MODEL_NET = fullfile(MODEL_DIR,'net-epoch-52.mat');
[~,STARTING_NET,~] = fileparts(MODEL_NET);
XFER_LEARN_DIR = '/home/subravcr/Projects/lab-palmeri/trainedImagenet/myModels/xferLearning';
RESULT_DIR = fullfile(XFER_LEARN_DIR, STARTING_NET);
% Copy starting model used for transfer training
if ~exist(RESULT_DIR,'dir')
    mkdir(RESULT_DIR);
end
copyfile(MODEL_NET, fullfile(RESULT_DIR,['baseModel-' STARTING_NET '.mat']));
startingNet = load(MODEL_NET);nvidis

addpath(WORK_DIR)
% Setup paths for matconvnet
MATCONVNET_DIR = '/home/subravcr/apps/matconvnet';
run(fullfile(MATCONVNET_DIR, 'matlab', 'vl_setupnn.m')) ;
% -------------------------------------------------------------------------
%                   Setup / check Image data - is done after setup
% -------------------------------------------------------------------------
% Setup imagenet Data
% opts.dataDir = fullfile(DATA_DIR, 'imagenet12') ;
% imdb = cnn_imagenet_setup_data(opts);

% -------------------------------------------------------------------------
%                   Setup options for training
% -------------------------------------------------------------------------
% Using code from cnn_imagenet.m
% vl_setupnn.m <= already run
opts.dataDir = fullfile(DATA_DIR, 'imagenet12') ;
opts.modelType = 'vgg-vd-16' ; % DO NOT Use resnet
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;% true creates52 layers bn_1
opts.weightInitMethod = 'gaussian' ;
[opts] = vl_argparse(opts,{}) ;

opts.expDir = fullfile(RESULT_DIR) ;

[opts] = vl_argparse(opts, {}) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
%opts.train = struct() ;
opts.train.gpus = [1 2] ;
opts = vl_argparse(opts,{}) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end
% -------------------------------------------------------------------------
%                     Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file') % = 2
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'images');
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
opts.batchSize = 8192; % from getImageStats.m 256s
% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath,'file')
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  tic
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [256 256], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus, ...
                                                    'batchSize', opts.batchSize) ;
  toc
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;
% -------------------------------------------------------------------------
%                       Prepare Model
% -------------------------------------------------------------------------
% It will NOT be resnet
net = cnn_imagenet_init('model', opts.modelType, ...
    'batchNormalization', opts.batchNormalization, ...
    'weightInitMethod', opts.weightInitMethod, ...
    'networkType', opts.networkType, ...
    'averageImage', averageImage, ... %rgbMean, ...
    'colorDeviation', rgbDeviation, ...
    'classNames', imdb.classes.name, ...
    'classDescriptions', imdb.classes.description) ;
net.meta.trainOpts.batchSize = 128;
net.meta.trainOpts.prefetch = true;
net.meta.trainOpts.numEpochs = 100;
% -------------------------------------------------------------------------
%              TEST1-Did Not WORK-Prepare model for Transfer Learning
% -------------------------------------------------------------------------
% fc7_index = find(strcmpi(cellfun(@(x) x.name,startingNet.layers,'UniformOutput',false)','fc7'));
% net1.layers = {startingNet.layers{1:fc7_index-1}};
% fc7_index = find(strcmpi(cellfun(@(x) x.name,net.layers,'UniformOutput',false)','fc7'));
% net1.layers = [net1.layers {net.layers{fc7_index:end}}];
% net1.meta = net.meta;
% net = net1;
% save(fullfile(RESULT_DIR,'initial-net.mat'),'net');
% -------------------------------------------------------------------------
%              TEST2-Failed to exec-Prepare model for Transfer Learning
% -------------------------------------------------------------------------
% see https://www.cc.gatech.edu/~hays/compuvision/proj6
% also https://github.com/vlfeat/matconvnet/issues/218
% net.meta = startingNet.meta;
% net.layers = startingNet.layers(1:end-2);
% % add new fc8 layer
% net.layers{end+1} = struct(...
%     'type', 'conv',...
%     'name', 'fc8new',...
%     'weights', {{randn(1,1,4096,4096,'single')}, zeros(1,1000,'single')},...
%     'stride', 1,...
%     'pad', 0,...
%     'dilate', 1,...
%     'learningRate', [0.001, 1],...
%     'weightDecay', [0.0005 0],...
%     'opts', {'CudnnWorkspaceLimit' [1262485504]},...
%     'precious', 1);
% % add softmax layer
% net.layers{end+1} = struct(...
%     'type', 'softmax',...
%     'name', 'probNew');

% set backPropDepth to last 2 layers (opts.train is passed as vararg to
% vl_simplenn
% Alternative to block back propagation see:
% https://github.com/vlfeat/matconvnet/issues/84
% --> set learningRate for the layers to zero
% opts.train.backPropDepth = 2;
% -------------------------------------------------------------------------
%              TEST3-Prepare model for Transfer Learning
% -------------------------------------------------------------------------
% see https://www.cc.gatech.edu/~hays/compuvision/proj6
% also https://github.com/vlfeat/matconvnet/issues/218
fc8 = find(strcmpi(cellfun(@(x) x.name,startingNet.layers,'UniformOutput',false)','fc8'));
net1.layers = {startingNet.layers{1:fc8-1}};
fc8 = find(strcmpi(cellfun(@(x) x.name,net.layers,'UniformOutput',false)','fc8'));
net1.layers = [net1.layers {net.layers{fc8:end}}];
net1.meta = net.meta;
net = net1;
save(fullfile(RESULT_DIR,'initial-net.mat'),'net');
% set backPropDepth to last 2 layers (opts.train is passed as vararg to
% vl_simplenn
% Alternative to block back propagation see:
% https://github.com/vlfeat/matconvnet/issues/84
% --> set learningRate for the layers to zero
opts.train.backPropDepth = 2;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
opts.prefetch = true;
opts.numEpochs = 100;
switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  %case 'dagnn', trainFn = @cnn_train_dag ;
end
[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;
end

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end
end
