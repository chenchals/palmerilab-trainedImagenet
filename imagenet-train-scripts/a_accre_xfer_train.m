function [net, info] = a_accre_xfer_train(workDir, dataDir, baseNetToUse)
%A_ACCRE_XFER_TRAIN For transfer learning
% ***** This fx needs to be run from within a shell script on Accre *****
% 1. workDir : location this file along with other cnn_train files
%
% 2. dataDir: ramdisk location where imagenet12 data is present. Usually
% created in the shell script that calls this matlab function
%
% 3. baseNetToUse: Use this deployed net as starting point for further training. 
% The directory will become the result directory where training snapshots are stored
%     example: Copy model to use to 
%          /scratch/subravcr/trainedImagenet/myModels/xferLearning/net-epoch-25/baseModel-net-epoch-25.mat
%
try
    fprintf('*******************************************************\n');
    fprintf('Doing xfer-training or Fine tuning.  %s\n',workDir);
    fprintf('Using Base net from.  %s\n',baseNetToUse);
    fprintf('*******************************************************\n');
    if ~exist(baseNetToUse,'file')
        error('Base Net file [%s] does not exist',baseNetToUse); 
    end
    resultDir = fileparts(baseNetToUse);
    fprintf('Work dir is %s\n',workDir);
    fprintf('Data dir is %s\n',dataDir);
    fprintf('Result dir is %s\n',resultDir);
    homeDir=getenv('HOME');
    % redundant
    if ~exist(resultDir,'dir')
        error('Result dir [%s] does not exist',resultDir); 
    end
    if ~exist(dataDir,'dir')
        error('Data dir [%s] does not exist',dataDir); 
    end
    if ~exist(workDir,'dir')
        error('Work dir [%s] does not exist',workDir); 
    end
    
    MATCONVNET_DIR = fullfile(homeDir,'apps/matconvnet');
    IMAGENET_DIR = fullfile(dataDir,'imagenet12');
    RESULT_DIR = resultDir;
    fprintf('Imagenet12 dir is %s\n',IMAGENET_DIR);
    cmd = ['ls -l ' IMAGENET_DIR];
    system(cmd,'-echo')
    fprintf('Setting up paths for MatConvNet\n');
    run(fullfile(MATCONVNET_DIR, 'matlab', 'vl_setupnn.m')) ;
    fprintf('Current working dir [%s]\n',pwd);
    % -------------------------------------------------------------------------
    %                   Setup options for training
    % -------------------------------------------------------------------------
    % Using code from cnn_imagenet.m
    % vl_setupnn.m <= already run
    opts.dataDir = IMAGENET_DIR;
    opts.modelType = 'vgg-vd-16' ; % DO NOT Use resnet
    opts.network = [] ;
    opts.networkType = 'simplenn' ;
    opts.batchNormalization = true ;% true creates52 layers bn_1
    opts.weightInitMethod = 'gaussian' ;
    [opts] = vl_argparse(opts,{}) ;
    
    opts.expDir = RESULT_DIR;
    [opts] = vl_argparse(opts, {}) ;
    
    opts.numFetchThreads = 12 ;
    opts.lite = false ;
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    %opts.train = struct() ;
    opts.train.gpus = [1 2 3 4] ;
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
    opts.batchSize = 4096; %8192 (for catlab); % from getImageStats.m 256s
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
    net.meta.trainOpts.batchSize = 96; %256; %128;
    net.meta.trainOpts.prefetch = true;
    net.meta.trainOpts.numEpochs = 100;
    % -------------------------------------------------------------------------
    %              TEST3-Prepare model for Transfer Learning
    % -------------------------------------------------------------------------
    startingNet = load(baseNetToUse);
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
    fprintf('Current working dir [%s]\n',pwd);
    which('vl_imreadjpeg', '-all')
    
    opts.prefetch = false;
    opts.numEpochs = 100;
    switch opts.networkType
        case 'simplenn', trainFn = @cnn_train ;
            %case 'dagnn', trainFn = @cnn_train_dag ;
    end
    [net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
        'expDir', opts.expDir, ...
        net.meta.trainOpts, ...
        opts.train) ;
    fprintf('Done training...time:%10.5f\n',toc);
catch me
    %me
    fprintf('\n%s\n\nExiting...\n',me.getReport);
end

end
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



