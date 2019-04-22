function [net, info] = a_accre_train(workDir, dataDir, resultDir)
%A_ACCRE_TRAIN Summary of this function goes here
%
try
    fprintf('Work dir is %s\n',workDir);
    fprintf('Data dir is %s\n',dataDir);
    fprintf('Result dir is %s\n',resultDir);
    homeDir=getenv('HOME');
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
    
    sfx = opts.modelType ;
    if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
    sfx = [sfx '-' opts.networkType] ;
    opts.expDir = fullfile(RESULT_DIR, ['imagenet12-' sfx]) ;
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



