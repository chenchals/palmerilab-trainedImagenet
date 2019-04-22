
baseDir = 'myModels/xferLearning';
modelDirs = strcat(baseDir,filesep,{
    'imagenet-vgg-verydeep-16'
    'net-epoch-05'
    'net-epoch-10'
    'net-epoch-20'
    'net-epoch-36'
    'net-epoch-52'
    });

for i=1:numel(modelDirs)
    % Deploy all snapshot nets
    modelDir = modelDirs{i};
    copyfile(fullfile(modelDir,'initial-net.mat'),fullfile(modelDir,'net-epoch-0.mat'));
    modelPattern = fullfile(modelDir,'net*.mat');
    outDir = fullfile(modelDir,'deployed');
    mkdir(outDir);
    copyfile(fullfile(modelDir,'base*.mat'),fullfile(outDir,'.'));
    
    % get all snapshot files
    dirStruct = dir(modelPattern);
    inFilenames = strcat({dirStruct.folder}',filesep,{dirStruct.name}');
    outFilenames = strcat(outDir,filesep,{dirStruct.name}');
    
    parfor i = 1:numel(inFilenames)
        inFile = inFilenames{i};
        outFile = outFilenames{i};
        fprintf('loading net [%s] to deploy ...\n',inFile);
        net = load(inFile);
        if isfield(net, 'stats')
            stats = net.stats;
        else
            stats = []
        end
        net = net.net;
        net = cnn_imagenet_deploy(net);
        net.stats = stats;
        saveDeployed(outFile,net);
    end
end

function saveDeployed(fname,var)
fprintf(' saving deployed net to [%s]...',fname);
save(fname,'-struct','var');
fprintf('done\n');

end