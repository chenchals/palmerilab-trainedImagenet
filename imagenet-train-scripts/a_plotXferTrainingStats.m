baseDir = 'myModels/xferLearning';
baseModelDirs = {
    'imagenet-vgg-verydeep-16'
    'net-epoch-05'
    'net-epoch-10'
    'net-epoch-20'
    'net-epoch-36'
    'net-epoch-52'
    };
modelDirs = strcat(baseDir,filesep,baseModelDirs);
deployedDirs = strcat(modelDirs,filesep,'deployed');
cleanField = @(x) regexprep(x,'[^a-zA-Z0-9]','_');
statsTable = @(x,y) struct2table(x.stats.(y));
outStats = struct();
for i = 1:numel(deployedDirs)
    baseModelField = cleanField(baseModelDirs{i});
    model = dir(fullfile(deployedDirs{i},'net*.mat'));
    % find max epoch no and index
    epochNums = cell2mat(arrayfun(@(x) str2double(cell2mat(regexp(x.name,'\d+','match'))),...
        model,'UniformOutput',false));
    [maxEpochNum, ind] = max(epochNums);
    temp = load(fullfile(model(ind).folder,model(ind).name),'stats');
    t = statsTable(temp,'train'); t.Properties.VariableNames = strcat('train_',t.Properties.VariableNames);
    v = statsTable(temp,'val'); v.Properties.VariableNames = strcat('val_',v.Properties.VariableNames);
    outStats.(baseModelField) = [t v];
    clearvars temp t v
end
figure; 
max_objective = max(structfun(@(x) max([x.train_objective;x.val_objective]),outStats));
max_top1err = max(structfun(@(x) max([x.train_top1err;x.val_top1err]),outStats));
max_top5err = max(structfun(@(x) max([x.train_top5err;x.val_top5err]),outStats));
plotIt(1,'train_objective',outStats,baseModelDirs, max_objective);
plotIt(2,'train_top1err',outStats,baseModelDirs, max_top1err);
plotIt(3,'train_top5err',outStats,baseModelDirs, max_top5err);
plotIt(4,'val_objective',outStats,baseModelDirs, max_objective);
plotIt(5,'val_top1err',outStats,baseModelDirs,max_top1err);
plotIt(6,'val_top5err',outStats,baseModelDirs,max_top5err);

clearvars -except outStats


function [aH] = plotIt(plotIndex,whichValues, stats, legendTxt, maxY)
aH = subplot(2,3,plotIndex);
hold on
structfun(@(x) plot(x.(whichValues),'o-'),stats);
title(upper(regexprep(whichValues,'[^a-zA-Z0-9]',' ')))
xlim([0 30])
ylim([0 max(ylim)])
xlabel('Epoch')
grid('on')
legend(legendTxt);
end