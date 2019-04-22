

epochList = (1:52); % use these epochs
stdVgg16Filename = 'net_imagenet_vgg_verydeep_16';
featureDir = '/Users/subravcr/Projects/lab-palmeri/trainedImagenet/novelObjects/features';
%colorFiles\
[fPaths, fNames] = getFilepaths(fullfile(featureDir,'color','net_epoch*.mat'));
sortIndex = getSortIndex(fPaths);
fPaths = fPaths(sortIndex);
fNames = fNames(sortIndex);
fprintf('Reading features for all training epochs...\n')
% Load features for all novel objects for all epochs
% Result: (4096 x 500 x nEpochs) 
% Matrix dimensions: featureVector, nImages, nEpochs
allFeatures = cellfun(@(x) load(x,'features'),fPaths,'UniformOutput',false);
allFeatures = cellfun(@(x) x.features, allFeatures,'UniformOutput',false);
allFeatures = reshape(cell2mat(allFeatures'),size(allFeatures{1},1),size(allFeatures{1},2),[]);

% Distance between feature vectors for each objcet for successive epochs
% pdist creates all combinations, squareform turns it into a square matrix,
% diag(X,1) ==> extracts the 1st off diagonal --> successive epochs
% (1,2),(2,3)... (n-1,n) epochs
% Result matrix = ((n-1)distances x nImages) = 51 x 500
fprintf('Computing Euclidean distances...\n')
euclidDist2NextEpoch=cell2mat(arrayfun(@(i) diag(squareform(pdist(squeeze(allFeatures(:,i,:))','euclidean')),1),...
                       1:500,'UniformOutput',false));
fprintf('Computing Cosine distances...\n')
cosineDist2NextEpoch=cell2mat(arrayfun(@(i) diag(squareform(pdist(squeeze(allFeatures(:,i,:))','cosine')),1),...
                       1:500,'UniformOutput',false));
% Plot feature similarity
figure
xlab = 'Epoch Number for VGG-16 ImageNet Training';
for p = 1:2
    subplot(1,2,p)
    switch p
        case 1
            distMat = euclidDist2NextEpoch;
            ylab = 'Euclidean distance ($d^k_{i,i+1}$)';
        case 2
            distMat = cosineDist2NextEpoch;
            ylab = 'Cosine distance ($d^k_{i,i+1}$)';
    end
    plotDists(distMat,xlab,ylab);
end

function plotDists(distMat, xlab, ylab)
    % plot into current axes
    hold on
    nSteps = size(distMat,1)+1;
    arrayfun(@(x) plot(2:nSteps,distMat(:,x)),1:size(distMat,2));
    xlabel(xlab,'Interpreter','latex','FontSize',16,'FontWeight','bold');
    ylabel(ylab, 'Interpreter','latex','FontSize',16,'FontWeight','bold');
    xlim([0 nSteps]);
    box('on');
    txt={'     COLOR    '
         '$d^k_{i,i+1}$ = dist($f^k_i,f^k_{i+1}$)'
         'where $f^k_i$ is feature vector for'
         '$k^{th}$ image for $i^{th}$ training epoch'};
   
     text(max(xlim)/4,max(ylim)*0.9,txt,'Interpreter','latex','FontSize',16, 'VerticalAlignment','top');
end


function [ sortIndex ] = getSortIndex(fList)
    pattern = '_(\d+)(.mat$)?';
    [~,sortIndex] = sort(cell2mat(cellfun(@(x) str2double(x{1}),...
        regexp(fList,pattern,'tokens'),'UniformOutput',false)));
    sortIndex = sortIndex(:,1);
end