novelObjectPathPattern = '~/Projects/lab-palmeri/trainedImagenet/novelObjects/original/*.tif';
% Novel object files
[novelObjectFiles, novelObjectNames] = getFilepaths(novelObjectPathPattern);
% Images to use for getting feature matrix
% populates global images var for reuse
fprintf('Loading novel object images from %s\n',novelObjectPathPattern);
readImages(novelObjectFiles);
