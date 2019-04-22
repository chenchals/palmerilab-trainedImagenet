function loadImages(filepaths)
%READIMAGES load all images as single 
%into global images variable
  global images;
  images = cellfun(@(x) single(imread(x)),filepaths,'UniformOutput',false);
  images = cellfun(@(x) imresize(x, [256, 256]),images,'UniformOutput',false);
end