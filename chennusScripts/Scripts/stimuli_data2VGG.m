net = load('imagenet-vgg-verydeep-16.mat');
files = dir('/home/chennus/apps/Scripts/stimuli_data');
raw_data = cell(1,500);
	for ii = 3:length(files)
		im = imread(files(ii).name);
		im_ = single(im);
		im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
		im_ = im_ - net.meta.normalization.averageImage;
		res = vl_simplenn(net,im_);
		vec = res(34).x;
		raw_data{ii-2} = vec(:);
	end 