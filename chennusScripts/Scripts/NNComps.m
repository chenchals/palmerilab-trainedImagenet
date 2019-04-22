
files = dir('Scripts/neural_nets');
files2 = files;

dat = zeros(length(3:length(files)));

for aa = 3:lengthbb(files)
    net1 = load(files(aa).name);
    mat1 = images2dissmat(net1);
    for bb = 3:length(files)
        net2 = load(files2(bb).name);


        mat2 = images2dissmat(net2);
        fprintf('%d,%d\n',aa-2,bb-2)
        dat(aa-2,bb-2) = EuclideanCompareDiss(mat1,mat2);
    end 
end 

figure(2); imagesc(dat)

function mat = images2dissmat(net)
    namecell = 0;
    files1 = dir('Scripts/stimuli_data');
    raw_data = cell(1,length(files1)-2);

    if iscell(net.layers(1))
        namecell = 1;
    end   


    for jj = 1:length(net.layers)
        if namecell == 1 
            if strcmp(net.layers{jj}.name, 'fc7')
                gg = jj;
            end
        else 
            if strcmp(net.layers(jj).name,'fc7')
                gg = jj;
            end
        end
    end



    for ii = 3:length(files1)
        im = imread(files1(ii).name);
        im = single(im);
        im = imresize(im, net.meta.normalization.imageSize(1:2));
        im = im - net.meta.normalization.averageImage;

        res = vl_simplenn(net,im);
        vec = res(gg).x;
        raw_data{ii-2} = vec(:);
        fprintf('progress: %d/%d\n', ii-2, numel(files1)-2)
    end

    mat = EuclideanDissimilarity(raw_data); 
end          