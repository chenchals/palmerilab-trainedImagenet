nets = dir('neural_nets'); %name of my neural nets folder

for ii = 4:length(nets)

    net = load(fullfile('neural_nets',nets(ii).name));
    [cosine_distance,euclidean_distance] = images2distruct(net);
    save(strcat('RDM',nets(ii).name),'cosine_distance','euclidean_distance')
    
    imagesc(cosine_distance)
    axis image
    colorbar
    title(strcat('Cosine Distance/',nets(ii).name))
    saveas(gcf,strcat('RDMEUC-',strrep(nets(ii).name,'.mat','.png')))

    imagesc(euclidean_distance)
    axis image
    colorbar
    title(strcat('Euclidean Distance/',nets(ii).name))
    saveas(gcf,strcat('RDMEUC-',strrep(nets(ii).name,'.mat','.png')))


end

function [mat1,mat2] = images2distruct(net)
%%input network, output structure with 2 fields 'cosine_distance' and 'euclidean_distance'
    namecell = 0;
    files1 = dir('stimuli_data');
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

    mat1 = EuclideanDissimilarity(raw_data);
    mat2 = CosineDissimilarity(raw_data);

end          