%%EuclideanDistance in Dissimilarity Matrix%%
function distance = EuclideanCompareDiss(mat1,mat2)
    %%FUNCTION EuclideanCompareDiss takes two dissimilarity
    if size(mat1) ~= size(mat2)
        error
    end
  
    vec1 = mat1(:);
    vec2 = mat2(:);

    V = vec1 - vec2;
    distance = sqrt(V' * V);    
end
        