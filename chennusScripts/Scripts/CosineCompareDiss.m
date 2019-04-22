%%CosineDistance in Dissimilarity Matrix%%
function distance = CosineCompareDiss(mat1,mat2)
    %%FUNCTION CosineCompareDiss takes two dissimilarity
    if size(mat1) ~= size(mat2)
        error
    end
    
    vec1 = mat1(:);
    vec2 = mat2(:);
    
    tempmat = dot(vec1,vec2)/(norm(vec1)*norm(vec2));
    
    distance = 1 - tempmat;
    
end
      