%%Euclidean Distances in Dissimilary Matrix%%
function diss_mat = EuclideanDissimilarity(raw_data)
	%%FUNCTION EuclideanDissimilarity takes a 1xn or nx1 cell and computes the pairwise Euclidean Distances. 
	%%The results of each indice are stored graphically in a 'dissimilarity matrix'
	ll = length(raw_data);
	diss_mat = zeros(ll);

	for ii = 1:ll
		for jj = 1:ll
			G1 = raw_data{ii};
			G2 = raw_data{jj};
			V = G1 - G2;
			diss_mat(ii,jj) = sqrt(V' * V);
		end 
	end 
end 


