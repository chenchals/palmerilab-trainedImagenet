%%Cosine Distances in Dissimilary Matrix%%
function diss_mat = CosineDissimilarity(raw_data)
	%%FUNCTION CosineDissimilarity takes a 1xn or nx1 cell and computes the pairwise Euclidean Distances. 
	%%The results of each indice are stored graphically in a 'Cosine matrix'
	ll = length(raw_data);
	sim_mat = zeros(ll);
	temp = ones(ll);

	for ii = 1:ll
		for jj = 1:ll
			G1 = raw_data{ii};
			G2 = raw_data{jj};
			sim_mat(ii,jj) = dot(G1,G2)/(norm(G1)*norm(G2));
			
		end
	end

	diss_mat = temp - sim_mat;

