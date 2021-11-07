function matches = matchDescriptors(...
    query_descriptors, database_descriptors, lambda)
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% database descriptor which matches to the i-th query descriptor.
% The descriptor vectors are MxQ and MxD where M is the descriptor
% dimension and Q and D the amount of query and database descriptors
% respectively. matches(i) will be zero if there is no database descriptor
% with an SSD < lambda * min(SSD). No two non-zero elements of matches will
% be equal.
    
    % get neares quer descriptor for each database descriptor
    % also return the index of that dat
    [dists,matches] = pdist2(double(database_descriptors)', double(query_descriptors)', ...
    'euclidean', 'Smallest', 1);
    
    % get the smallest, non-zero distance
    dists = dists(dists ~= 0);
    dmin = min(dists);
    % return only matches that are below a certain distance threshold
    matches(dists >= lambda * dmin) = 0;
    
    % each database descriptor can only be matched once by a query
    unique_matches = zeros(size(matches));
    [~, unique_idxs, ~] = unique(matches, 'stable');
    % return only unique database descriptor matches
    unique_matches(unique_idxs) = matches(unique_idxs);
    matches = unique_matches;

end