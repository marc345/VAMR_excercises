function keypoints = selectKeypoints(scores, num, r)
% Selects the num best scores as keypoints and performs non-maximum 
% supression of a (2r + 1)*(2r + 1) box around the current maximum.
    
    % pixel coordinates of the num keypoints with the highest scores
    keypoints = zeros(2, num);
    % paded scores, in order to perform non-maximum supression in a
    % (2r+1)x(2r+1) grid
    temp_scores = padarray(scores, [r r]);
    for i = 1:num
        % get the index of the keypoint with the current highest score
        % (:) unrolls the matrix in column order (each column at a time)
        % into a column vector
        [~, kp] = max(temp_scores(:));
        % get the respective coordinates (row, col) of the keypoint in the image
        [row, col] = ind2sub(size(temp_scores), kp);
        kp = [row;col];
        % save the image coordinates of the current keypoint, we need to
        % subtract r in each direction because the coordinates come from
        % the padded version of the scores matrix (so we have an offset of
        % r in each direction)
        keypoints(:, i) = kp - r;
        % set everything in the box centered at the current keypoint and
        % with radius r in each direction to 0 (actual non-maximum
        % supression)
        % the currently selected keypoint is also set to 0 in order to get
        % the next highest score in the next iteration of the for loop
        temp_scores(kp(1)-r:kp(1)+r, kp(2)-r:kp(2)+r) = ...
            zeros(2*r + 1, 2*r + 1);
    end
end
