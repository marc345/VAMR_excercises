function descriptors = describeKeypoints(img, keypoints, r)
% Returns a (2r+1)^2xN matrix of image patch vectors based on image
% img and a 2xN matrix containing the keypoint coordinates.
% r is the patch "radius".
    
    num_keypoints = size(keypoints, 2);
    descriptors = uint8(zeros((2*r+1)^2, num_keypoints));
    img_pad = padarray(img, [r, r]);
    
    for i = 1:num_keypoints
       % add the padded offset to the kepoint coordiantes in order to
       % extract the intensities around the keypoints from the padded image
       coords = keypoints(:, i) + r;
       row = coords(1);
       col = coords(2);
       % extract the intensities around the square box of size (2r+1)
       % around the keypoint from the (padded) image
       keypoint = img_pad(row-r:row+r, col-r:col+r);
       descriptors(:, i) = keypoint(:);
    end

end