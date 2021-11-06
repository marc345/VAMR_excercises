function scores = harris(img, patch_size, kappa)
       
    % seperable Sobel filters
    sobel_para = [1 2 1];
    sobel_ortho = [-1 0 1];
       
    % image gradients in x and y direction
    Ix = conv2(sobel_para', sobel_ortho, img, 'valid');
    Iy = conv2(sobel_ortho', sobel_para, img, 'valid');
    
    Ixx = Ix .^ 2;
    Iyy = Iy .^ 2;
    Ixy = Ix .* Iy;
    
    % box filter to compute SSD in patch_size x patch_size window
    ssd = ones(patch_size, patch_size);
    pr = floor(patch_size/2) + 1;
    
    sIxx = conv2(Ixx, ssd, 'valid');
    sIyy = conv2(Iyy, ssd, 'valid');
    sIxy = conv2(Ixy, ssd, 'valid');
    
    % trace and determinant of M (for every pixel in the image)
    tr = sIxx + sIyy;
    det = sIxx .* sIyy - sIxy .^ 2;
    
    scores = det - kappa * tr .^ 2;
    scores(scores < 0) = 0;
    scores = padarray(scores, [1+pr 1+pr]);
end