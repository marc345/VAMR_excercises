function scores = shi_tomasi(img, patch_size)
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    sobel_para = [-1 0 1];
    sobel_orth = [1 2 1];
    
    I_x = conv2(img, sobel_x, 'valid');
    I_y = conv2(img, sobel_y, 'valid');
    I_xx = I_x .^2;
    I_yy = I_y .^2;
    I_xy = I_x .* I_y;
    
    ssd = ones(patch_size, patch_size);
    pr = floor(patch_size / 2);  % patch radius
    
    I_xx_ssd = conv2(I_xx, ssd, 'valid');
    I_yy_ssd = conv2(I_yy, ssd, 'valid');
    I_xy_ssd = conv2(I_xy, ssd, 'valid');
    
    % the eigen values of a matrix M=[a,b;c,d] are 
    % lambda1/2 = (Tr(A)/2 +- ((Tr(A)/2)^2-det(A))^.5
    % The smaller one is the one with the negative sign
    
    tr = I_xx_ssd + I_yy_ssd;
    det = I_xx_ssd .* I_yy_ssd - I_xy_ssd .^ 2;
    
    % we care only about a lower bound for the 2 eigenvalues
    % so we only need to look at the smaller eigenvalue
    scores = 0.5 * tr - ((0.5 * tr) .^ 2 - det) .^ 0.5;
    scores(scores<0) = 0;
    % pad image into original size again
    % it was decreased due to valid convolutions (without padding)
    scores = padarray(scores, [1+pr 1+pr]);
    
    %figure('Color', 'w');
    %subplot(4, 1, 1);
    %imshow(img);
    %subplot(4, 1, 2);
    %imshow(I_x);
    %subplot(4, 1, 3);
    %imshow(I_y);
    %subplot(1, 1, 1);
    %imshow(scores);
end