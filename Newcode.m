clc; clear; close all;

%% Load Fingerprint Images (Input and Reference)
img1 = imread('C:\Users\Jalwa Khan\Documents\6th sem project\DIP\dataset_dip\dataset_FVC2000_DB4_B\dataset\real_data\00000.bmp');
img2 = imread('C:\Users\Jalwa Khan\Documents\6th sem project\DIP\dataset_dip\dataset_FVC2000_DB4_B\dataset\real_data\00000.bmp');

gray1 = preprocess_image(img1);
gray2 = preprocess_image(img2);

%% === Image 1 Processing ===
binary1 = imbinarize(gray1, 'adaptive');
cleaned1 = bwmorph(binary1, 'clean');

enh1 = gabor_enhancement(gray1);
bin_enh1 = imbinarize(mat2gray(enh1), 'adaptive');
thin1 = bwmorph(bin_enh1, 'thin', Inf);

[end1, bif1] = minutiae_extract(thin1);

% Bifurcations and All Minutiae Only (No Endpoints)
marked_bif1 = mark_minutiae(thin1, bif1, [0 1 0]); % Green
marked_all1 = mark_minutiae(thin1, end1 | bif1, [1 1 0]); % Yellow

%% === Image 2 Processing ===
binary2 = imbinarize(gray2, 'adaptive');
cleaned2 = bwmorph(binary2, 'clean');

enh2 = gabor_enhancement(gray2);
bin_enh2 = imbinarize(mat2gray(enh2), 'adaptive');
thin2 = bwmorph(bin_enh2, 'thin', Inf);

[end2, bif2] = minutiae_extract(thin2);

marked_all2 = mark_minutiae(thin2, end2 | bif2, [1 1 0]);

%% Matching
[y1, x1] = find(end1);
[y2, x2] = find(end2);

matched = 0;
tolerance = 15;
used = zeros(length(x2),1);

for i = 1:length(x1)
    pt1 = [x1(i), y1(i)];
    dists = sqrt((x2 - pt1(1)).^2 + (y2 - pt1(2)).^2);
    [min_dist, idx] = min(dists);
    if min_dist < tolerance && ~used(idx)
        matched = matched + 1;
        used(idx) = 1;
    end
end

total = max(length(x1), length(x2));
similarity = matched / total;

%% Decision
threshold = 0.3;
if similarity > (1 - threshold)
    result_msg = sprintf('✓ MATCHED\nScore: %.2f', similarity);
else
    result_msg = sprintf('✗ NOT MATCHED\nScore: %.2f', similarity);
end

%% === Visualization ===
figure('Name', 'Fingerprint Processing & Verification', 'Position', [100 100 1200 800]);

% Image 1 Processing Pipeline
subplot(3,4,1); imshow(gray1); title('1. Original Image');
subplot(3,4,2); imshow(cleaned1); title('2. Binarized & Cleaned');
subplot(3,4,3); imshow(enh1, []); title('3. Gabor Enhanced');
subplot(3,4,4); imshow(thin1); title('4. Thinned Image');

% Output 5: Terminations (Red Circles)
subplot(3,4,5);
imshow(thin1); title('5. Terminations (Red Circles)');
[~, terminationList1] = extractTerminations(thin1);
hold on;
plot(terminationList1(:,1), terminationList1(:,2), 'ro');

% Other Image 1 Visuals
subplot(3,4,6); imshow(marked_bif1); title('6. Bifurcations (Green)');
subplot(3,4,7); imshow(marked_all1); title('7. All Minutiae (Yellow)');

% Image 2 Visuals
subplot(3,4,8); imshow(gray2); title('8. Reference Image');
subplot(3,4,9); imshow(thin2); title('9. Ref Thinned Image');

% ✅ Output 10 (UPDATED): Ref Endpoints using same logic as Output 5
subplot(3,4,10);
imshow(thin2); title('10. Ref Endpoints (Red)');
[~, terminationList2] = extractTerminations(thin2);
hold on;
plot(terminationList2(:,1), terminationList2(:,2), 'r.', 'MarkerSize', 12);

% Rest
subplot(3,4,11); imshow(marked_all2); title('11. Ref All Minutiae (Yellow)');

subplot(3,4,12); axis off;
text(0.1, 0.5, result_msg, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'blue');

%% ----------- Helper Functions -----------

function gray = preprocess_image(img)
    if size(img,3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
end

function enhanced = gabor_enhancement(img)
    img = double(img);
    [rows, cols] = size(img);
    enhanced = zeros(rows, cols);
    theta_range = 0:pi/8:pi;
    freq = 0.1;

    for theta = theta_range
        gabor = gabor_fn(4, theta, freq);
        response = conv2(img, gabor, 'same');
        enhanced = max(enhanced, abs(response));
    end
end

function g = gabor_fn(sigma, theta, f)
    [x, y] = meshgrid(-7:7, -7:7);
    x_theta = x * cos(theta) + y * sin(theta);
    y_theta = -x * sin(theta) + y * cos(theta);
    g = exp(-0.5 * (x_theta.^2 + y_theta.^2) / sigma^2) .* cos(2 * pi * f * x_theta);
end

function [endpoints, bifurcations] = minutiae_extract(thinned_img)
    endpoints = false(size(thinned_img));
    bifurcations = false(size(thinned_img));
    [rows, cols] = size(thinned_img);

    cleaned = bwmorph(thinned_img, 'spur');
    cleaned = bwmorph(cleaned, 'clean');

    for i = 2:rows-1
        for j = 2:cols-1
            if cleaned(i, j)
                neighborhood = cleaned(i-1:i+1, j-1:j+1);
                neighborhood(2,2) = 0;
                CN = sum(neighborhood(:));
                neighbors = neighborhood([1:end 1]);
                transitions = sum(neighbors(1:end-1) ~= neighbors(2:end));

                if CN == 1 && transitions == 2
                    endpoints(i,j) = 1;
                elseif CN >= 3 && transitions >= 6
                    bifurcations(i,j) = 1;
                end
            end
        end
    end

    endpoints = bwmorph(endpoints, 'clean');
    bifurcations = bwmorph(bifurcations, 'clean');
end

function marked_img = mark_minutiae(base_img, minutiae_map, color)
    marked_img = repmat(mat2gray(base_img), 1, 1, 3);
    [y, x] = find(minutiae_map);
    for k = 1:length(x)
        for i = -1:1
            for j = -1:1
                if x(k)+j > 0 && x(k)+j <= size(base_img,2) && ...
                   y(k)+i > 0 && y(k)+i <= size(base_img,1)
                    marked_img(y(k)+i, x(k)+j, 1) = color(1);
                    marked_img(y(k)+i, x(k)+j, 2) = color(2);
                    marked_img(y(k)+i, x(k)+j, 3) = color(3);
                end
            end
        end
    end
end

function [terminationMap, terminationList] = extractTerminations(thinnedImage)
    paddedImage = padarray(thinnedImage, [1 1], 0);
    [rows, cols] = size(paddedImage);
    terminationMap = zeros(size(paddedImage));
    terminationList = [];

    for i = 2:rows-1
        for j = 2:cols-1
            if paddedImage(i, j) == 1
                P = [ ...
                    paddedImage(i-1, j), paddedImage(i-1, j+1), ...
                    paddedImage(i, j+1), paddedImage(i+1, j+1), ...
                    paddedImage(i+1, j), paddedImage(i+1, j-1), ...
                    paddedImage(i, j-1), paddedImage(i-1, j-1), ...
                    paddedImage(i-1, j)];
                CN = 0;
                for k = 1:8
                    CN = CN + abs(P(k) - P(k+1));
                end
                CN = CN / 2;

                if CN == 1
                    terminationMap(i, j) = 1;
                    terminationList = [terminationList; j-1, i-1];
                end
            end
        end
    end

    terminationMap = terminationMap(2:end-1, 2:end-1);
end
