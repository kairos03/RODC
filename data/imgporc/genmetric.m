clear;

%% CC Test
clean_front = double(imread('clean_front.jpg')) / 255;
clean_front = imfilter(clean_front, fspecial('average'));
clean_front = rgb2gray(clean_front);

clean_right = double(imread('clean_right.jpg')) / 255;
clean_right = imfilter(clean_right, fspecial('average'));
clean_right = rgb2gray(clean_right);

fileID = fopen('metric.txt', 'w');

genMetricFromDir(fileID, 'raw_data/cam1/', 'raw_data/cam2/', clean_front, clean_right);

clean_front = zeros(1024, 1280);
clean_right = zeros(1024, 1280);

genMetricFromDir(fileID, 'syntheticdata/contact/front/', 'syntheticdata/contact/right', clean_front, clean_right);
genMetricFromDir(fileID, 'syntheticdata/separate/front/', 'syntheticdata/separate/right', clean_front, clean_right);

fclose(fileID);

function [] = genMetricFromDir(fileID, frontDir, rightDir, clean_front, clean_right)
    filenames = dir(frontDir);

    for i=1:length(filenames)
        if filenames(i).isdir || i < 232
            continue;
        end

        filename = filenames(i).name;

        name_front = sprintf('%s/%s', frontDir, filename);
        name_right = sprintf('%s/%s', rightDir, filename);

        [img_front, mask_front] = getROI(imread(name_front), clean_front);
        [img_right, mask_right] = getROI(imread(name_right), clean_right);

        n_objects_front = 0;
        n_objects_right = 0;

        CC_front = bwconncomp(mask_front);
        CC_right = bwconncomp(mask_right);
        
        avg_size_front = 0;

        for j=1:CC_front.NumObjects
            avg_size_front = avg_size_front + length(CC_front.PixelIdxList{j});
        end
        
        avg_size_front = avg_size_front / CC_front.NumObjects;
            
        for j=1:CC_front.NumObjects
            if length(CC_front.PixelIdxList{j}) > avg_size_front
                n_objects_front = n_objects_front + 1;
            end
        end
        
        avg_size_right = 0;

        for j=1:CC_right.NumObjects
            avg_size_right = avg_size_right + length(CC_right.PixelIdxList{j});
            %disp(length(CC_right.PixelIdxList{j}));
        end
        
        avg_size_right = avg_size_right / CC_right.NumObjects;
        disp(avg_size_right);
        
        for j=1:CC_right.NumObjects
            if length(CC_right.PixelIdxList{j}) > avg_size_right
                n_objects_right = n_objects_right + 1;
            end
        end

        imwrite(imread(name_right), 'img_right.png');
        imwrite(img_right, 'img_right_ROI.png');
        imwrite(mask_right, 'img_right_mask.png');
        subplot(3,2,1), imshow(imread(name_front)), subplot(3,2,2), imshow(imread(name_right));
        subplot(3,2,3), imshow(mask_front), title(n_objects_front), subplot(3,2,4), imshow(mask_right), title(n_objects_right);

        lab_front = rgb2lab(img_front);
        lab_right = rgb2lab(img_right);

        res_front = getRedYellow(lab_front);
        res_right = getRedYellow(lab_right);
        
        imwrite(res_right, 'img_right_redyellow.png');

        res_front = res_front > mean(res_front(:));
        res_right = res_right > mean(res_right(:));
        
        imwrite(res_right, 'img_right_redyellow_ths.png');

        CC_front = bwconncomp(res_front);
        CC_right = bwconncomp(res_right);

        objects_front = zeros(max(2, CC_front.NumObjects), 1);
        objects_right = zeros(max(2, CC_right.NumObjects), 1);

        for j=1:CC_front.NumObjects
            objects_front(j) = length(CC_front.PixelIdxList{j});
        end

        for j=1:CC_right.NumObjects
            objects_right(j) = length(CC_right.PixelIdxList{j});
        end

        objects_front = sort(objects_front);
        objects_right = sort(objects_right);

        title_front = sprintf('%d %d', objects_front(end), objects_front(end-1));
        title_right = sprintf('%d %d', objects_right(end), objects_right(end-1));

        subplot(3,2,5), imshow(res_front), title(title_front);
        subplot(3,2,6), imshow(res_right), title(title_right);

%         fprintf(fileID, '%s %d %d %d %d %d %d\n', ...
%             filename, ...
%             n_objects_front, ...
%             n_objects_right, ...
%             objects_front(end), ...
%             objects_front(end-1), ...
%             objects_right(end), ...
%             objects_right(end-1));
        fprintf('%s %d %d %d %d %d %d\n', ...
            filename, ...
            n_objects_front, ...
            n_objects_right, ...
            objects_front(end), ...
            objects_front(end-1), ...
            objects_right(end), ...
            objects_right(end-1));
        waitforbuttonpress;
        %fprintf('%s\n', filename);
    end



end


function [redyellowimg] = getRedYellow(labimg)
    labimg_red = labimg(:,:,2);
    labimg_yellow = labimg(:,:,3);

    %labimg_red(labimg_red < 0) = 0;
    labimg_yellow(labimg_yellow < 0) = 0;

    redyellowimg = labimg_red + labimg_yellow;
    
    redyellowimg = redyellowimg / max(redyellowimg(:));
    
    redyellowimg(redyellowimg > 1) = 1;
    redyellowimg(redyellowimg < 0) = 0;
end


function [roiimg, mask] = getROI(img, smoothcleangrayimg)
    img = imfilter(double(img) / 255, fspecial('average'));
    gray = rgb2gray(img);

    diffimg = imabsdiff(gray, smoothcleangrayimg);
    diffimg = diffimg .^ 2;
    diffimg = diffimg * 10;

    
    SE = strel('Disk', 1, 4);

    imgdil = imdilate(diffimg, SE);
    imgero = imerode(diffimg, SE);
    
    morphologicalGradient = imsubtract(imgdil, imgero);
    morphologicalGradient = morphologicalGradient / max(morphologicalGradient(:));
    n = sum(sum(morphologicalGradient > mean(morphologicalGradient(:))));
    disp(sum(morphologicalGradient(:)) / n);
    
    disp(mean(morphologicalGradient(:)));
    mask = imbinarize(morphologicalGradient, 0.01); %015
    
    SE = strel('Disk', 3, 4);
    
    mask = imclose(mask, SE);
    mask = imfill(mask, 'holes');
    mask = bwareafilt(mask, [0 99999]);%1);
    
    notMask = ~mask;
    
    mask = mask | bwpropfilt(notMask, 'Area', [-Inf, 5000 - eps(5000)]);
    
    roiimg(:,:,1) = img(:,:,1) .* double(mask);
    roiimg(:,:,2) = img(:,:,2) .* double(mask);
    roiimg(:,:,3) = img(:,:,3) .* double(mask);
end

