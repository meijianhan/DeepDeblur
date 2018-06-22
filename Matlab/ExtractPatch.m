% Extract image patches from text images
%



%clear all;
%close all;
%clc;



NameImage = dir([PathTextImage, '*.jpg']);

SizePatch = 64;
%ThresholdVar = 0.001;
ThresholdVar = 65;
CountPatch = 0;
for iImage = 1:length(NameImage)
    ImageRead = imread([PathTextImage, NameImage(iImage).name]);
    ImageRead = imresize(ImageRead, 0.5);

%{
    % Normalize the image values into 0-1
    ImageRead = double(ImageRead);
    NormMax = max(ImageRead(:));
    if(NormMax > 0)
        ImageRead = ((ImageRead./NormMax) - 1).*(-1);
        %ImageRead = ImageRead./NormMax;
    end
%}

    SizeImageRead = size(ImageRead) - SizePatch;
    if(length(SizeImageRead) > 2)
        ImageRead = rgb2gray(ImageRead);
    end
    %ImageRead = im2single(ImageRead);

    iCols = 1;
    CountImagePatch = 0;
    while(iCols <= SizeImageRead(1))
        iRows = 1;
        while(iRows <= SizeImageRead(2))
            TextPatchCrop = ImageRead(iCols:(iCols + SizePatch), iRows:(iRows + SizePatch));

            VarPatch = var(double(TextPatchCrop(:)));
            if(VarPatch > ThresholdVar)
                CountPatch = CountPatch + 1;
                CountImagePatch = CountImagePatch + 1;
                imwrite(uint8(TextPatchCrop), [PathTextPatch, num2str(iImage), '_', num2str(CountImagePatch), '.png']);
            end

            iRows = iRows + SizePatch;
        end
        iCols = iCols + SizePatch;
    end
    disp(['iImage:', num2str(iImage)]);
end


disp(['CountPatch:', num2str(CountPatch)]);















