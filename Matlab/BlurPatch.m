% build the blurry image patch dataset from sharp text image patches
%



%clear all;
%close all;
%clc;



NameImage = dir([PathTextPatch, '*.png']);

LengthKMaxMotion = 15;
LengthKMaxDisk = 5
ThetaKMax = 360;
CountPatch = 0;
for iImage = 1:length(NameImage)
    ImageRead = imread([PathTextPatch, NameImage(iImage).name]);

    LengthK = floor(LengthKMaxMotion*rand()) + 1;
    ThetaK = ThetaKMax*rand();
    Kernel = fspecial('motion', LengthK, ThetaK);
    ImageBlur = imfilter(ImageRead, Kernel, 'symmetric');

    LengthK = floor(LengthKMaxDisk*rand()) + 1;
    Kernel = fspecial('disk', LengthK);
    ImageBlur = imfilter(ImageBlur, Kernel, 'symmetric');

    imwrite(uint8(ImageBlur), [PathTextBlur, NameImage(iImage).name]);
end



