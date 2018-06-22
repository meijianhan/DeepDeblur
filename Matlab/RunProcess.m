% Processing script
%



clear all;
close all;
%clc;

disp('ExtractPatch');
% Path of the text images
PathTextImage = './images/';
% Path of the text image patches
PathTextPatch = './TextPatch/';
ExtractPatch;


clear all;
close all;
%clc;

disp('BlurPatch');
% Path of the text image patches
PathTextPatch = './TextPatch/';
% Path of the blurry text image patches
PathTextBlur = './TextBlur/';
BlurPatch;

