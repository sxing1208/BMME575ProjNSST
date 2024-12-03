clear;
clc;
close all;


addpath('/Users/krystalluo/Desktop/24Fall/BME575/project/NSST-MSMG-PCNN-master/nsst_toolbox');
mri_path = '/Users/krystalluo/Desktop/24Fall/BME575/project/NSST-MSMG-PCNN-master/test/mri_output_images/';
pet_path = '/Users/krystalluo/Desktop/24Fall/BME575/project/NSST-MSMG-PCNN-master/test/pet_output_images/';
output_path = '/Users/krystalluo/Desktop/24Fall/BME575/project/NSST-MSMG-PCNN-master/test/fused_output_images/';

if ~exist(output_path, 'dir')
    mkdir(output_path);
end

num_images = 15;
file_prefix = 'image_';
file_suffix = '.png';

lpfilt = 'maxflat';
shear_parameters.dcomp = [4 4 3 3];
shear_parameters.dsize = [32 32 16 16];

Para.iterTimes = 200;
Para.link_arrange = 7;
Para.alpha_L = 0.02;
Para.alpha_Theta = 3;
Para.beta = 3;
Para.vL = 1;
Para.vTheta = 20;

for idx = 1:num_images
    mri_file = fullfile(mri_path, [file_prefix, num2str(idx), file_suffix]);
    pet_file = fullfile(pet_path, [file_prefix, num2str(idx), file_suffix]);
    
    Imr = imread(mri_file);
    Ipe = imread(pet_file);
   
    I1 = im2double(Imr);
    I1 = rgb2gray(I1);
    IpeRGB = im2double(Ipe);
    I3 = rgb2hsv(IpeRGB);
    I2 = I3(:, :, 3);
    
    [m, n] = size(I1);
    l = max(m, n);
    J1 = zeros(l, l);
    J2 = zeros(l, l);
    J1(1:m, 1:n) = I1;
    J2(1:m, 1:n) = I2;
    
    disp(['Decomposing images via NSST: ', mri_file, ' and ', pet_file]);
    [dst1, shear_f1] = nsst_dec2(J1, shear_parameters, lpfilt);
    [dst2, shear_f2] = nsst_dec2(J2, shear_parameters, lpfilt);
    
    disp('Processing Lowpass subband...');
    X1_1 = dst1{1};
    X1_2 = dst2{1};
    mB1 = mean(X1_1(:));
    mB2 = mean(X1_2(:));
    MB1 = median(X1_1(:));
    MB2 = median(X1_2(:));
    G1 = (mB1 + MB1) / 2;
    G2 = (mB2 + MB2) / 2;

    w1 = exp(4 * abs(X1_1 - G1));
    w2 = exp(4 * abs(X1_2 - G2));
    WB1 = w1 ./ (w1 + w2);
    WB2 = w2 ./ (w1 + w2);
    X1 = WB1 .* X1_1 + WB2 .* X1_2;
    dst{1} = X1;

    disp('Processing Bandpass subbands...');
    for j = 2:5
        band_count = size(dst1{j}, 3);
        X = zeros(size(dst1{j}));
        for k = 1:band_count
            X(:, :, k) = fusion_NSST_MSMG_PCNN(dst1{j}(:, :, k), dst2{j}(:, :, k), Para, 3);
        end
        dst{j} = X;
    end
    
    disp('Reconstructing fused image...');
    Ir = nsst_rec2(dst, shear_f1, lpfilt);
    Fi = Ir(1:m, 1:n);
    I3(:, :, 3) = Fi;
    FF = hsv2rgb(I3);
    
    fused_file = fullfile(output_path, ['fused_', file_prefix, num2str(idx), file_suffix]);
    imwrite(FF, fused_file);
    disp(['Fused image saved at: ', fused_file]);
end

disp('All images fused successfully.');