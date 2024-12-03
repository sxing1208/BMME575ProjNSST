clear;
clc;
close all;

% 添加工具箱路径
addpath('/Users/krystalluo/Desktop/24Fall/BME575/project/npy-matlab-master');
addpath('/Users/krystalluo/Desktop/24Fall/BME575/project/NSST-MSMG-PCNN-master/nsst_toolbox');

base_path = '/Users/krystalluo/Desktop/24Fall/BME575/project/';
diseases = {'alz', 'brchgen', 'hyp', 'motor', 'nrm'};
file_suffix_mri = '_mri_train.npy';
file_suffix_pet = '_pet_train.npy';

for d = 1:length(diseases)
    disease = diseases{d};
    mri_path = fullfile(base_path, [disease, file_suffix_mri]);
    pet_path = fullfile(base_path, [disease, file_suffix_pet]);
    
    % 检查文件是否存在
    if exist(mri_path, 'file') ~= 2
        error('MRI file not found: %s', mri_path);
    end
    if exist(pet_path, 'file') ~= 2
        error('PET file not found: %s', pet_path);
    end
    
    % 尝试加载 .npy 文件
    try
        mri_data = readNPY(mri_path);
        pet_data = readNPY(pet_path);
    catch ME
        error('Failed to load .npy file: %s\nError: %s', mri_path, ME.message);
    end
    
    % 打印加载成功信息
    disp(['Loaded MRI and PET data for ', disease, '.']);
    disp(['MRI size: ', mat2str(size(mri_data))]);
    disp(['PET size: ', mat2str(size(pet_data))]);
end

disp('All files loaded successfully.');
clear;
clc;
close all;

% 添加工具箱路径
addpath('/Users/krystalluo/Desktop/24Fall/BME575/project/npy-matlab-master');
addpath('/Users/krystalluo/Desktop/24Fall/BME575/project/NSST-MSMG-PCNN-master/nsst_toolbox');

base_path = '/Users/krystalluo/Desktop/24Fall/BME575/project/';
diseases = {'alz', 'brchgen', 'hyp', 'motor', 'nrm'};
file_suffix_mri = '_mri_train.npy';
file_suffix_pet = '_pet_train.npy';

for d = 1:length(diseases)
    disease = diseases{d};
    mri_path = fullfile(base_path, [disease, file_suffix_mri]);
    pet_path = fullfile(base_path, [disease, file_suffix_pet]);
    
    % 加载 MRI 和 PET 数据
    mri_data = readNPY(mri_path);
    pet_data = readNPY(pet_path);
    
    % 检查 MRI 和 PET 数据维度是否一致
    assert(isequal(size(mri_data), size(pet_data)), 'MRI and PET size mismatch for %s', disease);
    [h, w, c, num_slices] = size(mri_data);
    
    % 初始化存储融合结果
    fused_data = zeros(h, w, c, num_slices);
    
    for i = 1:num_slices
        for ch = 1:c
            % 获取每个通道的 MRI 和 PET 图像
            I1 = im2double(mri_data(:, :, ch, i)); % MRI
            I2 = im2double(pet_data(:, :, ch, i)); % PET
            
            % 调整图像大小为正方形
            l = max(h, w);
            J1 = zeros(l, l);
            J2 = zeros(l, l);
            J1(1:h, 1:w) = I1;
            J2(1:h, 1:w) = I2;

            % NSST 分解参数
            lpfilt = 'maxflat';
            shear_parameters.dcomp = [4 4 3 3];
            shear_parameters.dsize = [32 32 16 16];
            
            % PCNN 参数
            Para.iterTimes = 200;
            Para.link_arrange = 7;
            Para.alpha_L = 0.02;
            Para.alpha_Theta = 3;
            Para.beta = 3;
            Para.vL = 1;
            Para.vTheta = 20;
            
            % NSST 分解
            [dst1, shear_f1] = nsst_dec2(J1, shear_parameters, lpfilt);
            [dst2, shear_f2] = nsst_dec2(J2, shear_parameters, lpfilt);

            % 低频子带融合
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

            % 高频子带融合
            for j = 2:5
                band_count = size(dst1{j}, 3);
                X = zeros(size(dst1{j}));
                for k = 1:band_count
                    X(:, :, k) = fusion_NSST_MSMG_PCNN(dst1{j}(:, :, k), dst2{j}(:, :, k), Para, 3);
                end
                dst{j} = X;
            end

            % NSST 重构
            Ir = nsst_rec2(dst, shear_f1, lpfilt);
            fused_slice = Ir(1:h, 1:w);
            
            % 保存融合结果
            fused_data(:, :, ch, i) = fused_slice;
        end
    end
    
    % 保存每种疾病的融合结果
    save_path = fullfile(base_path, [disease, '_fused.mat']);
    save(save_path, 'fused_data');
    fprintf('Fused data for %s saved at %s\n', disease, save_path);
end

disp('Fusion completed for all diseases.');