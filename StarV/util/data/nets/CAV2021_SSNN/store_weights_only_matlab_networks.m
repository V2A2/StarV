clear; clc;
% net_name = 'm2nist_62iou_dilatedcnn_avgpool'
% net_name = 'm2nist_75iou_transposedcnn_avgpool'
% net_name = 'm2nist_dilated_72iou_24layer'
% net_name = 'mnist_dilated_net_21_later_83iou'
% net_name = 'net_mnist_3_relu'
net_name = 'net_mnist_3_relu_maxpool'
load(append(net_name, '.mat'));
n = length(net.Layers);

network = cell(1, n);

for i=1:n
    L = net.Layers(i);
    name = L.Name;
    name_ = extractBefore(L.Name,'_');
    if strcmp(name, 'imageinput') || strcmp(name, 'input')
        L1 = {'input', L.Mean};
    elseif strcmp(name(1:2), 'BN')
        L1 = {'batchnorm2d', L.Offset, L.Scale, L.TrainedMean, L.TrainedVariance, L.Epsilon};
    elseif strcmp(name_, 'relu') || strcmp(name(1:4), 'relu')
        L1 = {'relu'};
    elseif strcmp(name_, 'conv') || strcmp(name(1:4), 'conv')
        L1 = {'conv2d', L.Weights, L.Bias, L.FilterSize, L.NumChannels, L.NumFilters, L.Stride, L.DilationFactor, L.PaddingSize};
    elseif strcmp(name_, 'transposed-conv')
        L1 = {'convtransposed2d', L.Weights, L.Bias, L.FilterSize, L.NumChannels, L.NumFilters, L.Stride, L.CroppingSize};    
    elseif strcmp(name_, 'maxpool') || strcmp(name(1:6), 'maxpool')
        L1 = {'maxpool2d', L.PoolSize, L.Stride, L.PaddingSize};
    elseif strcmp(name, 'softmax')
        L1 = {'softmax'};
    elseif strcmp(name, 'labels')
        classes = length(L.Classes);
        L1 = {'pixelclassification', classes};
    elseif strcmp(name_, 'avgpool2d') || strcmp(name_, 'pool') || strcmp(name(1:9), 'avgpool2d')
        L1 = {'avgpool2d', L.PoolSize, L.Stride, L.PaddingSize};
    else
        fprintf(name)
    end
    network{i} = L1;
end

save(append(net_name, '_weights.mat'), 'network');