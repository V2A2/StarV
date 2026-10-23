clear; clc;
load("oval21_memory.mat");
Z = ImageZono(lb, ub);
S = Z.toImageStar;
clear Z;

net = importNetworkFromONNX('cifar_deep_kw.onnx');
nnvNet = CNN.parse(net, 'oval21');


fprintf('\nPerform reachability analysis for the network %s...', nnvNet.Name);

memory_usage = [];
reach_time = [];
layers = nnvNet.Layers;

rs = S;
for i=2:nnvNet.numLayers+1
    if strcmp(nnvNet.dis_opt, 'display')
        fprintf('\nPerforming analysis for Layer %d (%s)...', i-1, nnvNet.Layers{i-1}.Name);
    end
    start_time = tic;
    rs_new = nnvNet.Layers{i-1}.reach(rs, nnvNet.reachMethod, nnvNet.reachOption, nnvNet.relaxFactor, nnvNet.dis_opt, nnvNet.lp_solver);
    nnvNet.reachTime(i-1) = toc(start_time);
    rs = rs_new;
    nnvNet.reachSet{i-1} = rs_new;
    
    mem = get_memory(rs_new);
    memory_usage = [memory_usage, mem];
    reach_time = [reach_time, nnvNet.reachTime(i-1)];
    fprintf('\nReachability analysis for Layer %d (%s) is done in %.5f seconds', i-1, nnvNet.Layers{i-1}.Name, nnvNet.reachTime(i-1));
    fprintf('\nThe number of reachable sets at Layer %d (%s) is: %d', i-1, nnvNet.Layers{i-1}.Name, length(rs_new));
    fprintf('\nMemory usage : %d bytes', mem);
end

save(['nnv_oval21_memory_usage.mat'], 'memory_usage', 'reach_time', 'layers');

function memory = get_memory(var)
    V = var.V;
    C = var.C;
    d = var.d;
    pred_lb = var.pred_lb;
    pred_ub = var.pred_ub;
    
    V_size = whos('V');
    C_size = whos('C');
    d_size = whos('d');
    pred_lb_size = whos('pred_lb');
    pred_ub_size = whos('pred_ub');

    memory = V_size.bytes + C_size.bytes + d_size.bytes + pred_lb_size.bytes + pred_ub_size.bytes;
end