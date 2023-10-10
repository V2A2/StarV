"""
Verify sigmoidal neural networks
Author: Sung Woo Choi
Date: 08/28/2023
"""

from StarV.util.load import load_sigmoidal_networks
from StarV.verifier.certifier import certifyRobustness
import pickle
import time

def verify_sigmoidal_networks(dtype, net_size, func, delta=0.98, opt=False, epsilon=0.01, veri_method='BFS', reach_method='approx', lp_solver='gurobi', RF=0.0, DR=0, pool=None, show=False):

    net, image, label = load_sigmoidal_networks(data_type=dtype, net_size=net_size, func=func, opt=opt, delta=delta)

    if show:
        print('epsilon: {}'.format(epsilon))
        print('depth reduction: {}'.format(DR))
        print('relaxation factor: {}'.format(RF))
        print('verification method: {}'.format(veri_method))
        print('reachability method: {}'.format(reach_method))
        print('LP solver: {}'.format(lp_solver))
        print('activation function optimization: {}\n'.format(opt))
        print('activation function delta: {}\n'.format(delta))
        net.info()

    rb, vt = certifyRobustness (
        net = net, 
        input = image, 
        epsilon = epsilon, 
        veriMethod = veri_method, 
        reachMethod = reach_method, 
        lp_solver = lp_solver, 
        pool = pool, 
        RF = relaxation_factor, 
        DR = depth_reduction, 
        show=show
    )

    cur_time = time.strftime('__%Y-%m-%d_%H-%M-%S__', time.localtime(time.time()))
    save_dir = 'sigmoidalNet_eval/' + net.type + cur_time + 'eval.pkl'
    print('Saving the evaluation results to {}'.format(save_dir))

    with open(save_dir, 'wb') as f:  
        pickle.dump([rb, vt, epsilon, RF, DR, opt, delta, net.type], f)

if __name__ == "__main__":

    data_type = 'mnist'
    network_size = 'small'
    sigmoidal_function = 'tanh'
    func_optimization = False

    # relaxation_factor = [0.0, 0.5]
    # relaxation_factor = [0.2, 0.4]
    relaxation_factor = [0.0]
    depth_reduction = [0]
    # depth_reduction = [3]

    delta = 0.99

    epsilon = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
    # epsilon = [0.01, 0.014]
    verification_method = 'BFS'
    reachability_method = 'approx'
    lp_solver = 'gurobi'

    parallel_processing = None
    show = True

    verify_sigmoidal_networks(
        dtype = data_type, 
        net_size = network_size, 
        func = sigmoidal_function,
        opt = func_optimization,
        delta = delta,
        epsilon = epsilon, 
        veri_method = verification_method, 
        reach_method = reachability_method, 
        lp_solver = lp_solver, 
        RF = relaxation_factor, 
        DR = depth_reduction, 
        pool = parallel_processing, 
        show = show)
