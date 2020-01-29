import numpy as np
import scipy
import pickle

import sys
sys.path.append('../')

from methods import RBFGS, BFGS, Nesterov
from methods.rbfgs import Uniform, Gaussian, CustomDiscrete


def run_rbfgs_experiment(oracle, x_0, mat_distr, sketch_sizes, max_iter, output_file, 
                         random_state=None, **rbfgs_kwargs):
    '''
    Varies method.s_distr.size and runs expeiments
    '''
    
    import copy
    results = dict()
    
    np.random.seed(random_state)
    mat_distr_init_size = mat_distr.size
    for sketch_size in sketch_sizes:
        s_distr = copy.copy(mat_distr)
        if isinstance(mat_distr, (Uniform, Gaussian)):
            mat_distr.size[-1] = sketch_size
        else:
            mat_distr.size = sketch_size
        method = RBFGS(oracle, x_0, mat_distr, **rbfgs_kwargs)
        try:
            method.run(max_iter)
        except ValueError as e:
            continue
        method.oracle = None
        results[sketch_size] = method
    
    with open(output_file, 'wb') as file:
        pickle.dump(results, file)


def run_all_methods(oracle, sketch_sizes, mat, max_iter, output_folder, 
                    sigma_tolerance=1e-10, method_tolerance=1e-16, 
                    stopping_criteria='func_abs', add_text='', 
                    random_state=None):
    
    import os
    os.system('mkdir -p {}'.format(output_folder))
    
    np.random.seed(random_state)
    x_0 = np.random.normal(loc=0., scale=1., size=mat.shape[1])
#     method_tolerance /= max(1., np.linalg.norm(oracle.grad(x_0))**2)

    add_text = '_{}'.format(add_text) if add_text != '' else ''
    def run_rbfgs(mat_distr, line_search_options, distr_name):
        output_file='{}/rbfgs_{}_linesearch={}{}.pkl'.format(
            output_folder, distr_name, line_search_options['method'].lower(), 
            add_text
        )
        run_rbfgs_experiment(
            oracle, x_0, mat_distr=mat_distr, sketch_sizes=sketch_sizes, 
            max_iter=max_iter, tolerance=method_tolerance, 
            stopping_criteria=stopping_criteria, output_file=output_file
        )
    
    U, sigma_diag, Vh = scipy.linalg.svd(mat.T, full_matrices=False)
    nondeg_count = (sigma_diag > sigma_tolerance).sum()
    print('Singular values above tolerance: {}'.format(nondeg_count))
    print()
    
    print('RBFGS-SVD sketch... ', end='')
    mat_distr = CustomDiscrete(U[:, :nondeg_count] / sigma_diag[:nondeg_count])
    run_rbfgs(mat_distr, {'method': 'Wolfe'}, 'svd')
    run_rbfgs(mat_distr, {'method': 'Constant'}, 'svd')
    print('Done')
    
    print('RBFGS-uniform... ', end='')
    mat_distr = Uniform(-1., 1., [mat.shape[1], 1])
    run_rbfgs(mat_distr, {'method': 'Wolfe'}, 'uni')
    run_rbfgs(mat_distr, {'method': 'Constant'}, 'uni')
    print('Done')
    
    print('RBFGS-gauss... ', end='')
    mat_distr = Gaussian(-1., 1., [mat.shape[1], 1])
    run_rbfgs(mat_distr, {'method': 'Wolfe'}, 'gauss')
    run_rbfgs(mat_distr, {'method': 'Constant'}, 'gauss')
    print('Done')

    print('BFGS... ', end='')
    method = BFGS(oracle, x_0, tolerance=method_tolerance, 
                  stopping_criteria=stopping_criteria, 
                  line_search_options={'method': 'Wolfe'})
    method.run(max_iter)
    method.oracle = None
    with open('{}/bfgs_linesearch={}{}.pkl'\
              .format(output_folder, 'wolfe', add_text), 'wb') as file:
        pickle.dump(method, file)

    method = BFGS(oracle, x_0, tolerance=method_tolerance, 
                  stopping_criteria=stopping_criteria, 
                  line_search_options={'method': 'Constant'})
    method.run(max_iter)
    method.oracle = None
    with open('{}/bfgs_linesearch={}{}.pkl'\
              .format(output_folder, 'constant', add_text), 'wb') as file:
        pickle.dump(method, file)
    print('Done')

    print('Nesterov...', end='')
    method = Nesterov(oracle, x_0, stepsize=0.1, momentum=0.99, 
                      stopping_criteria=stopping_criteria, 
                      tolerance=method_tolerance)
    method.run(max_iter)
    method.oracle = None
    with open('{}/nesterov{}.pkl'.format(output_folder, add_text), 'wb') as file:
        pickle.dump(method, file)
    print('Done')
    
    print()
    print('All runs completed.')
