import numpy as np
import scipy
import pickle

import sys
sys.path.append('../')

from methods import RBFGS, BFGS, Nesterov
from methods.rbfgs import Uniform, Gaussian, CustomDiscrete, ConstantDistribution


def run_rbfgs_experiment(oracle, x_0, mat_distr, sketch_sizes, max_iter, output_file, 
                         random_state=None, overwrite=False, **rbfgs_kwargs):
    '''
    Varies method.s_distr.size and runs experiments
    '''
    
    import copy
    try:
        with open(output_file, 'rb') as file:
            results = pickle.load(file)
    except FileNotFoundError:
        results = dict()
    
    np.random.seed(random_state)
    mat_distr_init_size = mat_distr.size
    for sketch_size in sketch_sizes:
        if sketch_size in results.keys() and not overwrite:
            continue
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
        method.B_0 = None
        method.B_k = None
        method.grad_k = None
        method.id_mat = None
        method.s_distr = None
        method.x_k = None
        method.x_0 = None
        results[sketch_size] = method
    
    mat_distr.size = mat_distr_init_size
    with open(output_file, 'wb') as file:
        pickle.dump(results, file)


def run_nesterov_experiment(oracle, x_0, stepsize_range, momentum_range, max_iter, output_file, overwrite=False, **nesterov_kwargs):
    import itertools
    try:
        with open(output_file, 'rb') as file:
            results = pickle.load(file)
    except FileNotFoundError:
        results = dict()
    
    for stepsize, momentum in itertools.product(stepsize_range, momentum_range):
        if (stepsize, momentum) in results.keys() and not overwrite:
            continue
        method = Nesterov(oracle, x_0, stepsize=stepsize, momentum=momentum, **nesterov_kwargs)
        method.run(max_iter)
        method.x_k = None
        method.y_k = None
        method.grad_k = None
        method.oracle = None
        results[(stepsize, momentum)] = method
    
    with open(output_file, 'wb') as file:
        pickle.dump(results, file)
    

def run_all_methods(oracle, sketch_sizes, mat, max_iter, output_folder, x_0=None, 
                    sigma_tolerance=1e-10, method_tolerance=1e-16, 
                    stopping_criteria='func_abs', add_text='', 
                    random_state=None, linesearch_methods=['Wolfe'], methods=None, 
                    overwrite=False):
    
    import os
    os.system('mkdir -p {}'.format(output_folder))
    
    if methods is None:
        methods = ['svd', 'svd-no-sigma', 'gauss', 'coord', 'bfgs', 'nesterov']
    
    np.random.seed(random_state)
    if x_0 is None:
        x_0 = np.random.normal(loc=0., scale=1., size=mat.shape[1])

    add_text = '_{}'.format(add_text) if add_text != '' else ''
    def run_rbfgs(mat_distr, line_search_options, distr_name, **kwargs):
        output_file='{}/rbfgs_{}_linesearch={}{}.pkl'.format(
            output_folder, distr_name, line_search_options['method'].lower(), 
            add_text
        )
        run_rbfgs_experiment(
            oracle, x_0, mat_distr=mat_distr, sketch_sizes=sketch_sizes, 
            max_iter=max_iter, tolerance=method_tolerance, 
            stopping_criteria=stopping_criteria, output_file=output_file, 
            overwrite=overwrite, **kwargs
        )
    
    if 'svd' in methods or 'svd-no-sigma' in methods:
        try:
            with open('{}/svd.pkl'.format(output_folder), 'rb') as file:
                U, sigma_diag, Vh = pickle.load(file)
                print('Read SVD from {}/svd.pkl'.format(output_folder))
        except FileNotFoundError:
            print('Computing SVD...', end='')
            U, sigma_diag, Vh = scipy.linalg.svd(mat.T, full_matrices=False)
            print('Done')
            with open('{}/svd.pkl'.format(output_folder), 'wb') as file:
                pickle.dump((U, sigma_diag, Vh), file)
        nondeg_count = (sigma_diag > sigma_tolerance).sum()
        print('Singular values above tolerance: {}'.format(nondeg_count))
        print()
    
    if 'svd' in methods:
        print('RBFGS-SVD sketch... ', end='')
        mat_distr = CustomDiscrete(U[:, :nondeg_count], sort_ids=True)
        for ls in linesearch_methods:
            run_rbfgs(mat_distr, {'method': ls}, 'svd')
        print('Done')
    
    if 'svd-no-sigma' in methods:
        print('RBFGS-SVD sketch no sigma... ', end='')
        mat_distr = CustomDiscrete(U, sort_ids=True)
        for ls in linesearch_methods:
            run_rbfgs(mat_distr, {'method': ls}, 'svd-no-sigma')
        print('Done')
    
    if 'gauss' in methods:
        print('RBFGS-gauss... ', end='')
        mat_distr = Gaussian(-1., 1., [mat.shape[1], 1])
        for ls in linesearch_methods:
            run_rbfgs(mat_distr, {'method': ls}, 'gauss')
        print('Done')
    
    if 'coord' in methods:
        print('RBFGS-coord...', end='')
        mat_distr = CustomDiscrete(np.eye(mat.shape[1]), sort_ids=True)
        for ls in linesearch_methods:
            run_rbfgs(mat_distr, {'method': ls}, 'coord')
        print('Done')

    if 'bfgs' in methods:
        print('BFGS... ', end='')
        for ls in linesearch_methods:
            if 'bfgs_linesearch={}{}.pkl'\
                .format(ls.lower(), add_text) not in os.listdir(output_folder):

                method = BFGS(oracle, x_0, tolerance=method_tolerance, 
                              stopping_criteria=stopping_criteria, 
                              line_search_options={'method': 'Wolfe'})
                method.run(max_iter)
                method.oracle = None
                method.H_k = None
                method.x_0 = None
                with open('{}/bfgs_linesearch={}{}.pkl'\
                          .format(output_folder, ls.lower(), add_text), 'wb') as file:
                    pickle.dump(method, file)
        print('Done')

    print()
    print('All runs completed.')
