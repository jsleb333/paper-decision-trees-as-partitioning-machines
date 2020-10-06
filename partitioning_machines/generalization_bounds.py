import numpy as np
from copy import copy
from scipy.special import zeta
from partitioning_machines import growth_function_upper_bound
from partitioning_machines import wedderburn_etherington


def shawe_taylor_bound(n_examples,
                       n_errors,
                       growth_function,
                       errors_logprob,
                       complexity_logprob,
                       delta=.05,
                       ):
    """
    Theorem 2.3 of Shawe-Taylor et al. (1997), Structural Risk Minimization over Data-Dependent Hierarchies, with the modification that Sauer's lemma is not used.
    """
    epsilon = 2*n_errors + 4*(np.log(float(growth_function(2*n_examples)))
                              + np.log(4)
                              - np.log(delta)
                              - errors_logprob
                              - complexity_logprob)
    return epsilon / n_examples

def shawe_taylor_bound_pruning_objective_factory(n_features,
                                                 table={},
                                                 loose_pfub=True,
                                                 errors_logprob_prior=None,
                                                 complexity_logprob_prior=None,
                                                 delta=.05):
    if errors_logprob_prior is None:
        r = 1/2
        errors_logprob_prior = lambda n_errors: np.log(1-r) + n_errors*np.log(r)
        
    if complexity_logprob_prior is None:
        s = 2
        complexity_logprob_prior = lambda complexity_idx: -np.log(zeta(s)) - s*np.log(complexity_idx) - np.log(float(wedderburn_etherington(complexity_idx)))
        
    def shawe_taylor_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()
        
        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        errors_logprob = errors_logprob_prior(n_errors)
        complexity_logprob = complexity_logprob_prior(copy_of_tree.n_leaves)
        
        return shawe_taylor_bound(n_examples, n_errors, growth_function, errors_logprob, complexity_logprob, delta)

    return shawe_taylor_bound_pruning_objective


def vapnik_bound(n_examples,
                 n_errors,
                 growth_function,
                 errors_logprob,
                 complexity_logprob,
                 delta=.05,
                 ):
    """
    Equation (4.41) of Vapnik's book (1998) extended to SRM.
    """
    epsilon = 4 / n_examples * (np.log(float(growth_function(2*n_examples)))
                                + np.log(4)
                                - np.log(delta)
                                - errors_logprob
                                - complexity_logprob)
    
    empirical_risk = n_errors / n_examples
    
    return empirical_risk + epsilon/2 * (1 + np.sqrt(1 + 4*empirical_risk/epsilon))

def vapnik_bound_pruning_objective_factory(n_features,
                                           table={},
                                           loose_pfub=True,
                                           errors_logprob_prior=None,
                                           complexity_logprob_prior=None,
                                           delta=.05):
    if errors_logprob_prior is None:
        r = 1/2
        errors_logprob_prior = lambda n_errors: np.log(1-r) + n_errors*np.log(r)
        
    if complexity_logprob_prior is None:
        s = 2
        complexity_logprob_prior = lambda complexity_idx: -np.log(zeta(s)) - s*np.log(complexity_idx) - np.log(float(wedderburn_etherington(complexity_idx)))
        
    def vapnik_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()
        
        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        errors_logprob = errors_logprob_prior(n_errors)
        complexity_logprob = complexity_logprob_prior(copy_of_tree.n_leaves)
        
        return vapnik_bound(n_examples, n_errors, growth_function, errors_logprob, complexity_logprob, delta)
    return vapnik_bound_pruning_objective