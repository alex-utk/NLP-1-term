import artm

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

SEED = 12
N_TOKENS = 10000



def run_hyperparams_search(minimalisation_task):
    num_topics = [10, 11, 12, 13]
    space={
        'num_topics': hp.choice('num_topics', num_topics),
        'phi_tau': hp.uniform('SparsePhi', -1, 1),
        'theta_tau': hp.uniform('SparseTheta', -1, 1),
        'decorrelation_tau': hp.uniform('DecorrelatorPhi', 1e+2, 1e+5),
    }

    trials = Trials()

    best_hyperparams = fmin(
        fn=minimalisation_task,
        space=space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials
    )

    best_hyperparams['num_topics'] = num_topics[best_hyperparams['num_topics']]
    return best_hyperparams


def fit_best_model(best_params, bv, seed):
    model = artm.ARTM(num_topics=best_params['num_topics'], dictionary=bv.dictionary, cache_theta=True, seed=seed)
    model.scores.add(artm.PerplexityScore(name='perplexity_score',
                                         dictionary=bv.dictionary))

    model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
    model.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
    model.scores.add(artm.TopTokensScore(name='top_tokens_score', num_tokens=10000))

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=best_params['SparsePhi'])) # сглаживание/разреживание матрицы Phi
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=best_params['SparseTheta'])) # сглаживание/разреживание матрицы Theta
    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=best_params['DecorrelatorPhi'])) # сделать темы более разнообразными

    model.fit_offline(bv, num_collection_passes=30)
    return model


def prepare_vis_data(model, n_wd):
    phi = model.get_phi()
    theta = model.get_theta().to_numpy().T
    theta = theta / theta.sum(axis=1, keepdims=1)
    data = {'topic_term_dists': phi.to_numpy().T,
            'doc_topic_dists': theta,
            'doc_lengths': n_wd.sum(axis=0).tolist(),
            'vocab': phi.T.columns,
            'term_frequency': n_wd.sum(axis=1).tolist()}
    return data