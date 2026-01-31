import numpy as np
import sys
from watchbird.fusion.plda_scorer import PLDAScorer

# Write to file for debugging
with open('plda_debug_output.txt', 'w') as f:
    plda = PLDAScorer()
    plda.load('data/index/plda.npz')

    f.write('Model info:\n')
    f.write(f'  whiten_transform shape: {plda.model.whiten_transform.shape}\n')
    f.write(f'  phi_b: {plda.model.phi_b}\n')
    f.write(f'  phi_w: {plda.model.phi_w}\n')
    f.write(f'  llr_mean: {plda.model.llr_mean}\n')
    f.write(f'  llr_std: {plda.model.llr_std}\n')

    # Compute raw LLR
    ira = plda.model.identity_embeddings['ira'][0]
    mike = plda.model.identity_embeddings['mike'][0]

    raw_same = plda._compute_llr_pair(ira, plda.model.identity_centroids['ira'])
    raw_diff = plda._compute_llr_pair(ira, plda.model.identity_centroids['mike'])

    f.write(f'\nRaw LLR:\n')
    f.write(f'  ira vs ira: {raw_same}\n')
    f.write(f'  ira vs mike: {raw_diff}\n')
    f.write(f'  diff: {raw_same - raw_diff}\n')

    # Test calibrated score() function
    f.write(f'\nCalibrated score() function:\n')
    score_ira_ira = plda.score(ira, 'ira')
    score_ira_mike = plda.score(ira, 'mike')
    score_mike_mike = plda.score(mike, 'mike')
    score_mike_ira = plda.score(mike, 'ira')

    f.write(f'  ira vs ira: {score_ira_ira:.6f}\n')
    f.write(f'  ira vs mike: {score_ira_mike:.6f}\n')
    f.write(f'  mike vs mike: {score_mike_mike:.6f}\n')
    f.write(f'  mike vs ira: {score_mike_ira:.6f}\n')

    # Test score_against_all
    f.write(f'\nscore_against_all():\n')
    best_id, best_llr, margin, scores = plda.score_against_all(ira)
    f.write(f'  ira probe: best={best_id}, llr={best_llr:.6f}, margin={margin:.6f}\n')
    f.write(f'  all scores: {scores}\n')

    best_id, best_llr, margin, scores = plda.score_against_all(mike)
    f.write(f'  mike probe: best={best_id}, llr={best_llr:.6f}, margin={margin:.6f}\n')
    f.write(f'  all scores: {scores}\n')

    # Test unknown
    f.write(f'\nUnknown test:\n')
    np.random.seed(42)
    unknown = np.random.randn(512)
    unknown = unknown / np.linalg.norm(unknown)
    best_id, best_llr, margin, scores = plda.score_against_all(unknown)
    f.write(f'  random probe: best={best_id}, llr={best_llr:.6f}\n')
    f.write(f'  unknown_prob: {plda.compute_unknown_score(unknown):.4f}\n')

print("Done - check plda_debug_output.txt")
