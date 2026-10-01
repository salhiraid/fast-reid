import numpy as np

from validate_encoder import map_cmc


def test_map_cmc_known_values():
    # query id 1 (cam 1); gallery: [id1 cam1 (ignored), id2, id1 cam2, id1 cam3]
    sim = np.array([[0.9, 0.8, 0.7, 0.1]])
    mAP, cmc = map_cmc(sim, np.array([1]), np.array([1]), np.array([1, 2, 1, 1]), np.array([1, 1, 2, 3]))
    # after dropping the same-camera item: ranking = [id2, id1, id1] -> AP = (1/2 + 2/3) / 2
    assert np.isclose(mAP, (1 / 2 + 2 / 3) / 2)
    assert cmc[0] == 0 and cmc[1] == 1
