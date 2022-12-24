"""
Roulette selection function definition.
"""

from typing import List, Tuple

import numpy as np


def roulette_selection(item_probs: List[Tuple[any, float]]) -> any:
    """
    Probabilistically selects an item from a list of possible items
    where each has a selection probability.

    :param item_probs: dictionary of {item: selection probability}
    :return: selected item
    """
    if len(item_probs) < 1:
        return None
    if len(item_probs) == 1:
        return item_probs[0][0]

    sorted_items = sorted(item_probs, key=lambda x: x[1])  # sort by probabilities
    threshold = np.random.rand()
    cum_sum = 0
    for item, prob in sorted_items:
        cum_sum += prob
        if threshold < cum_sum:
            return item
    # if we didn't select any, the random threshold was large. Select the last item.
    return sorted_items[-1][0]
