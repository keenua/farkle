from typing import *
from .dice import Dice
import numpy as np
from .mdp import get_best_action

MAX_POINTS = 4000
W_FILE = f'{MAX_POINTS}_parallel.pkl.npy'

max_points_cache = None
w_cache = None

def move(dice_values: List[int], max_points: int, hero_round: int, hero_total: int, opp_total: int) -> Tuple[List[int], bool]:
    global max_points_cache
    global w_cache

    if max_points_cache != max_points or w_cache is None:
        max_points_cache = max_points
        w_cache = np.load(f'{max_points}_parallel.pkl.npy')

    dice = Dice(dice_values)
    score = dice.score()

    (reroll, should_roll) = get_best_action(w_cache, score, hero_round, hero_total, opp_total)
    move = dice.best_move(reroll)

    keep = sorted(dice.copy())
    for d in move:
        keep.remove(d)
    
    return (keep, should_roll)

if __name__ == '__main__':
    W = np.load(W_FILE)

    while True:
        DICE = list(map(int, input('Dice: ').split()))
        BANKED = int(input('Banked: '))
        OPPONENT_BANKED = int(input('Opponent: '))
        TURN_POINTS = int(input('Turn points: '))

        (keep, should_roll) = move(DICE, MAX_POINTS, TURN_POINTS, BANKED, OPPONENT_BANKED)

        if should_roll:
            print(f'Keep {keep} and roll')
        else:
            print(f'Keep {keep} and bank')