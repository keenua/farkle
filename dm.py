from dice import best_move, encode, score
import numpy as np
from mdp import get_best_action

MAX_POINTS = 4000
W_FILE = f'{MAX_POINTS}_parallel.pkl.npy'

W = np.load(W_FILE)

while True:
    DICE = list(map(int, input('Dice: ').split()))
    BANKED = int(input('Banked: '))
    OPPONENT_BANKED = int(input('Opponent: '))
    TURN_POINTS = int(input('Turn poitns: '))

    s = score(DICE)
    state_hash = encode(s)

    (reroll, should_roll) = get_best_action(W, s, TURN_POINTS, BANKED, OPPONENT_BANKED)
    move = best_move(DICE, reroll)

    keep = sorted(DICE.copy())
    for d in move:
        keep.remove(d)

    if should_roll:
        print(f'Keep {keep} and roll')
    else:
        print(f'Keep {keep} and bank')