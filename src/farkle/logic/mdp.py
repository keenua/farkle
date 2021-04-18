from typing import *
from .probs import PROBS, SIX
import numpy as np
import ctypes
import multiprocessing as mp
from multiprocessing import Process

MAX_POINTS = 4000
THREAD_COUNT = 10
STATES = list(set([k for p in PROBS for k, _ in p.items()]))
STATE_INDEXES = {s: i for i, s in enumerate(STATES)}

POINT_STEPS = MAX_POINTS // 50
W_SHAPE = (len(STATES), POINT_STEPS, POINT_STEPS, POINT_STEPS)
LOAD_FROM_FILE = False
SAVE_FILE_NAME = f'{MAX_POINTS}_parallel.pkl'


def to_points(index: int) -> int:
    return index * 50


def to_index(points: int) -> int:
    return points // 50


def decode(hash: int) -> List[int]:
    result = []
    for _ in range(6):
        result.append((hash & 255) * 50)
        hash = hash >> 8
    return result


def get_w(W: np.ndarray, state_hash: int, turn_points: int, banked: int, opponent: int) -> float:
    if banked + turn_points >= MAX_POINTS:
        return 1
    if opponent >= MAX_POINTS:
        return 0

    s = STATE_INDEXES[state_hash]
    return W[s, to_index(turn_points), to_index(banked), to_index(opponent)]


def get_bank_action_w(W: np.ndarray, dp: Dict, banked: int, opponent: int) -> float:
    key = (banked, opponent)

    if key in dp:
        return dp[key]

    opponent_w = sum([get_w(W, s, 0, opponent, banked)
                     * p for s, p in SIX.items()])
    res = 1 - opponent_w
    dp[key] = res
    return res


def get_roll_action_w(W: np.ndarray, dp: Dict, roll: int, turn_points: int, banked: int, opponent: int) -> float:
    key = (roll, turn_points, banked, opponent)

    if key in dp:
        return dp[key]

    roll = 6 if roll == 0 else roll
    probs = PROBS[roll-1]

    res = sum([get_w(W, s, turn_points, banked, opponent)
              * p for s, p in probs.items()])
    dp[key] = res
    return res


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr, dtype=float).reshape(W_SHAPE)


def get_max_w(W: np.ndarray, dp: Dict, state: List[int], turn_points, banked, opponent_banked):
    max_w = 0

    for si, score in enumerate(state):
        if score == 0:
            continue

        bank = get_bank_action_w(
            W, dp, banked + turn_points + score, opponent_banked)
        roll = get_roll_action_w(
            W, dp, si, turn_points + score, banked, opponent_banked)

        max_w = max(bank, roll, max_w)

    return max_w


def get_best_action(W: np.ndarray, state: List[int], turn_points: int, banked: int, opponent_banked) -> Tuple[int, bool]:
    max_w = 0
    keep = 0
    should_roll = False

    dp = dict()

    for si, score in enumerate(state):
        if score == 0:
            continue

        bank = get_bank_action_w(
            W, dp, banked + turn_points + score, opponent_banked)
        if bank > max_w:
            keep = si
            should_roll = False
            max_w = bank

        roll = get_roll_action_w(
            W, dp, si, turn_points + score, banked, opponent_banked)
        if roll > max_w:
            keep = si
            should_roll = True
            max_w = roll

    return (keep, should_roll)


def update(shared_w: mp.Array, convergence: mp.Array, thread_index: int):
    convergence[thread_index] = 0
    dp = dict()

    W = tonumpyarray(shared_w)
    it = np.nditer(W, flags=['multi_index'])

    counter = 0

    for w in it:
        if thread_index == 0 and counter % 10000 == 0:
            print('{:.1%}   '.format(counter / W.size), end='\r')

        counter += 1

        if counter % THREAD_COUNT != thread_index:
            continue

        index = it.multi_index
        (s, t, b, o) = index

        state_hash = STATES[s]
        state = decode(state_hash)
        turn_points = to_points(t)
        banked = to_points(b)
        opponent_banked = to_points(o)

        if state_hash == 0:
            W[index] = get_bank_action_w(W, dp, banked, opponent_banked)
            continue

        max_w = get_max_w(W, dp, state, turn_points, banked, opponent_banked)

        convergence[thread_index] = max(
            abs(w - max_w), convergence[thread_index])

        W[index] = max_w


def update_parallel(shared_w: mp.Array, convergence: mp.Array) -> float:
    threads = []

    for i in range(THREAD_COUNT):
        t = Process(target=update, args=(shared_w, convergence, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def train():
    size = int(np.prod(W_SHAPE))
    shared_w = mp.Array(ctypes.c_double, size, lock=False)
    w = tonumpyarray(shared_w)

    if LOAD_FROM_FILE:
        loaded = np.load(SAVE_FILE_NAME + '.npy')
        print(f'Loaded from file: {loaded.size}')
        np.copyto(w, loaded)
    else:
        np.copyto(w, np.random.rand(*W_SHAPE))

    step = 0

    convergence = mp.Array(ctypes.c_double, [1.0] * THREAD_COUNT, lock=False)
    while max(convergence) > 0:
        update_parallel(shared_w, convergence)
        step += 1
        np.save(SAVE_FILE_NAME, w)
        print('-----------\n')
        print(f'{step}: {max(convergence)}')
