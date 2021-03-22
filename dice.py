from probs import PROBS
from typing import *
from combinations import all_combinations, Dice, Info
from collections import defaultdict, deque
from itertools import product 

def get_max_score(dice:Dice) -> Tuple[int, Dice]:
    initial:Info = (0, dice, True)

    queue = deque([initial])
    max_score = 0
    keep_dice = dice

    while queue:
        state = queue.pop()
        for comb in all_combinations:
            res = comb(state)
            score, keep, worked = res
            if worked:
                if score > max_score:
                    max_score = score
                    keep_dice = keep
                queue.append(res)

    return max_score, keep_dice

def score(dice:Dice) -> List[int]:
    initial:Info = (0, dice, True)

    queue = deque([initial])
    max_score = [0] * len(dice)

    while queue:
        state = queue.pop()
        for comb in all_combinations:
            res = comb(state)
            score, keep, worked = res
            if worked:
                kept = len(keep)
                if score > max_score[kept]:
                    max_score[kept] = score
                queue.append(res)

    return max_score

def encode(moves: List[int]) -> int:
    res = 0

    for index, i in enumerate(moves):
        res += (i // 50) << index * 8

    return res

def decode(hash: int) -> List[int]:
    result = []
    for _ in range(6):
        result.append((hash & 255) * 50)
        hash = hash >> 8
    return result

def get_probabilities(dice: int) -> Mapping[int, int]:
    result = defaultdict(int)

    temp = [list(range(1, 7)) for _ in range(dice)] 

    for dice in product(*temp):
        cscore = score(list(dice))
        hash = encode(cscore)
        result[hash] += 1

    factor=1.0/sum([v for _,v in result.items()])
    for k in result:
        result[k] *= factor
    return result

def expected_value(dice: int, depth: int) -> Tuple[float, float]:
    bust_chance = 0
    ev_if_not_busted = 0

    for hash, p in PROBS[dice - 1].items():
        if hash == 0:
            bust_chance += p
            continue

        state = decode(hash)
        max_score = max(state)

        if max_score > 0:
            stop_ev = max_score * p
            max_ev = stop_ev

            if depth > 0:
                for i, s in enumerate(state):
                    if s > 0:
                        reroll = 6 if i == 0 else i
                        c_bust, continue_ev = expected_value(reroll, depth - 1)
                        max_ev = max(continue_ev, max_ev)
                
            ev_if_not_busted += max_ev
    
    return bust_chance, ev_if_not_busted




#print(expected_value(6, 3, 0, 1))
# print(sum([len(p) for p in PROBS]))

dice:Dice = [1, 1, 1]
scores = score(dice)
print(scores)
# hash = encode(scores) 
# print(hash)
# print(decode(hash))
