from typing import *
from .combinations import all_combinations, CombState
from collections import defaultdict, deque
from itertools import product 

class Dice:
    def __init__(self, values: List[int]):
        self.values = values

    def get_max_score(self) -> Tuple[int, 'Dice']:
        initial:CombState = (0, self.values, True)

        queue = deque([initial])
        max_score = 0
        keep_dice = self.values.copy()

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

        return max_score, Dice(keep_dice)

    def score(self) -> List[int]:
        initial:CombState = (0, self.values, True)

        queue = deque([initial])
        max_score = [0] * len(self.values)

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

    def best_move(self, discard: int) -> List[int]:
        initial:CombState = (0, self.values, True)

        n = len(self.values)
        queue = deque([initial])
        max_score = [0] * n
        best_moves = [self.values] * n

        while queue:
            state = queue.pop()
            for comb in all_combinations:
                res = comb(state)
                score, keep, worked = res
                if worked:
                    kept = len(keep)
                    if score > max_score[kept]:
                        max_score[kept] = score
                        best_moves[kept] = keep

                    queue.append(res)

        return best_moves[discard]

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

def get_probabilities(dice_count: int) -> Mapping[int, float]:
    result = defaultdict(int)

    temp = [list(range(1, 7)) for _ in range(dice_count)] 

    for dice_count in product(*temp):
        cscore = Dice(list(dice_count)).score()
        hash = encode(cscore)
        result[hash] += 1

    factor=1.0/sum([v for _,v in result.items()])
    for k in result:
        result[k] *= factor
    return result