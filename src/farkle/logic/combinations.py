from collections import namedtuple
from typing import *

DiceValues = List[int]
CombState = namedtuple('Info', ['score', 'dice', 'applied'])

def straight(info: CombState) -> CombState:
    if len(set(info.dice)) == 6:
        return CombState(info.score + 1500, [], True)
    
    return CombState(info.score, info.dice, False)

def high_straight(info: CombState) -> CombState:
    high = set([2,3,4,5,6])

    if len(set(info.dice) & high) == 5:
        new_dice = info.dice.copy()
        for die in high:
            new_dice.remove(die)
        return CombState(info.score + 750, new_dice, True)
    
    return CombState(info.score, info.dice, False)

def low_straight(info: CombState) -> CombState:
    low = set([1,2,3,4,5])
    
    if len(set(info.dice) & low) == 5:
        new_dice = info.dice.copy()
        for die in low:
            new_dice.remove(die)
        return CombState(info.score + 750, new_dice, True)
    
    return CombState(info.score, info.dice, False)

def __of_a_kind(info: CombState, count: int) -> CombState:
    counter = Counter(info.dice)
    priority = [1, 6, 5, 4, 3, 2]

    for number in priority:
        if counter[number] >= count:
            new_dice = info.dice.copy()
            for _ in range(count):
                new_dice.remove(number)

            bonus = 1000 if number == 1 else 100 * number
            shift = 2**(count-3)
            new_score = info.score + bonus * shift

            return CombState(new_score, new_dice, True)
    
    return CombState(info.score, info.dice, False)


def six_of_a_kind(info: CombState) -> CombState:
    return __of_a_kind(info, 6)

def five_of_a_kind(info: CombState) -> CombState:
    return __of_a_kind(info, 5)

def four_of_a_kind(info: CombState) -> CombState:
    return __of_a_kind(info, 4)

def three_of_a_kind(info: CombState) -> CombState:
    return __of_a_kind(info, 3)

def two_of_a_kind(info: CombState) -> CombState:
    counter = Counter(info.dice)
    priority = [1, 5]

    for number in priority:
        if counter[number] >= 2:
            new_dice = info.dice.copy()
            for _ in range(2):
                new_dice.remove(number)

            bonus = 200 if number == 1 else 100
            new_score = info.score + bonus
            return CombState(new_score, new_dice, True)
    
    return CombState(info.score, info.dice, False)

def single(info: CombState) -> CombState:
    priority = [1, 5]

    for number in priority:
        if number in info.dice:
            new_dice = info.dice.copy()
            new_dice.remove(number)

            bonus = 100 if number == 1 else 50
            new_score = info.score + bonus
            
            return CombState(new_score, new_dice, True)
    
    return CombState(info.score, info.dice, False)

all_combinations = [
        straight, 
        high_straight, 
        low_straight, 
        six_of_a_kind,
        five_of_a_kind,
        four_of_a_kind,
        three_of_a_kind,
        two_of_a_kind,
        single
    ]