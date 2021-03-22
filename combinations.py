from typing import *

empty_dice = []
high = set([2,3,4,5,6])
low = set([1,2,3,4,5])


Dice = List[int]
Info = Tuple[int, Dice, bool]

def straight(info: Info) -> Info:
    score, dice, _ = info
    
    if len(set(dice)) == 6:
        return (score + 1500, empty_dice, True)
    
    return score, dice, False

def high_straight(info: Info) -> Info:
    score, dice, _ = info

    if len(set(dice) & high) == 5:
        new_dice = dice.copy()
        for die in high:
            new_dice.remove(die)
        return score + 750, new_dice, True
    
    return score, dice, False

def low_straight(info: Info) -> Info:
    score, dice, _ = info

    if len(set(dice) & low) == 5:
        new_dice = dice.copy()
        for die in low:
            new_dice.remove(die)
        return score + 750, new_dice, True
    
    return score, dice, False

def __of_a_kind(info: Info, count: int) -> Info:
    score, dice, _ = info

    counter = Counter(dice)
    priority = [1, 6, 5, 4, 3, 2]

    for number in priority:
        if counter[number] >= count:
            new_dice = dice.copy()
            for _ in range(count):
                new_dice.remove(number)

            bonus = 1000 if number == 1 else 100 * number
            shift = 2**(count-3)
            return score + bonus * shift, new_dice, True
    
    return score, dice, False


def six_of_a_kind(info: Info) -> Info:
    return __of_a_kind(info, 6)

def five_of_a_kind(info: Info) -> Info:
    return __of_a_kind(info, 5)

def four_of_a_kind(info: Info) -> Info:
    return __of_a_kind(info, 4)

def three_of_a_kind(info: Info) -> Info:
    return __of_a_kind(info, 3)

def two_of_a_kind(info: Info) -> Info:
    score, dice, _ = info

    counter = Counter(dice)
    priority = [1, 5]

    for number in priority:
        if counter[number] >= 2:
            new_dice = dice.copy()
            for _ in range(2):
                new_dice.remove(number)

            bonus = 200 if number == 1 else 100
            return score + bonus, new_dice, True
    
    return score, dice, False

def single(info: Info) -> Info:
    score, dice, _ = info

    priority = [1, 5]

    for number in priority:
        if number in dice:
            new_dice = dice.copy()
            new_dice.remove(number)

            bonus = 100 if number == 1 else 50
            return score + bonus, new_dice, True
    
    return score, dice, False

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