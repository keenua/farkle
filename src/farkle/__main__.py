from argparse import ArgumentParser
from typing import *

from farkle.game import play
from farkle.logic.mdp import train
from farkle.visual.adaptive import experiment

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', '--action', help='action to perform: play (default), train or experiment')
    parser.add_argument('-f', '--file', help='file to experiment on, only used by \'experiment\' action')
    args = parser.parse_args()

    action = args.action.lower() if args.action else 'play'

    if action == 'train':
        train()
    elif action == 'experiment':
        file = args.file or 'bug.png'
        experiment(file)
    else:
        play()
