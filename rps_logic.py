import enum
import random


class PlayChoices(enum.Enum):
    rock = "rock"
    paper = "paper"
    scissors = "scissors"


class RandomPlay:
    @staticmethod
    def play() -> str:
        return random.choice(list(PlayChoices)).value


random_play = RandomPlay()
