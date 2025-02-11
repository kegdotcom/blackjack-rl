from agent import DQNAgent
from blackjack import Blackjack
# import tensorflow as tf


def main():
    game = Blackjack()
    agent = DQNAgent(game)

    load_from_checkpoint = True
    train = False
    agent.loop(train=train, load_from_checkpoint=load_from_checkpoint)


if __name__ == "__main__":
    main()
