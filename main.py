from agent import DQNAgent
from blackjack import Blackjack
# import tensorflow as tf


def main():
    game = Blackjack()
    agent = DQNAgent(game)

    load_from_checkpoint = False
    agent.loop(load_from_checkpoint)


if __name__ == "__main__":
    main()
