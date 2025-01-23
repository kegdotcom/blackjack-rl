from agent import DQNAgent
from blackjack import Blackjack
# import tensorflow as tf


def main():
    game = Blackjack()
    agent = DQNAgent(game)

    agent.train()


if __name__ == "__main__":
    main()
