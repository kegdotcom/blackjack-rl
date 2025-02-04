import random
import numpy as np


class Blackjack:
    def get_hand_score(hand):
        score = np.sum(hand)
        n_11_aces = np.sum(hand == 11)

        while score > 21 and n_11_aces > 0:
            score -= 10
            n_aces -= 1

        is_soft = n_11_aces > 0

        return score, is_soft

    def __init__(self, n_decks=2):
        self.state_size = 3  # [dealer, player, soft]
        self.action_size = 3  # [stand, hit, double] TODO: add splitting

        self.illegal_reward = -10

        self.n_decks = n_decks
        self.dealer_hand = []
        self.player_hand = []
        self.doubled = False
        self.get_new_deck()
        self.card_idx = -1
        self.deal()

    def get_new_deck(self):
        suit = np.array([11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10])
        self.deck = np.tile(suit, 4*self.n_decks)
        np.random.shuffle(self.deck)

    def get_next_card(self):
        self.card_idx += 1
        return self.deck[self.card_idx]

    def deal(self):
        for _ in range(2):
            self.player_hand.append(self.get_next_card())
            self.dealer_hand.append(self.get_next_card())

    def reset(self):
        while True:
            self.dealer_hand = []
            self.player_hand = []
            self.doubled = False
            self.get_new_deck()
            self.card_idx = -1
            self.deal()
            if not self.does_player_have_blackjack():
                break

    def does_player_have_blackjack(self):
        return len(self.player_hand) == 2 and Blackjack.get_hand_score(self.player_hand)[0] == 21

    def peek_dealer_score(self):
        return self.dealer_hand[0]

    def get_player_score(self):
        return self.player_hand

    def get_dealer_score(self):
        return self.dealer_hand

    def should_dealer_hit(self):
        score, is_soft = self.get_dealer_score()
        return score < 17 or (score == 17 and is_soft)

    def dealer_turn(self):
        while self.should_dealer_hit():
            self.dealer_hand.append(self.get_next_card())

    def get_dealer_score(self):
        return Blackjack.get_hand_score(self.dealer_hand)

    def get_player_score(self):
        return Blackjack.get_hand_score(self.player_hand)

    def get_state(self, final=False):
        # state like: [score dealer is showing (int), player score (int), does the player have a soft hand (binary indicator)]
        player_score, player_soft = self.get_player_score()
        terminal = final or player_score >= 21
        if terminal:
            dealer_state = self.get_dealer_score()[0]
        else:
            dealer_state = self.peek_dealer_score()
        return (dealer_state, player_score, int(player_soft)), terminal

    def stand(self):
        self.dealer_turn()

    def hit(self):
        self.player_hand.append(self.get_next_card())

    def double(self):
        self.player_hand.append(self.get_next_card())
        self.doubled = True
        self.dealer_turn()

    def split(self):
        # TODO
        pass

    def take_action(self, action):
        if action == 0:
            self.stand()
        elif action == 1:
            self.hit()
        elif action == 2:
            self.double()
        elif action == 3:
            # TODO
            self.split()
        else:
            pass
        return self.get_state(action == 0 or action == 2)

    def get_player_reward(self):
        dealer_score, dealer_soft = self.get_dealer_score()
        player_score, player_soft = self.get_player_score()
        if player_score > 21:
            reward = -1
        elif dealer_score > 21:
            reward = 1
        elif dealer_score > player_score:
            # bust or dealer higher score
            reward = -1
        elif player_score > dealer_score:
            # player higher score
            reward = 1
        else:
            # tie
            reward = 0

        if self.doubled:
            return 2*reward
        else:
            return reward

    def is_action_illegal(self, action):
        return (action == 2 or action == 3) and len(self.player_hand) > 2

    def step(self, action):
        # print(self.dealer_hand, self.player_hand)
        state, terminal = self.get_state()
        if terminal or self.is_action_illegal(action):
            return state, action, self.illegal_reward, state, True
        else:
            state_prime, terminal = self.take_action(action)
            reward = self.get_player_reward() if terminal else 0
            # print(self.dealer_hand, self.player_hand)
            return state, action, reward, state_prime, terminal

    def print_game(self, player=0):
        print(f"Dealer is showing: {self.peek_dealer_card()}")
        for i in range(self.n_players):
            print(f"Player {i} has {" and ".join(self.get_hand(i))}")

    def print_results(self, player=0):
        print(f"You had {" and ".join(self.get_hand(player))}")
        print(f"The dealer had {" and ".join(self.get_dealer_hand())}")
        for i in range(self.n_players):
            print(f"Player {i} received a reward of {
                  self.get_player_reward(i)}")
