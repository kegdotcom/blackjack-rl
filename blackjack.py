import random


class Blackjack:
    card_values = {
        'A': 11,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'T': 10,
        'J': 10,
        'Q': 10,
        'K': 10,
    }

    def get_hand_score(hand):
        return sum([Blackjack.card_values[card] for card in hand])

    def is_hand_soft(hand):
        raw_score = Blackjack.get_hand_score(hand)
        n_aces = hand.count('A')
        return n_aces > 0 and raw_score - 10*(n_aces-1) <= 21

    def __init__(self, n_decks=2):
        self.state_size = 3  # [dealer, player, soft]
        self.action_size = 3  # [stand, hit, double] TODO: add splitting

        self.illegal_reward = -10

        self.n_decks = n_decks
        self.dealer_hand = []
        self.player_hand = []
        self.doubled = False
        self.deck = self.get_new_deck(n_decks)
        self.shuffle()
        self.deal()

    def get_new_deck(self, n):
        cards = ['A', '2', '3', '4', '5', '6',
                 '7', '8', '9', 'T', 'J', 'Q', 'K']
        deck = []
        for _ in range(n):
            for _ in range(4):
                for card in cards:
                    deck.append(card)
        return deck

    def shuffle(self):
        for i in range(len(self.deck) - 1):
            j = random.randint(i, len(self.deck)-1)
            self.deck[i], self.deck[j] = self.deck[j], self.deck[i]

    def deal(self):
        for _ in range(2):
            self.player_hand.append(self.deck.pop())
            self.dealer_hand.append(self.deck.pop())

    def reset(self):
        while True:
            self.dealer_hand = []
            self.player_hand = []
            self.doubled = False
            self.deck = self.get_new_deck(self.n_decks)
            self.shuffle()
            self.deal()
            if not self.does_player_have_blackjack():
                break

    def does_player_have_blackjack(self):
        return len(self.player_hand) == 2 and Blackjack.get_hand_score(self.player_hand) == 21

    def peek_dealer_card(self):
        return self.dealer_hand[0]

    def get_player_hand(self):
        return self.player_hand

    def get_dealer_hand(self):
        return self.dealer_hand

    def peek_dealer_score(self):
        return Blackjack.card_values[self.dealer_hand[0]]

    def should_dealer_hit(self):
        dealer_score = self.get_dealer_score()
        return dealer_score < 17 or (dealer_score == 17 and Blackjack.is_hand_soft(self.dealer_hand))

    def dealer_turn(self):
        while self.should_dealer_hit():
            self.dealer_hand.append(self.deck.pop())

    def get_dealer_score(self):
        score = Blackjack.get_hand_score(self.dealer_hand)
        n_aces = self.dealer_hand.count('A')
        while score > 21 and n_aces > 0:
            score -= 10
            n_aces -= 1
        return score

    def get_player_score(self):
        score = Blackjack.get_hand_score(self.player_hand)
        n_aces = self.player_hand.count('A')
        while score > 21 and n_aces > 0:
            score -= 10
            n_aces -= 1
        return score

    def get_raw_player_score(self):
        return Blackjack.get_hand_score(self.player_hand)

    def get_state(self, final=False):
        # state like: [score dealer is showing (int), player score (int), does the player have a soft hand (binary indicator)]
        player_score = self.get_player_score()
        terminal = final or player_score >= 21
        dealer_state = self.get_dealer_score() if terminal else self.peek_dealer_score()
        return (dealer_state, player_score, 1 if Blackjack.is_hand_soft(self.player_hand) else 0), terminal

    def stand(self):
        pass

    def hit(self):
        self.player_hand.append(self.deck.pop())

    def double(self):
        self.player_hand.append(self.deck.pop())
        self.doubled = True

    def split(self):
        # TODO
        pass

    def take_action(self, action):
        if action == 0:
            self.stand()
            self.dealer_turn()
        elif action == 1:
            self.hit()
        elif action == 2:
            self.double()
            self.dealer_turn()
        elif action == 3:
            # TODO
            self.split()
        else:
            pass
        return self.get_state(action == 0 or action == 2)

    def get_player_reward(self):
        dealer_score = self.get_dealer_score()
        player_score = self.get_player_score()
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
            reward *= 2
        return reward

    def is_action_illegal(self, action):
        return (action == 2 or action == 3) and len(self.player_hand) > 2

    def step(self, action):
        print(self.dealer_hand, self.player_hand)
        state, terminal = self.get_state()
        if terminal or self.is_action_illegal(action):
            return state, action, self.illegal_reward, state, True
        else:
            state_prime, terminal = self.take_action(action)
            reward = self.get_player_reward() if terminal else 0
            print(self.dealer_hand, self.player_hand)
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
