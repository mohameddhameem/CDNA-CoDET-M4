import itertools

def count_sets(cards):
    subsets = list(itertools.combinations(cards, 3))
    set_subsets = list(filter(lambda x: is_set(x), subsets))
    return len(set_subsets)

def is_set(cards):
    return all(card_i == card_j for card_i in cards for card_j in cards if card_i!= card_j)

if __name__ == '__main__':
    n, k = list(map(int, input().split()))
    cards = [input() for _ in range(n)]
    count = count_sets(cards)
    print(count)