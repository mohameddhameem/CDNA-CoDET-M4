def arena(n, heroes):
    winners = 0
    for i in range(100000):
        hero1, hero2 = heroes[i % n], heroes[(i + 1) % n]
        if hero1 < hero2:
            winners += 1
    return winners