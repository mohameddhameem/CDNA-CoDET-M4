def kill_monsters(monsters):
    while True:
        if all(monster <= 1 for monster in monsters):
            return len(monsters)
        monsters = [max(monster - 1, 0) for monster in monsters]