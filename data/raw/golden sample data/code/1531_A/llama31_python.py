color = 'blue'
blocked = False

for _ in range(int(input())):
    message = input()
    if message in ['lock', 'unlock']:
        if message == 'lock':
            blocked = True
        else:
            blocked = False
    elif not blocked:
        color = message

print(color)