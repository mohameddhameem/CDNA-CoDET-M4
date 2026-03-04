def is_perfect_keyboard(password):
    seen = set()
    for i in range(len(password)):
        if password[i] in seen:
            return False
        seen.add(password[i])
    return True

def get_keyboard_layout(password):
    if not is_perfect_keyboard(password):
        return "NO"
    else:
        keyboard = []
        for i in range(len(password)):
            keyboard.append(password[i])
        return "YES " + "".join(keyboard)

# test cases
test_cases = int(input())
for i in range(test_cases):
    password = input()
    print(get_keyboard_layout(password))