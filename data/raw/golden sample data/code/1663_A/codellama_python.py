def who_tested(input_list):
    tested = []
    for item in input_list:
        if item == 'tested':
            tested.append(item)
    return ', '.join(tested)