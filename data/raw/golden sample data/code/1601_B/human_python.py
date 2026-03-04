from collections import deque

import io, os





input = io.BytesIO(os.read(0, \

         os.fstat(0).st_size)).readline





def frog(n, a, b):

    queue = deque()



    levels = {i: None for i in range(n + 1)}

    levels[n] = (0, None, None)



    queue.append(n)



    best_jump = n



    while len(queue) > 0:

        current_pos = queue.popleft()

        no_of_jumps, pos_before_slip, came_from = levels[current_pos]



        starting_jump = current_pos - best_jump



        best_jump = min(best_jump, current_pos - a[current_pos - 1])



        jumps = [0] + list(range(starting_jump, a[current_pos - 1] + 1))

        for jump in jumps:

            position_after_jump = current_pos - jump



            if position_after_jump <= 0:

                sequence = find_sequence(current_pos, levels)

                print(len(sequence))

                print(" ".join(map(str, sequence)))

                return



            position_after_slip = position_after_jump + b[position_after_jump - 1]



            if levels[position_after_slip] is None:

                levels[position_after_slip] = (no_of_jumps + 1, position_after_jump, current_pos)

                queue.append(position_after_slip)



    print(-1)

    return





def find_sequence(current_pos, levels):

    sequence = [0]

    while True:

        no_of_jumps, pos_before_slip, came_from = levels[current_pos]

        if pos_before_slip is None:

            break

        sequence += [pos_before_slip]

        current_pos = came_from

    return sequence[::-1]





n = int(input())

a = list(map(int, input().decode().split(" ")))

b = list(map(int, input().decode().split(" ")))

frog(n, a, b)

