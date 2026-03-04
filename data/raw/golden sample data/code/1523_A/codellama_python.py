import itertools

def game_of_life(n, m, queries):
    # Initialize the board with the initial state
    board = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            board[i][j] = int(queries[i][j])
    
    # Run the game iterations
    for i in range(queries[0][0]):
        # Get the current states of the cells
        states = list(itertools.chain(*board))

        # Compute the new states of the cells based on the current states
        new_states = []
        for state in states:
            if state == 1:
                new_states.append(state)
            else:
                new_states.append(0)
        
        # Update the board with the new states
        for i in range(n):
            for j in range(m):
                board[i][j] = new_states[i * m + j]
    
    # Return the answers to the queries
    return [board[i][j] for i, j, p in queries[1:]]