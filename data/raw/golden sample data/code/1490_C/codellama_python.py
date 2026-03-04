def sort_cubes(n, cubes):
    # Initialize a queue to hold the initial configuration of cubes
    queue = [cubes]
    
    # While the queue is not empty
    while queue:
        # Dequeue a configuration
        config = queue.pop(0)
        
        # If the configuration is sorted, return "YES"
        if is_sorted(config):
            return "YES"
        
        # Otherwise, enqueue all the possible next configurations by exchanging two adjacent cubes
        for i in range(n-1):
            # Get the two adjacent cubes
            cube1 = config[i]
            cube2 = config[i+1]
            
            # If the two cubes are in the correct order, continue to the next cube
            if cube1 > cube2:
                continue
            
            # Else, exchange the two cubes and enqueue the new configuration
            else:
                new_cubes = config[:i] + [cube1] + config[i+1:]
                queue.append(new_cubes)

    # If the queue is empty after all configurations have been dequeued and no sorted configuration was found, return "NO"
    return "NO"

def is_sorted(cubes):
    for i in range(len(cubes)-1):
        if cubes[i] > cubes[i+1]:
            return False
    return True