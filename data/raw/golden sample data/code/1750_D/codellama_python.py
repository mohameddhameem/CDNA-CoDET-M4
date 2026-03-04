# Define a function to count the number of red nodes in a subtree
def count_red(subtree):
  count = 0
  for node in subtree:
    if node["color"] == "red":
      count += 1
  return count

# Define a function to play the game
def play_game(n, k):
  # Initialize the tree with white nodes
  tree = [{"color": "white"}] * n

  # Define Red's and Blue's turn functions
  def red_turn(subtree):
    # Pick a subtree
    subtree_index = random.choice(range(len(tree)))
    subtree = tree[subtree_index:subtree_index+1]

    # Color up to k nodes red
    for node in subtree:
      if node["color"] == "white":
        node["color"] = "red"
        count_red(subtree) <= k
        break

  def blue_turn(subtree):
    # Pick a subtree
    subtree_index = random.choice(range(len(tree)))
    subtree = tree[subtree_index:subtree_index+1]

    # Color up to k nodes blue
    for node in subtree:
      if node["color"] == "white" and count_red(subtree) < k:
        node["color"] = "blue"
        break

  # Play the game
  while True:
    # Check if the game is over
    if count_red(tree) > 0 and count_blue(tree) > 0:
      break

    # Take a turn
    if count_red(tree) > 0 and count_blue(tree) > 0:
      # Take a red turn
      red_turn([tree[0]])
    else:
      # Take a blue turn
      blue_turn([tree[0]])

  # Calculate the score
  score = 0
  for node in tree:
    if node["color"] == "red":
      score += 1
    elif node["color"] == "blue":
      score -= 1

  return score

# Test the program
print(play_game(4, 2)) # 1
print(play_game(5, 2)) # 6
print(play_game(7, 2)) # 4