import numpy as np


def mine_probability(matrix, weights):
    """
    Calculate the probability of the center square being a mine based on the surrounding squares.
    """
    # Extract surrounding elements (exclude the center element)
    elements=np.delete(matrix.flatten(), 4)
    # revealed empty squares = -1
    revealed=np.sum(elements == -1)
    # Check for revealed numbers and calculate the required mines
    numbers=[value for value in elements if value > 0]
    total_required_mines = sum(numbers)

    # If there is one revealed square and total_required_mines >= 1
    if revealed >= 7 and total_required_mines >= 1:
        return 1.0

    # Apply weights to surrounding elements
    weighted=np.array([weights[value] * np.sum(elements == value)
        for value in np.unique(elements)])
    total_weighted_counts = np.sum(weighted)

    # If total weighted counts are zero, return zero probability
    if total_weighted_counts == 0:
        return 0.0
    probability = total_weighted_counts/8

    return max(0, min(1, probability))

def color(probability):
    """
    Map a probability (0 to 1) to a color gradient between blue and red.
    """
    probability = max(0, min(1, probability))
    r = int(probability * 255)      # Red component = increased probability
    b = int((1-probability) * 255)  # Blue component = decreased probability
    g = 0

    return r, g, b


# Test:
array = np.array([
    [-3, 2, 2],
    [-3, 0, 1],  # Center element is 0 for the square we are calculating
    [-3, -3, -1]])

# Weight map for each value
maps = {-3: 0.0,  # Unrevealed square
    -1: 1.0,  # Revealed square
    0: 0.0,   # Boundary
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0,
    5: 5.0,
    6: 6.0,
    7: 7.0,
    8: 8.0}

mine_prob = mine_probability(array, maps)
color = color(mine_prob)