import sys

def snail_goal(n:int) -> list[int]:
    goal = [[0]*n for _ in range(n)]
    num = 1
    left, right, top, bottom = 0, n - 1, 0, n - 1
    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            goal[top][i] = num
            num += 1
        top += 1
        for i in range(top, bottom + 1):
            goal[i][right] = num
            num += 1
        right -= 1
        if top <= bottom:
            for i in range(right, left - 1, -1):
                goal[bottom][i] = num
                num += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                goal[i][left] = num
                num += 1
            left += 1
    # Flatten and replace the last number (n*n) with 0 for blank tile
    flat = [goal[i][j] for i in range(n) for j in range(n)]
    flat = [0 if x == n * n else x for x in flat]
    return flat

def print_board(board: list[int], n: int) -> None:
    for i in range(n):
        for j in range(n):
            val = board[i * n + j]
            # Treat 0 as the blank tile
            if val == 0:
                print("  ", end=" ")
            else:
                print(f"{val:2}", end=" ")
        print()

def snail_random(n:int) -> list[int]:
    import random
    goal = snail_goal(n)
    shuffled = goal[:]
    random.shuffle(shuffled)
    return shuffled

def neighbors(state: list[int], n: int) -> list[list[int]]:
    result = []
    zero_index = state.index(0)
    x, y = divmod(zero_index, n)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            new_index = nx * n + ny
            new_state = state[:]
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            result.append(new_state)
    return result

def load_puzzle_from_file(path: str) -> tuple[list[int], int]:
    with open(path, "r", encoding="utf-8") as f:
        # Keep only non-empty, non-comment lines
        raw_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not raw_lines:
        raise ValueError("File is empty or only contains comments.")

    # First line = N
    n = int(raw_lines[0])

    # We expect at least N more lines for the puzzle
    if len(raw_lines) - 1 < n:
        raise ValueError(f"Not enough rows for a {n}x{n} puzzle.")

    values: list[int] = []

    # Next N lines = the board rows
    for i in range(1, 1 + n):
        parts = raw_lines[i].split()
        if len(parts) != n:
            raise ValueError(f"Row {i} does not contain {n} numbers.")
        values.extend(int(x) for x in parts)

    # Check that we have exactly N*N values
    if len(values) != n * n:
        raise ValueError(f"Expected {n*n} values, got {len(values)}.")

    # Check that the puzzle contains all numbers from 0 to N^2-1
    if sorted(values) != list(range(n * n)):
        raise ValueError("Puzzle must contain all numbers from 0 to N^2-1 exactly once.")

    return values, n

def is_solvable(start: list[int], goal: list[int]) -> bool:
    """
    Determine if a puzzle is solvable by comparing the parity of the
    permutation needed to reach the goal. This method works for any goal
    pattern (including the snail/spiral goal).
    """

    # 1. Build goal index (tile -> order in goal, ignoring 0)
    goal_index = {}
    index = 0
    for tile in goal:
        if tile != 0:
            goal_index[tile] = index
            index += 1

    # 2. Convert the start tiles into the goal-order permutation (ignore 0)
    perm = []
    for tile in start:
        if tile != 0:
            perm.append(goal_index[tile])

    # 3. Count inversions in the permutation
    inv = 0
    length = len(perm)
    for i in range(length):
        for j in range(i + 1, length):
            if perm[i] > perm[j]:
                inv += 1

    # 4. Solvable if inversion count is even
    return (inv % 2) == 0


if __name__ == "__main__":
    board, n = load_puzzle_from_file("puzzle3.txt")
    goal = snail_goal(n)

    print("Start state:")
    print_board(board, n)
    print("Goal state:")
    print_board(goal, n)

    if not is_solvable(board, goal):
        print("This puzzle is UNSOLVABLE with respect to the snail goal.")
        sys.exit(0)  # or return

    print("This puzzle is solvable! (we will solve it later)")
