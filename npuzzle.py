#!/usr/bin/env python3
import argparse
import random
import heapq
from dataclasses import dataclass, field


# ==========================
# 1) Puzzle representation
# ==========================

def snail_goal(n: int) -> tuple[int, ...]:
    """
    Build the goal 'snail/spiral' configuration as a flat tuple of length n*n.
    We fill numbers 1..n^2-1 in a spiral, the remaining cell stays 0.
    """
    grid = [[0] * n for _ in range(n)]
    num = 1
    max_num = n * n - 1

    top, left = 0, 0
    bottom, right = n - 1, n - 1

    while num <= max_num:
        # left → right
        for j in range(left, right + 1):
            if num > max_num:
                break
            grid[top][j] = num
            num += 1
        top += 1

        # top → bottom
        for i in range(top, bottom + 1):
            if num > max_num:
                break
            grid[i][right] = num
            num += 1
        right -= 1

        # right → left
        for j in range(right, left - 1, -1):
            if num > max_num:
                break
            grid[bottom][j] = num
            num += 1
        bottom -= 1

        # bottom → top
        for i in range(bottom, top - 1, -1):
            if num > max_num:
                break
            grid[i][left] = num
            num += 1
        left += 1

    # Flatten the grid into a tuple
    flat = []
    for i in range(n):
        for j in range(n):
            flat.append(grid[i][j])
    return tuple(flat)


def print_board(tiles: tuple[int, ...], n: int) -> None:
    """Pretty-print a board."""
    for i in range(n):
        row = tiles[i * n:(i + 1) * n]
        print(" ".join(f"{x:2d}" for x in row))
    print()


def neighbors(tiles: tuple[int, ...], n: int):
    """
    Generate all neighbor states by sliding the blank (0) up/down/left/right.
    """
    idx0 = tiles.index(0)
    r0, c0 = divmod(idx0, n)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < n and 0 <= nc < n:
            new_idx = nr * n + nc
            new_tiles = list(tiles)
            # swap 0 with neighbor
            new_tiles[idx0], new_tiles[new_idx] = new_tiles[new_idx], new_tiles[idx0]
            yield tuple(new_tiles)


# ==========================
# 2) Input / random puzzles
# ==========================

def load_puzzle_from_file(path: str) -> tuple[tuple[int, ...], int]:
    """
    Format:
      - Comments start with '#'
      - First non-comment line: N
      - Next N non-comment lines: N integers, including 0
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not lines:
        raise ValueError("File is empty or only comments.")

    n = int(lines[0])
    if len(lines) - 1 < n:
        raise ValueError("Not enough rows for the puzzle.")

    values = []
    for i in range(1, 1 + n):
        parts = lines[i].split()
        if len(parts) != n:
            raise ValueError(f"Row {i} does not contain {n} numbers.")
        values.extend(int(x) for x in parts)

    if sorted(values) != list(range(n * n)):
        raise ValueError("Puzzle must contain all numbers from 0 to N^2-1 exactly once.")

    return tuple(values), n


def random_puzzle(n: int, shuffle_moves: int = 1000) -> tuple[int, ...]:
    """
    Start from the snail goal and apply random valid moves.
    This guarantees solvability.
    """
    current = snail_goal(n)
    for _ in range(shuffle_moves):
        current = random.choice(list(neighbors(current, n)))
    return current


# ==========================
# 3) Solvability check
# ==========================

def is_solvable(start: tuple[int, ...], goal: tuple[int, ...]) -> bool:
    """
    General solvability check based on permutation parity with respect to the goal.
    Works with any goal pattern (not just row-major).
    """
    if sorted(start) != sorted(goal):
        return False

    # Map each tile (≠0) to an index in the goal order
    goal_index = {}
    seq_idx = 0
    for t in goal:
        if t == 0:
            continue
        goal_index[t] = seq_idx
        seq_idx += 1

    # Build the permutation representing where each goal tile appears in start
    perm = []
    for t in start:
        if t == 0:
            continue
        perm.append(goal_index[t])

    # Count inversions in perm
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1

    # Even number of inversions → solvable
    return (inv % 2) == 0


# ==========================
# 4) Heuristics
# ==========================

def build_goal_positions(goal: tuple[int, ...], n: int) -> dict[int, tuple[int, int]]:
    """For each tile value, store its (row, col) in the goal."""
    pos = {}
    for idx, tile in enumerate(goal):
        if tile == 0:
            continue
        pos[tile] = divmod(idx, n)
    return pos


def h_manhattan(tiles: tuple[int, ...], goal_pos: dict[int, tuple[int, int]], n: int) -> int:
    """Sum of Manhattan distances of each tile to its goal position."""
    d = 0
    for idx, tile in enumerate(tiles):
        if tile == 0:
            continue
        r, c = divmod(idx, n)
        gr, gc = goal_pos[tile]
        d += abs(r - gr) + abs(c - gc)
    return d


def h_misplaced(tiles: tuple[int, ...], goal_pos: dict[int, tuple[int, int]], n: int) -> int:
    """Number of tiles that are not in their goal position (excluding 0)."""
    count = 0
    for idx, tile in enumerate(tiles):
        if tile == 0:
            continue
        gr, gc = goal_pos[tile]
        if idx != gr * n + gc:
            count += 1
    return count


def h_max_of_two(tiles: tuple[int, ...], goal_pos: dict[int, tuple[int, int]], n: int) -> int:
    """
    Third heuristic: max(Manhattan, Misplaced).
    Max of admissible heuristics is still admissible.
    """
    return max(h_manhattan(tiles, goal_pos, n),
               h_misplaced(tiles, goal_pos, n))


HEURISTICS = {
    "manhattan": h_manhattan,
    "misplaced": h_misplaced,
    "max":       h_max_of_two,
}


# ==========================
# 5) A* search
# ==========================

@dataclass(order=True)
class Node:
    f: int
    g: int = field(compare=False)
    h: int = field(compare=False)
    tiles: tuple[int, ...] = field(compare=False)
    parent: "Node | None" = field(compare=False, default=None)


def reconstruct_path(node: Node) -> list[tuple[int, ...]]:
    """Follow parent pointers back to the root and reverse."""
    path = []
    cur = node
    while cur is not None:
        path.append(cur.tiles)
        cur = cur.parent
    path.reverse()
    return path


def astar(start: tuple[int, ...],
          goal: tuple[int, ...],
          n: int,
          heuristic_name: str = "manhattan"):
    """
    Core A* algorithm:
      - open set = min-heap on f = g + h
      - closed set = visited states
      - returns (goal_node or None, expanded_count, max_open_closed_size)
    """
    if heuristic_name not in HEURISTICS:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    h_func = HEURISTICS[heuristic_name]
    goal_pos = build_goal_positions(goal, n)

    h0 = h_func(start, goal_pos, n)
    root = Node(f=h0, g=0, h=h0, tiles=start, parent=None)

    open_heap: list[Node] = [root]
    heapq.heapify(open_heap)

    best_g: dict[tuple[int, ...], int] = {start: 0}
    closed: set[tuple[int, ...]] = set()

    expanded = 0
    max_in_memory = 1

    while open_heap:
        node = heapq.heappop(open_heap)
        expanded += 1

        if node.tiles == goal:
            max_in_memory = max(max_in_memory, len(open_heap) + len(closed))
            return node, expanded, max_in_memory

        closed.add(node.tiles)

        for neigh in neighbors(node.tiles, n):
            if neigh in closed:
                continue

            tentative_g = node.g + 1

            if neigh in best_g and tentative_g >= best_g[neigh]:
                continue  # not a better path

            best_g[neigh] = tentative_g
            h_val = h_func(neigh, goal_pos, n)
            child = Node(f=tentative_g + h_val,
                         g=tentative_g,
                         h=h_val,
                         tiles=neigh,
                         parent=node)
            heapq.heappush(open_heap, child)

        max_in_memory = max(max_in_memory, len(open_heap) + len(closed))

    # No solution found
    return None, expanded, max_in_memory


# ==========================
# 6) Command-line interface
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="N-Puzzle solver using A*.")
    parser.add_argument("file", nargs="?", help="Puzzle file (optional).")
    parser.add_argument("--size", "-s", type=int, default=3,
                        help="Size N of the puzzle (for random generation).")
    parser.add_argument("--random", action="store_true",
                        help="Generate a random puzzle instead of reading from file.")
    parser.add_argument("--heuristic", "-H",
                        choices=list(HEURISTICS.keys()),
                        default="manhattan",
                        help="Heuristic to use.")
    parser.add_argument("--shuffle", type=int, default=1000,
                        help="Number of random moves to shuffle the goal state (for --random).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.file and args.random:
        raise SystemExit("Choose either a file or --random, not both.")

    if args.file:
        start, n = load_puzzle_from_file(args.file)
    else:
        n = args.size
        start = random_puzzle(n, shuffle_moves=args.shuffle)

    goal = snail_goal(n)

    print("Initial state:")
    print_board(start, n)
    print("Goal state (snail pattern):")
    print_board(goal, n)

    if not is_solvable(start, goal):
        print("This puzzle is UNSOLVABLE with respect to the snail goal.")
        return

    print(f"Solving with A* and heuristic '{args.heuristic}'...")
    goal_node, expanded, max_mem = astar(start, goal, n, heuristic_name=args.heuristic)

    if goal_node is None:
        print("No solution found by A* (unexpected if solvability test passed).")
        print(f"States expanded: {expanded}")
        print(f"Max states in memory: {max_mem}")
        return

    path = reconstruct_path(goal_node)

    print("\n=== RESULTS ===")
    print(f"Time complexity (states expanded): {expanded}")
    print(f"Space complexity (max states in memory): {max_mem}")
    print(f"Number of moves in solution: {len(path) - 1}")

    print("\nSolution path:")
    for step, state in enumerate(path):
        print(f"Step {step}:")
        print_board(state, n)


if __name__ == "__main__":
    main()
