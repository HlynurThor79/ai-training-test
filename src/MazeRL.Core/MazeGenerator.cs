namespace MazeRL.Core;

/// <summary>
/// Random maze generation using a recursive backtracker (depth-first carving).
/// Produces a "perfect" maze (every open cell reachable, no loops), then knocks
/// out a few extra walls so there are alternative routes — more interesting for
/// an RL agent, because now there are short paths AND long paths to compare.
/// </summary>
public static class MazeGenerator
{
    public static Maze Generate(int rows, int cols, int? seed = null, double braidChance = 0.08)
    {
        if (rows < 3 || cols < 3)
            throw new ArgumentException("Maze must be at least 3x3.");

        var rnd = seed.HasValue ? new Random(seed.Value) : new Random();
        var maze = new Maze(rows, cols);

        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                maze.SetWall(r, c, true);

        // Carve rooms on the even-index lattice, moving 2 cells at a time and
        // opening the wall in between. IsWall doubles as the "unvisited" flag.
        (int, int)[] jumps = [(-2, 0), (2, 0), (0, -2), (0, 2)];
        var stack = new Stack<(int Row, int Col)>();
        maze.SetWall(0, 0, false);
        stack.Push((0, 0));

        var candidates = new List<(int Row, int Col)>(4);
        while (stack.Count > 0)
        {
            var (r, c) = stack.Peek();
            candidates.Clear();
            foreach (var (dr, dc) in jumps)
                if (maze.InBounds(r + dr, c + dc) && maze.IsWall(r + dr, c + dc))
                    candidates.Add((r + dr, c + dc));

            if (candidates.Count == 0)
            {
                stack.Pop();
                continue;
            }

            var (nr, nc) = candidates[rnd.Next(candidates.Count)];
            maze.SetWall((r + nr) / 2, (c + nc) / 2, false);
            maze.SetWall(nr, nc, false);
            stack.Push((nr, nc));
        }

        // Even dimensions leave the last row/col solid — open stretches of it
        // next to carved cells so it becomes usable corridor instead of dead border.
        if (rows % 2 == 0)
            for (var c = 0; c < cols; c++)
                if (!maze.IsWall(rows - 2, c) && rnd.NextDouble() < 0.5)
                    maze.SetWall(rows - 1, c, false);
        if (cols % 2 == 0)
            for (var r = 0; r < rows; r++)
                if (!maze.IsWall(r, cols - 2) && rnd.NextDouble() < 0.5)
                    maze.SetWall(r, cols - 1, false);

        // Braid: remove some walls that separate two corridors, creating loops.
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
            {
                if (!maze.IsWall(r, c) || rnd.NextDouble() >= braidChance) continue;
                var vertical = maze.InBounds(r - 1, c) && maze.InBounds(r + 1, c)
                    && !maze.IsWall(r - 1, c) && !maze.IsWall(r + 1, c);
                var horizontal = maze.InBounds(r, c - 1) && maze.InBounds(r, c + 1)
                    && !maze.IsWall(r, c - 1) && !maze.IsWall(r, c + 1);
                if (vertical || horizontal)
                    maze.SetWall(r, c, false);
            }

        // Start at the carve origin; goal = farthest reachable cell (BFS),
        // which guarantees solvability and a decently long solution.
        maze.Start = (0, 0);
        maze.Goal = FarthestFrom(maze, maze.Start);
        return maze;
    }

    private static (int Row, int Col) FarthestFrom(Maze maze, (int Row, int Col) from)
    {
        var visited = new bool[maze.Rows, maze.Cols];
        var queue = new Queue<(int Row, int Col)>();
        visited[from.Row, from.Col] = true;
        queue.Enqueue(from);
        var last = from;

        (int, int)[] moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        while (queue.Count > 0)
        {
            var (r, c) = queue.Dequeue();
            last = (r, c);
            foreach (var (dr, dc) in moves)
            {
                var (nr, nc) = (r + dr, c + dc);
                if (maze.InBounds(nr, nc) && !maze.IsWall(nr, nc) && !visited[nr, nc])
                {
                    visited[nr, nc] = true;
                    queue.Enqueue((nr, nc));
                }
            }
        }
        return last;
    }
}
