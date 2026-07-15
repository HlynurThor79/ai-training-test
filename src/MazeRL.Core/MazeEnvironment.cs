namespace MazeRL.Core;

/// <summary>
/// Wraps a Maze as a reinforcement-learning environment.
/// This is where the reward function lives — the only "teaching signal" the agent ever gets.
/// </summary>
public class MazeEnvironment : IEnvironment
{
    public const int Up = 0;
    public const int Down = 1;
    public const int Left = 2;
    public const int Right = 3;

    public static readonly string[] ActionNames = ["Up", "Down", "Left", "Right"];

    private static readonly (int DRow, int DCol)[] Moves =
    [
        (-1, 0), // Up
        (1, 0),  // Down
        (0, -1), // Left
        (0, 1),  // Right
    ];

    public Maze Maze { get; }

    public double StepReward { get; init; } = -1;      // every move costs a little → prefer short paths
    public double WallReward { get; init; } = -5;      // bumping a wall or the edge hurts more
    public double GoalReward { get; init; } = 100;     // reaching the goal is the jackpot

    private (int Row, int Col) _pos;

    public MazeEnvironment(Maze maze)
    {
        Maze = maze;
        _pos = maze.Start;
    }

    public int StateCount => Maze.Rows * Maze.Cols;
    public int ActionCount => Moves.Length;

    public (int Row, int Col) Position => _pos;

    /// <summary>State is just the cell index: row * Cols + col.</summary>
    public int StateOf(int row, int col) => row * Maze.Cols + col;

    public (int Row, int Col) CellOf(int state) => (state / Maze.Cols, state % Maze.Cols);

    public int Reset()
    {
        _pos = Maze.Start;
        return StateOf(_pos.Row, _pos.Col);
    }

    public StepResult Step(int action)
    {
        var (dRow, dCol) = Moves[action];
        var newRow = _pos.Row + dRow;
        var newCol = _pos.Col + dCol;

        // Hitting the edge or a wall: stay in place, pay the penalty.
        if (!Maze.InBounds(newRow, newCol) || Maze.IsWall(newRow, newCol))
            return new StepResult(StateOf(_pos.Row, _pos.Col), WallReward, false);

        _pos = (newRow, newCol);

        if (_pos == Maze.Goal)
            return new StepResult(StateOf(newRow, newCol), GoalReward, true);

        return new StepResult(StateOf(newRow, newCol), StepReward, false);
    }
}
