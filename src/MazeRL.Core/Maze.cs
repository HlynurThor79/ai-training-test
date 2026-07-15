using System.Text.Json;
using System.Text.Json.Serialization;

namespace MazeRL.Core;

/// <summary>
/// The maze layout: a grid of walls/open cells plus a start and a goal.
/// Pure data — knows nothing about agents, rewards, or learning.
/// </summary>
public class Maze
{
    public int Rows { get; }
    public int Cols { get; }
    public (int Row, int Col) Start { get; set; }
    public (int Row, int Col) Goal { get; set; }

    private readonly bool[,] _walls;

    public Maze(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        _walls = new bool[rows, cols];
        Start = (0, 0);
        Goal = (rows - 1, cols - 1);
    }

    public bool IsWall(int row, int col) => _walls[row, col];

    public void SetWall(int row, int col, bool wall) => _walls[row, col] = wall;

    public bool InBounds(int row, int col) => row >= 0 && row < Rows && col >= 0 && col < Cols;

    /// <summary>
    /// Parse a maze from text lines: '#' = wall, '.' or ' ' = open, 'S' = start, 'G' = goal.
    /// Handy for defining test mazes in code before the GUI editor exists.
    /// </summary>
    public static Maze FromText(params string[] lines)
    {
        var maze = new Maze(lines.Length, lines[0].Length);
        for (var r = 0; r < lines.Length; r++)
        {
            if (lines[r].Length != maze.Cols)
                throw new ArgumentException($"Line {r} has length {lines[r].Length}, expected {maze.Cols}.");
            for (var c = 0; c < maze.Cols; c++)
            {
                switch (lines[r][c])
                {
                    case '#': maze.SetWall(r, c, true); break;
                    case 'S': maze.Start = (r, c); break;
                    case 'G': maze.Goal = (r, c); break;
                }
            }
        }
        return maze;
    }

    public string ToJson()
    {
        var dto = new MazeDto
        {
            Rows = Rows,
            Cols = Cols,
            Start = new[] { Start.Row, Start.Col },
            Goal = new[] { Goal.Row, Goal.Col },
            Walls = Enumerable.Range(0, Rows)
                .Select(r => string.Concat(Enumerable.Range(0, Cols).Select(c => _walls[r, c] ? '#' : '.')))
                .ToArray(),
        };
        return JsonSerializer.Serialize(dto, JsonOptions);
    }

    public static Maze FromJson(string json)
    {
        var dto = JsonSerializer.Deserialize<MazeDto>(json, JsonOptions)
                  ?? throw new ArgumentException("Invalid maze JSON.");
        var maze = new Maze(dto.Rows, dto.Cols)
        {
            Start = (dto.Start[0], dto.Start[1]),
            Goal = (dto.Goal[0], dto.Goal[1]),
        };
        for (var r = 0; r < dto.Rows; r++)
            for (var c = 0; c < dto.Cols; c++)
                maze.SetWall(r, c, dto.Walls[r][c] == '#');
        return maze;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    private class MazeDto
    {
        public int Rows { get; set; }
        public int Cols { get; set; }
        public int[] Start { get; set; } = [];
        public int[] Goal { get; set; } = [];
        public string[] Walls { get; set; } = [];
    }
}
