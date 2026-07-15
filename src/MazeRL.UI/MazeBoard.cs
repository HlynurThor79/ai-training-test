using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using MazeRL.Core;

namespace MazeRL.UI;

public enum EditMode { Walls, Start, Goal }

/// <summary>
/// Renders the maze plus everything the agent knows: a Q-value heatmap,
/// policy arrows, the test-run path, and the ball. Also acts as the maze
/// editor — click/drag paints walls, or places start/goal depending on Mode.
/// </summary>
public class MazeBoard : Control
{
    public Maze? Maze { get; set; }
    public MazeEnvironment? Env { get; set; }
    public IAgent? Agent { get; set; }

    public bool ShowHeatmap { get; set; } = true;
    public bool ShowArrows { get; set; } = true;
    public EditMode Mode { get; set; } = EditMode.Walls;

    public (int Row, int Col)? Ball { get; set; }
    public HashSet<int>? PathCells { get; set; }
    public (int Row, int Col)? SelectedCell { get; set; }

    /// <summary>Raised when the user clicks a cell (for the Q-value inspector).</summary>
    public event Action<int, int>? CellSelected;

    private bool _dragging;
    private bool _paintWall;

    private static readonly Color WallColor = Color.FromRgb(24, 24, 30);
    private static readonly Color UnvisitedColor = Color.FromRgb(46, 46, 56);
    private static readonly Color ColdColor = Color.FromRgb(58, 60, 74);
    private static readonly Color HotColor = Color.FromRgb(46, 204, 113);
    private static readonly IBrush StartBrush = new SolidColorBrush(Color.FromRgb(52, 152, 219));
    private static readonly IBrush GoalBrush = new SolidColorBrush(Color.FromRgb(241, 196, 15));
    private static readonly IBrush BallBrush = new SolidColorBrush(Color.FromRgb(230, 126, 34));
    private static readonly IBrush PathBrush = new SolidColorBrush(Color.FromArgb(180, 255, 255, 255));
    private static readonly IPen ArrowPen = new Pen(new SolidColorBrush(Color.FromArgb(200, 235, 235, 245)), 1.6);
    private static readonly IPen SelectedPen = new Pen(Brushes.White, 2);

    private double CellSize => Maze is null ? 0
        : Math.Min(Bounds.Width / Maze.Cols, Bounds.Height / Maze.Rows);

    private Point Origin
    {
        get
        {
            if (Maze is null) return default;
            var size = CellSize;
            return new Point(
                (Bounds.Width - size * Maze.Cols) / 2,
                (Bounds.Height - size * Maze.Rows) / 2);
        }
    }

    public override void Render(DrawingContext ctx)
    {
        ctx.FillRectangle(new SolidColorBrush(Color.FromRgb(27, 27, 34)), new Rect(Bounds.Size));
        if (Maze is null || Env is null || Agent is null) return;

        var size = CellSize;
        var origin = Origin;

        // Normalize best-Q over open cells so the heatmap gradient is visible.
        var min = double.MaxValue;
        var max = double.MinValue;
        for (var r = 0; r < Maze.Rows; r++)
            for (var c = 0; c < Maze.Cols; c++)
            {
                if (Maze.IsWall(r, c)) continue;
                var v = Agent.BestValue(Env.StateOf(r, c));
                if (v < min) min = v;
                if (v > max) max = v;
            }
        var range = max - min;

        for (var r = 0; r < Maze.Rows; r++)
        {
            for (var c = 0; c < Maze.Cols; c++)
            {
                var rect = new Rect(origin.X + c * size, origin.Y + r * size, size - 1, size - 1);
                var state = Env.StateOf(r, c);
                var isWall = Maze.IsWall(r, c);

                Color fill;
                if (isWall)
                {
                    fill = WallColor;
                }
                else
                {
                    var best = Agent.BestValue(state);
                    if (!ShowHeatmap || !Agent.HasLearned || best == 0 || range <= 0)
                        fill = UnvisitedColor;
                    else
                        fill = Lerp(ColdColor, HotColor, (best - min) / range);
                }
                ctx.FillRectangle(new SolidColorBrush(fill), rect);

                if ((r, c) == Maze.Goal)
                {
                    ctx.FillRectangle(GoalBrush, rect);
                    DrawLabel(ctx, "G", rect, size);
                    continue;
                }
                if ((r, c) == Maze.Start)
                {
                    ctx.DrawRectangle(new Pen(StartBrush, Math.Max(2, size * 0.08)), rect);
                    DrawLabel(ctx, "S", rect, size);
                }

                if (!isWall && ShowArrows && Agent.HasLearned && Agent.BestValue(state) != 0)
                    DrawArrow(ctx, Agent.BestAction(state), rect);

                if (!isWall && PathCells is not null && PathCells.Contains(state))
                    ctx.DrawEllipse(PathBrush, null, rect.Center, size * 0.08, size * 0.08);
            }
        }

        if (SelectedCell is { } sel && Maze.InBounds(sel.Row, sel.Col))
            ctx.DrawRectangle(SelectedPen,
                new Rect(origin.X + sel.Col * size, origin.Y + sel.Row * size, size - 1, size - 1));

        if (Ball is { } ball)
        {
            var center = new Point(origin.X + (ball.Col + 0.5) * size, origin.Y + (ball.Row + 0.5) * size);
            ctx.DrawEllipse(BallBrush, null, center, size * 0.3, size * 0.3);
        }
    }

    private static void DrawLabel(DrawingContext ctx, string text, Rect rect, double size)
    {
        var ft = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
            new Typeface(FontFamily.Default, FontStyle.Normal, FontWeight.Bold),
            size * 0.45, Brushes.Black);
        ctx.DrawText(ft, new Point(
            rect.X + (rect.Width - ft.Width) / 2,
            rect.Y + (rect.Height - ft.Height) / 2));
    }

    private static void DrawArrow(DrawingContext ctx, int action, Rect rect)
    {
        // Screen-space direction per action: Up, Down, Left, Right.
        var (dx, dy) = action switch
        {
            MazeEnvironment.Up => (0.0, -1.0),
            MazeEnvironment.Down => (0.0, 1.0),
            MazeEnvironment.Left => (-1.0, 0.0),
            _ => (1.0, 0.0),
        };

        var center = rect.Center;
        var len = rect.Width * 0.26;
        var tip = new Point(center.X + dx * len, center.Y + dy * len);
        var tail = new Point(center.X - dx * len * 0.6, center.Y - dy * len * 0.6);
        ctx.DrawLine(ArrowPen, tail, tip);

        // Arrowhead: two short lines angled back from the tip.
        var head = rect.Width * 0.12;
        var (px, py) = (-dy, dx); // perpendicular
        ctx.DrawLine(ArrowPen, tip, new Point(tip.X - dx * head + px * head, tip.Y - dy * head + py * head));
        ctx.DrawLine(ArrowPen, tip, new Point(tip.X - dx * head - px * head, tip.Y - dy * head - py * head));
    }

    private static Color Lerp(Color a, Color b, double t)
    {
        t = Math.Clamp(t, 0, 1);
        return Color.FromRgb(
            (byte)(a.R + (b.R - a.R) * t),
            (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t));
    }

    private (int Row, int Col)? CellAt(Point p)
    {
        if (Maze is null) return null;
        var size = CellSize;
        if (size <= 0) return null;
        var origin = Origin;
        var col = (int)((p.X - origin.X) / size);
        var row = (int)((p.Y - origin.Y) / size);
        return Maze.InBounds(row, col) ? (row, col) : null;
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (CellAt(e.GetPosition(this)) is not { } cell) return;
        var (row, col) = cell;

        SelectedCell = (row, col);
        CellSelected?.Invoke(row, col);

        switch (Mode)
        {
            case EditMode.Walls:
                _dragging = true;
                _paintWall = !Maze!.IsWall(row, col); // drag paints whatever the first cell became
                ApplyWall(row, col);
                break;
            case EditMode.Start:
                if (!Maze!.IsWall(row, col) && (row, col) != Maze.Goal)
                    Maze.Start = (row, col);
                break;
            case EditMode.Goal:
                if (!Maze!.IsWall(row, col) && (row, col) != Maze.Start)
                    Maze.Goal = (row, col);
                break;
        }
        InvalidateVisual();
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (!_dragging || Mode != EditMode.Walls) return;
        if (CellAt(e.GetPosition(this)) is not { } cell) return;
        ApplyWall(cell.Row, cell.Col);
        InvalidateVisual();
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _dragging = false;
    }

    private void ApplyWall(int row, int col)
    {
        if (Maze is null) return;
        if ((row, col) == Maze.Start || (row, col) == Maze.Goal) return;
        Maze.SetWall(row, col, _paintWall);
    }
}
