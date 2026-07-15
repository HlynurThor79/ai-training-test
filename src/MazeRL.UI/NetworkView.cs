using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using MazeRL.Core;

namespace MazeRL.UI;

/// <summary>
/// Live picture of the DQN's neural network. Shows the forward pass for one
/// state (the selected cell, or the ball while it runs):
///  • the input layer drawn as a mini-maze — the lit cell is the one-hot "1";
///  • hidden nodes glowing with their activation strength;
///  • edges colored by weight sign (green +, blue −) and sized by magnitude;
///  • the four Q-value outputs, best one ringed.
/// Redrawn every training tick, so you can watch the weights reorganize.
/// </summary>
public class NetworkView : Control
{
    public DqnAgent? Agent { get; set; }
    public MazeEnvironment? Env { get; set; }

    /// <summary>Which state's forward pass to visualize.</summary>
    public int State { get; set; }

    private static readonly Color Background = Color.FromRgb(22, 22, 28);
    private static readonly Color NodeOff = Color.FromRgb(52, 52, 62);
    private static readonly Color NodeOn = Color.FromRgb(46, 204, 113);
    private static readonly Color PositiveEdge = Color.FromRgb(46, 204, 113);
    private static readonly Color NegativeEdge = Color.FromRgb(52, 152, 219);
    private static readonly IBrush ActiveInputBrush = new SolidColorBrush(Color.FromRgb(230, 126, 34));
    private static readonly IBrush WallBrush = new SolidColorBrush(Color.FromRgb(15, 15, 19));
    private static readonly IBrush TextBrush = new SolidColorBrush(Color.FromRgb(180, 180, 190));
    private static readonly IPen BestPen = new Pen(Brushes.White, 2);

    public override void Render(DrawingContext ctx)
    {
        var bounds = new Rect(Bounds.Size);
        ctx.FillRectangle(new SolidColorBrush(Background), bounds);

        if (Agent is null || Env is null)
        {
            DrawText(ctx, "Switch the agent to “Neural net (DQN)” to watch the network learn.",
                new Point(14, bounds.Height / 2 - 8), 12, Brushes.Gray);
            return;
        }

        var net = Agent.Network;
        var acts = net.Forward(Agent.Encode(State));
        var layers = net.LayerSizes.Length;

        var maze = Env.Maze;
        var w = bounds.Width;
        var h = bounds.Height;
        var top = 26.0;   // room for layer captions
        var bottom = 16.0;
        var usableH = h - top - bottom;

        // ---- Input layer: a mini-maze where the agent's cell is lit ----
        var cell = Math.Min(150.0 / maze.Cols, usableH / maze.Rows);
        var gridW = cell * maze.Cols;
        var gridH = cell * maze.Rows;
        var gridX = 14.0;
        var gridY = top + (usableH - gridH) / 2;

        // Node positions for every layer (input nodes = centers of the mini-maze cells).
        var pos = new Point[layers][];
        pos[0] = new Point[net.LayerSizes[0]];
        for (var s = 0; s < net.LayerSizes[0]; s++)
        {
            var (r, c) = Env.CellOf(s);
            pos[0][s] = new Point(gridX + (c + 0.5) * cell, gridY + (r + 0.5) * cell);
        }

        var firstX = gridX + gridW + 70;
        var lastX = w - 110; // leave room for output labels
        for (var l = 1; l < layers; l++)
        {
            var x = layers == 2 ? lastX : firstX + (lastX - firstX) * (l - 1) / (layers - 2.0);
            var n = net.LayerSizes[l];
            var spacing = Math.Min(l == layers - 1 ? 44 : 26, usableH / n);
            var y0 = top + (usableH - spacing * (n - 1)) / 2;
            pos[l] = new Point[n];
            for (var i = 0; i < n; i++)
                pos[l][i] = new Point(x, y0 + i * spacing);
        }

        // ---- Edges (drawn first, nodes on top) ----
        for (var l = 0; l < layers - 1; l++)
        {
            var maxAbs = 1e-9;
            foreach (var row in net.Weights[l])
                foreach (var wt in row)
                    maxAbs = Math.Max(maxAbs, Math.Abs(wt));

            if (l == 0)
            {
                // 100+ input nodes → only draw the edges that carry signal right now:
                // the ones leaving the active (one-hot) input.
                for (var to = 0; to < net.LayerSizes[1]; to++)
                    DrawEdge(ctx, pos[0][State], pos[1][to], net.Weights[0][to][State], maxAbs);
            }
            else
            {
                for (var to = 0; to < net.LayerSizes[l + 1]; to++)
                    for (var from = 0; from < net.LayerSizes[l]; from++)
                        DrawEdge(ctx, pos[l][from], pos[l + 1][to], net.Weights[l][to][from], maxAbs);
            }
        }

        // ---- Input mini-maze ----
        for (var r = 0; r < maze.Rows; r++)
        {
            for (var c = 0; c < maze.Cols; c++)
            {
                var rect = new Rect(gridX + c * cell, gridY + r * cell, cell - 0.5, cell - 0.5);
                var s = Env.StateOf(r, c);
                if (s == State)
                    ctx.FillRectangle(ActiveInputBrush, rect);
                else
                    ctx.FillRectangle(maze.IsWall(r, c) ? WallBrush : new SolidColorBrush(NodeOff), rect);
            }
        }

        // ---- Hidden nodes: brightness = activation ----
        for (var l = 1; l < layers - 1; l++)
        {
            var layerMax = Math.Max(1e-9, acts[l].Max());
            for (var i = 0; i < net.LayerSizes[l]; i++)
            {
                var t = acts[l][i] / layerMax;
                var brush = new SolidColorBrush(Lerp(NodeOff, NodeOn, t));
                ctx.DrawEllipse(brush, null, pos[l][i], 7, 7);
            }
        }

        // ---- Output nodes: the four Q-values ----
        var outputs = acts[^1];
        var best = 0;
        for (var i = 1; i < outputs.Length; i++)
            if (outputs[i] > outputs[best]) best = i;

        var outMin = outputs.Min();
        var outRange = Math.Max(1e-9, outputs.Max() - outMin);
        for (var i = 0; i < outputs.Length; i++)
        {
            var t = (outputs[i] - outMin) / outRange;
            var brush = new SolidColorBrush(Lerp(NodeOff, NodeOn, t));
            ctx.DrawEllipse(brush, i == best ? BestPen : null, pos[^1][i], 10, 10);
            DrawText(ctx, $"{MazeEnvironment.ActionNames[i]} {outputs[i]:F1}",
                new Point(pos[^1][i].X + 16, pos[^1][i].Y - 8), 12,
                i == best ? Brushes.White : TextBrush);
        }

        // ---- Captions ----
        DrawText(ctx, $"input {net.LayerSizes[0]} (one-hot cell)", new Point(gridX, 6), 11, Brushes.Gray);
        for (var l = 1; l < layers - 1; l++)
            DrawText(ctx, $"hidden {net.LayerSizes[l]}", new Point(pos[l][0].X - 24, 6), 11, Brushes.Gray);
        DrawText(ctx, "Q-values", new Point(pos[^1][0].X - 20, 6), 11, Brushes.Gray);
        DrawText(ctx, "edge: green = positive weight, blue = negative, thickness = strength — click a maze cell to change the input",
            new Point(gridX, h - 15), 10, Brushes.Gray);
    }

    private static void DrawEdge(DrawingContext ctx, Point from, Point to, double weight, double maxAbs)
    {
        var t = Math.Abs(weight) / maxAbs;
        if (t < 0.03) return; // near-zero weights are just visual noise
        var color = weight >= 0 ? PositiveEdge : NegativeEdge;
        var pen = new Pen(new SolidColorBrush(color, 0.15 + 0.75 * t), 0.6 + 2.0 * t);
        ctx.DrawLine(pen, from, to);
    }

    private static void DrawText(DrawingContext ctx, string text, Point at, double size, IBrush brush)
    {
        var ft = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
            Typeface.Default, size, brush);
        ctx.DrawText(ft, at);
    }

    private static Color Lerp(Color a, Color b, double t)
    {
        t = Math.Clamp(t, 0, 1);
        return Color.FromRgb(
            (byte)(a.R + (b.R - a.R) * t),
            (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t));
    }
}
