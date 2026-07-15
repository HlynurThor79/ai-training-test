using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace MazeRL.UI;

/// <summary>
/// Minimal line chart for steps-per-episode. Keeps the last MaxPoints values
/// and draws them as a polyline — enough to see the learning curve fall.
/// </summary>
public class LineChart : Control
{
    private readonly List<double> _values = [];

    public int MaxPoints { get; set; } = 300;

    /// <summary>Number format for the min/max/last label ("0" for steps, "0.###" for loss).</summary>
    public string LabelFormat { get; set; } = "0";

    public void AddPoints(IEnumerable<double> values)
    {
        _values.AddRange(values);
        if (_values.Count > MaxPoints)
            _values.RemoveRange(0, _values.Count - MaxPoints);
        InvalidateVisual();
    }

    public void Clear()
    {
        _values.Clear();
        InvalidateVisual();
    }

    public override void Render(DrawingContext ctx)
    {
        var bounds = new Rect(Bounds.Size);
        ctx.FillRectangle(new SolidColorBrush(Color.FromRgb(34, 34, 42)), bounds);

        if (_values.Count < 2) return;

        var min = _values.Min();
        var max = _values.Max();
        var range = Math.Max(1, max - min);

        var pen = new Pen(new SolidColorBrush(Color.FromRgb(46, 204, 113)), 1.5);
        var w = bounds.Width;
        var h = bounds.Height - 6;

        Point PointAt(int i) => new(
            i * w / (_values.Count - 1),
            3 + h - (_values[i] - min) / range * h);

        for (var i = 1; i < _values.Count; i++)
            ctx.DrawLine(pen, PointAt(i - 1), PointAt(i));

        var label = new FormattedText(
            $"min {min.ToString(LabelFormat, CultureInfo.InvariantCulture)}  " +
            $"max {max.ToString(LabelFormat, CultureInfo.InvariantCulture)}  " +
            $"last {_values[^1].ToString(LabelFormat, CultureInfo.InvariantCulture)}",
            CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
            Typeface.Default, 11, Brushes.Gray);
        ctx.DrawText(label, new Point(6, 4));
    }
}
