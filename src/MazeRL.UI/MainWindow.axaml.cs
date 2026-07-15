using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using MazeRL.Core;

namespace MazeRL.UI;

public partial class MainWindow : Window
{
    private Maze _maze = null!;
    private MazeEnvironment _env = null!;
    private IAgent _agent = null!;
    private Trainer _trainer = null!;

    private readonly DispatcherTimer _trainTimer;
    private readonly DispatcherTimer _ballTimer;
    private List<(int Row, int Col)> _ballPath = [];
    private int _ballIndex;
    private int _bestSteps = int.MaxValue;
    private bool _switchingAgent; // guards the agent radios while we set them from code

    public MainWindow()
    {
        InitializeComponent();

        _trainTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(33) };
        _trainTimer.Tick += TrainTick;
        _ballTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(70) };
        _ballTimer.Tick += BallTick;

        Board.CellSelected += OnCellSelected;

        ModeWalls.IsCheckedChanged += (_, _) => { if (ModeWalls.IsChecked == true) Board.Mode = EditMode.Walls; };
        ModeStart.IsCheckedChanged += (_, _) => { if (ModeStart.IsChecked == true) Board.Mode = EditMode.Start; };
        ModeGoal.IsCheckedChanged += (_, _) => { if (ModeGoal.IsChecked == true) Board.Mode = EditMode.Goal; };

        HeatmapCheck.IsCheckedChanged += (_, _) => { Board.ShowHeatmap = HeatmapCheck.IsChecked == true; Board.InvalidateVisual(); };
        ArrowsCheck.IsCheckedChanged += (_, _) => { Board.ShowArrows = ArrowsCheck.IsChecked == true; Board.InvalidateVisual(); };

        // Switching agent type = a different kind of brain → start it fresh.
        AgentQTable.IsCheckedChanged += OnAgentTypeChanged;
        AgentDqn.IsCheckedChanged += OnAgentTypeChanged;

        SpeedSlider.PropertyChanged += (_, e) =>
        {
            if (e.Property == RangeBase.ValueProperty)
                SpeedLabel.Text = $"Speed: {(int)SpeedSlider.Value} episodes per tick";
        };

        SetupWorld(DefaultMaze());
    }

    private static Maze DefaultMaze() => Maze.FromText(
        "S...#.....",
        ".##.#.###.",
        ".#..#...#.",
        ".#.###.##.",
        ".#.....#..",
        ".#####.#.#",
        ".....#.#..",
        "####.#.#.#",
        "...#...#.#",
        ".#...#...G");

    private IAgent CreateAgent() => AgentDqn.IsChecked == true
        ? new DqnAgent(_env.StateCount, _env.ActionCount)
        : new QLearningAgent(_env.StateCount, _env.ActionCount);

    /// <summary>Wire a maze into a fresh environment/agent/trainer and reset all views.</summary>
    private void SetupWorld(Maze maze, IAgent? keepAgent = null)
    {
        _maze = maze;
        _env = new MazeEnvironment(maze);
        _agent = keepAgent ?? CreateAgent();
        _trainer = new Trainer(_env, _agent);
        _bestSteps = int.MaxValue;

        Board.Maze = _maze;
        Board.Env = _env;
        Board.Agent = _agent;
        Board.Ball = null;
        Board.PathCells = null;
        Board.SelectedCell = null;
        Board.InvalidateVisual();

        // The network view only has something to show when the brain is a network.
        NetView.Agent = _agent as DqnAgent;
        NetView.Env = _env;
        NetView.State = _env.StateOf(_maze.Start.Row, _maze.Start.Col);
        NetView.InvalidateVisual();

        StepsChart.Clear();
        LossChart.Clear();
        UpdateStats(null);
    }

    private void OnAgentTypeChanged(object? sender, RoutedEventArgs e)
    {
        if (_switchingAgent || sender is not RadioButton { IsChecked: true }) return;
        StopTraining();
        StopBallAnimation();
        SetupWorld(_maze); // same maze, fresh brain of the chosen kind
    }

    /// <summary>Set the agent radios from code (e.g. after loading a brain) without resetting the world.</summary>
    private void SyncAgentRadios()
    {
        _switchingAgent = true;
        AgentDqn.IsChecked = _agent is DqnAgent;
        AgentQTable.IsChecked = _agent is QLearningAgent;
        _switchingAgent = false;
    }

    private void OnSettingsToggled(object? sender, RoutedEventArgs e) =>
        SettingsPanel.IsVisible = SettingsToggle.IsChecked == true;

    // ---- Training ----

    private void OnStartPause(object? sender, RoutedEventArgs e)
    {
        if (_trainTimer.IsEnabled)
        {
            StopTraining("▶ Resume training");
        }
        else
        {
            StopBallAnimation();
            _trainTimer.Start();
            StartPauseButton.Content = "⏸ Pause training";
        }
    }

    private void StopTraining(string buttonText = "▶ Start training")
    {
        _trainTimer.Stop();
        StartPauseButton.Content = buttonText;
    }

    private void TrainTick(object? sender, EventArgs e)
    {
        // Run up to the requested number of episodes, but never hog the UI thread:
        // the neural net costs much more per episode than the table, especially early on.
        var sw = Stopwatch.StartNew();
        var maxEpisodes = (int)SpeedSlider.Value;
        var steps = new List<double>(maxEpisodes);
        EpisodeResult last = default;

        for (var i = 0; i < maxEpisodes && sw.ElapsedMilliseconds < 25; i++)
        {
            last = _trainer.RunEpisode();
            steps.Add(last.Steps);
            if (last.ReachedGoal && last.Steps < _bestSteps)
                _bestSteps = last.Steps;
        }

        StepsChart.AddPoints(steps);
        if (_agent is DqnAgent dqn && dqn.TrainSteps > 0)
            LossChart.AddPoints([dqn.AverageLoss]);

        UpdateStats(last);
        Board.InvalidateVisual();
        NetView.InvalidateVisual(); // weights just changed — redraw the network
    }

    private void UpdateStats(EpisodeResult? last)
    {
        var best = _bestSteps == int.MaxValue ? "-" : _bestSteps.ToString();
        var text =
            $"Episodes: {_trainer.EpisodesCompleted}\n" +
            $"Epsilon:  {_agent.Epsilon:F3}  (exploration)\n" +
            $"Last:     {(last is { } l ? $"{l.Steps} steps, reward {l.TotalReward:0}" : "-")}\n" +
            $"Best:     {best} steps";

        if (_agent is DqnAgent dqn)
            text += dqn.TrainSteps > 0
                ? $"\nLoss:     {dqn.AverageLoss:F3}\n" +
                  $"Memory:   {dqn.ReplayCount} moves, {dqn.TrainSteps} train steps"
                : $"\nLoss:     - (warming up: {dqn.ReplayCount}/{dqn.WarmupSize} moves remembered)";

        StatsLabel.Text = text;
    }

    private void OnResetBrain(object? sender, RoutedEventArgs e)
    {
        SetupWorld(_maze); // fresh agent, same maze
    }

    // ---- Test run (greedy, no learning) ----

    private void OnTestRun(object? sender, RoutedEventArgs e)
    {
        if (_trainTimer.IsEnabled) OnStartPause(sender, e); // pause training while we watch

        var (path, _, reachedGoal) = _trainer.TestRun(maxSteps: _maze.Rows * _maze.Cols * 4);
        _ballPath = path.Select(s => _env.CellOf(s)).ToList();
        _ballIndex = 0;

        Board.PathCells = reachedGoal ? [.. path] : null;
        _ballTimer.Start();
    }

    private void BallTick(object? sender, EventArgs e)
    {
        if (_ballIndex >= _ballPath.Count)
        {
            StopBallAnimation(keepPath: true);
            return;
        }
        var cell = _ballPath[_ballIndex++];
        Board.Ball = cell;
        Board.InvalidateVisual();

        // Feed the ball's cell into the network view — watch the net "think" as it walks.
        NetView.State = _env.StateOf(cell.Row, cell.Col);
        NetView.InvalidateVisual();
    }

    private void StopBallAnimation(bool keepPath = false)
    {
        _ballTimer.Stop();
        Board.Ball = null;
        if (!keepPath) Board.PathCells = null;
        Board.InvalidateVisual();
    }

    // ---- Inspector ----

    private void OnCellSelected(int row, int col)
    {
        var s = _env.StateOf(row, col);
        var q = _agent.QValues(s);
        InspectorLabel.Text =
            $"Cell ({row},{col})  state #{s}\n" +
            $"  Up:    {q[MazeEnvironment.Up],10:F3}\n" +
            $"  Down:  {q[MazeEnvironment.Down],10:F3}\n" +
            $"  Left:  {q[MazeEnvironment.Left],10:F3}\n" +
            $"  Right: {q[MazeEnvironment.Right],10:F3}\n" +
            $"  → best: {MazeEnvironment.ActionNames[_agent.BestAction(s)]}";

        NetView.State = s;
        NetView.InvalidateVisual();
    }

    // ---- Maze editing ----

    private void OnNewMaze(object? sender, RoutedEventArgs e) =>
        ReplaceMaze(new Maze((int)(RowsUpDown.Value ?? 10), (int)(ColsUpDown.Value ?? 10)));

    private void OnRandomMaze(object? sender, RoutedEventArgs e) =>
        ReplaceMaze(MazeGenerator.Generate((int)(RowsUpDown.Value ?? 10), (int)(ColsUpDown.Value ?? 10)));

    private void ReplaceMaze(Maze maze)
    {
        StopTraining();
        StopBallAnimation();
        SetupWorld(maze);
    }

    private void OnClearWalls(object? sender, RoutedEventArgs e)
    {
        for (var r = 0; r < _maze.Rows; r++)
            for (var c = 0; c < _maze.Cols; c++)
                _maze.SetWall(r, c, false);
        Board.InvalidateVisual();
        NetView.InvalidateVisual();
    }

    // ---- Save / load ----

    private static readonly FilePickerFileType JsonType = new("JSON") { Patterns = ["*.json"] };

    private async void OnSaveMaze(object? sender, RoutedEventArgs e)
    {
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save maze",
            SuggestedFileName = "maze.json",
            FileTypeChoices = [JsonType],
        });
        if (file is not null)
            await File.WriteAllTextAsync(file.Path.LocalPath, _maze.ToJson());
    }

    private async void OnLoadMaze(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Load maze",
            FileTypeFilter = [JsonType],
        });
        if (files.Count == 0) return;

        var maze = Maze.FromJson(await File.ReadAllTextAsync(files[0].Path.LocalPath));
        StopTraining();
        StopBallAnimation();

        // Same grid size → keep the brain (fun: watch it adapt); different size → fresh brain.
        var sameSize = maze.Rows == _maze.Rows && maze.Cols == _maze.Cols;
        SetupWorld(maze, sameSize ? _agent : null);
    }

    private async void OnSaveBrain(object? sender, RoutedEventArgs e)
    {
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = _agent is DqnAgent ? "Save brain (network weights)" : "Save brain (Q-table)",
            SuggestedFileName = "brain.json",
            FileTypeChoices = [JsonType],
        });
        if (file is not null)
            await File.WriteAllTextAsync(file.Path.LocalPath, _agent.ToJson());
    }

    private async void OnLoadBrain(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Load brain",
            FileTypeFilter = [JsonType],
        });
        if (files.Count == 0) return;

        var json = await File.ReadAllTextAsync(files[0].Path.LocalPath);
        // A DQN brain has layerSizes; a Q-table brain has the table itself.
        IAgent agent = json.Contains("\"layerSizes\"")
            ? DqnAgent.FromJson(json)
            : QLearningAgent.FromJson(json);

        if (agent.StateCount != _env.StateCount)
        {
            InspectorLabel.Text = $"Brain has {agent.StateCount} states but this maze has {_env.StateCount} — size mismatch.";
            return;
        }

        StopTraining();
        StopBallAnimation();
        SetupWorld(_maze, agent);
        SyncAgentRadios();
    }
}
