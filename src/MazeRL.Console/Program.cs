using MazeRL.Core;

// A 10x10 maze: '#' wall, '.' open, 'S' start, 'G' goal.
var maze = Maze.FromText(
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

var env = new MazeEnvironment(maze);
var agent = new QLearningAgent(env.StateCount, env.ActionCount, seed: 42);
var trainer = new Trainer(env, agent);

Console.WriteLine("=== MazeRL: tabular Q-learning from scratch ===");
Console.WriteLine();
Console.WriteLine("The maze:");
PrintMaze(maze);
Console.WriteLine();

// Train in stages so we can watch it improve.
int[] checkpoints = [0, 10, 50, 100, 250, 500, 1000, 2000];
for (var i = 1; i < checkpoints.Length; i++)
{
    trainer.Train(checkpoints[i] - checkpoints[i - 1]);

    var (path, steps, reachedGoal) = trainer.TestRun();
    var status = reachedGoal ? $"solves it in {steps} steps" : "cannot solve it yet";
    Console.WriteLine($"After {checkpoints[i],5} episodes (epsilon {agent.Epsilon:F3}): greedy policy {status}");
}

Console.WriteLine();
Console.WriteLine("Learned policy (arrows = best action per cell, from the Q-table):");
PrintPolicy(maze, env, agent);
Console.WriteLine();

Console.WriteLine("Q-value heatmap (higher = closer to goal; '.' = never visited):");
PrintHeatmap(maze, env, agent);
Console.WriteLine();

var (finalPath, finalSteps, solved) = trainer.TestRun();
Console.WriteLine($"Final greedy run: {(solved ? "reached goal" : "FAILED")} in {finalSteps} steps.");
PrintPath(maze, env, finalPath);
Console.WriteLine();

// Save the brain — open this file to see that "learning" is just numbers.
var brainPath = Path.Combine(AppContext.BaseDirectory, "brain.json");
File.WriteAllText(brainPath, agent.ToJson());
Console.WriteLine($"Brain saved to: {brainPath}");

// Prove the save/load round-trip: a fresh agent loaded from JSON is instantly smart.
var loaded = QLearningAgent.FromJson(File.ReadAllText(brainPath), seed: 42);
var trainer2 = new Trainer(env, loaded);
var (_, loadedSteps, loadedSolved) = trainer2.TestRun();
Console.WriteLine($"Reloaded brain from JSON: {(loadedSolved ? "reached goal" : "FAILED")} in {loadedSteps} steps — no retraining needed.");

static void PrintMaze(Maze maze)
{
    for (var r = 0; r < maze.Rows; r++)
    {
        for (var c = 0; c < maze.Cols; c++)
        {
            if ((r, c) == maze.Start) Console.Write("S ");
            else if ((r, c) == maze.Goal) Console.Write("G ");
            else Console.Write(maze.IsWall(r, c) ? "# " : ". ");
        }
        Console.WriteLine();
    }
}

static void PrintPolicy(Maze maze, MazeEnvironment env, QLearningAgent agent)
{
    var arrows = new[] { '^', 'v', '<', '>' };
    for (var r = 0; r < maze.Rows; r++)
    {
        for (var c = 0; c < maze.Cols; c++)
        {
            if (maze.IsWall(r, c)) { Console.Write("# "); continue; }
            if ((r, c) == maze.Goal) { Console.Write("G "); continue; }

            var state = env.StateOf(r, c);
            // A cell whose Q-values are all zero has never been visited — no opinion yet.
            Console.Write(agent.BestValue(state) == 0 ? ". " : $"{arrows[agent.BestAction(state)]} ");
        }
        Console.WriteLine();
    }
}

static void PrintHeatmap(Maze maze, MazeEnvironment env, QLearningAgent agent)
{
    // Normalize best Q-values to 0-9 so the gradient toward the goal is visible.
    var max = double.MinValue;
    var min = double.MaxValue;
    for (var r = 0; r < maze.Rows; r++)
        for (var c = 0; c < maze.Cols; c++)
        {
            if (maze.IsWall(r, c)) continue;
            var v = agent.BestValue(env.StateOf(r, c));
            if (v > max) max = v;
            if (v < min) min = v;
        }

    for (var r = 0; r < maze.Rows; r++)
    {
        for (var c = 0; c < maze.Cols; c++)
        {
            if (maze.IsWall(r, c)) { Console.Write("# "); continue; }
            if ((r, c) == maze.Goal) { Console.Write("G "); continue; }

            var v = agent.BestValue(env.StateOf(r, c));
            if (v == 0) { Console.Write(". "); continue; }
            var scaled = (int)Math.Round((v - min) / (max - min) * 9);
            Console.Write($"{scaled} ");
        }
        Console.WriteLine();
    }
}

static void PrintPath(Maze maze, MazeEnvironment env, List<int> path)
{
    var onPath = new HashSet<int>(path);
    for (var r = 0; r < maze.Rows; r++)
    {
        for (var c = 0; c < maze.Cols; c++)
        {
            if ((r, c) == maze.Start) { Console.Write("S "); continue; }
            if ((r, c) == maze.Goal) { Console.Write("G "); continue; }
            if (maze.IsWall(r, c)) { Console.Write("# "); continue; }
            Console.Write(onPath.Contains(env.StateOf(r, c)) ? "o " : ". ");
        }
        Console.WriteLine();
    }
}
