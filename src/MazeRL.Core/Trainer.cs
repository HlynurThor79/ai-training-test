namespace MazeRL.Core;

/// <summary>Summary of one training episode — what the UI charts over time.</summary>
public readonly record struct EpisodeResult(int Episode, int Steps, double TotalReward, bool ReachedGoal, double Epsilon);

/// <summary>
/// Runs the training loop: episodes of (choose action → step → learn) until done.
/// Raises an event after every episode so a UI can chart progress or redraw heatmaps.
/// </summary>
public class Trainer
{
    public IEnvironment Environment { get; }
    public IAgent Agent { get; }

    /// <summary>Safety valve: end an episode after this many steps even without reaching the goal.</summary>
    public int MaxStepsPerEpisode { get; set; } = 1000;

    /// <summary>After each episode, epsilon is multiplied by this — explore a lot early, trust knowledge later.</summary>
    public double EpsilonDecay { get; set; } = 0.995;

    /// <summary>Never stop exploring entirely; the world might change (or you might edit the maze).</summary>
    public double MinEpsilon { get; set; } = 0.05;

    public int EpisodesCompleted { get; private set; }

    public event Action<EpisodeResult>? EpisodeFinished;

    public Trainer(IEnvironment environment, IAgent agent)
    {
        Environment = environment;
        Agent = agent;
    }

    /// <summary>Run one full episode with learning enabled.</summary>
    public EpisodeResult RunEpisode()
    {
        var state = Environment.Reset();
        var totalReward = 0.0;
        var steps = 0;
        var reachedGoal = false;

        while (steps < MaxStepsPerEpisode)
        {
            var action = Agent.ChooseAction(state);
            var (nextState, reward, done) = Environment.Step(action);

            Agent.Update(state, action, reward, nextState, done);

            totalReward += reward;
            state = nextState;
            steps++;

            if (done)
            {
                reachedGoal = true;
                break;
            }
        }

        EpisodesCompleted++;
        Agent.Epsilon = Math.Max(MinEpsilon, Agent.Epsilon * EpsilonDecay);

        var result = new EpisodeResult(EpisodesCompleted, steps, totalReward, reachedGoal, Agent.Epsilon);
        EpisodeFinished?.Invoke(result);
        return result;
    }

    public void Train(int episodes)
    {
        for (var i = 0; i < episodes; i++)
            RunEpisode();
    }

    /// <summary>
    /// Run one episode greedily with NO exploration and NO learning —
    /// pure "show me what you know". Returns the path taken.
    /// </summary>
    public (List<int> Path, int Steps, bool ReachedGoal) TestRun(int maxSteps = 1000)
    {
        var state = Environment.Reset();
        var path = new List<int> { state };
        var steps = 0;

        while (steps < maxSteps)
        {
            var action = Agent.BestAction(state);
            var (nextState, _, done) = Environment.Step(action);
            path.Add(nextState);
            state = nextState;
            steps++;

            if (done)
                return (path, steps, true);

            // Greedy policy can loop forever early in training; bail if we revisit without progress.
            if (nextState == path[^2] && path.Count > 2)
                return (path, steps, false);
        }

        return (path, steps, false);
    }
}
