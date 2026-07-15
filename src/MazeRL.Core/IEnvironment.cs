namespace MazeRL.Core;

/// <summary>
/// The contract every "world" implements. The agent only ever sees this:
/// states are integers, actions are integers, and stepping returns a reward.
/// Swap in a slippery maze, a monster maze, anything — the agent code never changes.
/// </summary>
public interface IEnvironment
{
    /// <summary>Total number of distinct states (for sizing the Q-table).</summary>
    int StateCount { get; }

    /// <summary>Number of actions available in every state.</summary>
    int ActionCount { get; }

    /// <summary>Put the world back to its starting configuration. Returns the initial state.</summary>
    int Reset();

    /// <summary>Take an action from the current state.</summary>
    StepResult Step(int action);
}

/// <summary>What the world hands back after every action.</summary>
public readonly record struct StepResult(int NextState, double Reward, bool Done);
