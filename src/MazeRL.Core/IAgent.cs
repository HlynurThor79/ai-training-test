namespace MazeRL.Core;

/// <summary>
/// What the trainer and UI need from any learning agent.
/// Two implementations exist: the tabular QLearningAgent (a lookup table)
/// and the DqnAgent (a neural network that *approximates* the same table).
/// </summary>
public interface IAgent
{
    int StateCount { get; }
    int ActionCount { get; }

    /// <summary>Exploration rate ε: probability of trying a random action instead of the best-known one.</summary>
    double Epsilon { get; set; }

    /// <summary>
    /// True once the agent has actually learned something; false for a fresh brain.
    /// A fresh Q-table is all zeros, but a fresh neural net is random weights that
    /// still produce Q-values everywhere — the UI uses this to hide that noise.
    /// </summary>
    bool HasLearned { get; }

    /// <summary>Epsilon-greedy choice used during training.</summary>
    int ChooseAction(int state);

    /// <summary>Purely greedy choice — used when testing what has actually been learned.</summary>
    int BestAction(int state);

    /// <summary>Value of the best action in a state (drives the heatmap).</summary>
    double BestValue(int state);

    /// <summary>Current Q-value estimate for every action in a state (drives the inspector).</summary>
    double[] QValues(int state);

    /// <summary>Learn from one experienced transition.</summary>
    void Update(int state, int action, double reward, int nextState, bool done);

    /// <summary>Serialize the brain — a table of numbers or a bag of weights, but always just numbers.</summary>
    string ToJson();
}
