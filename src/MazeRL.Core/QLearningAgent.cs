using System.Text.Json;

namespace MazeRL.Core;

/// <summary>
/// Tabular Q-learning agent. The entire "intelligence" is the Q-table:
/// for every (state, action) pair, one number estimating the total future
/// reward of taking that action from that state. Learning = nudging those
/// numbers toward reality after every single step.
/// </summary>
public class QLearningAgent : IAgent
{
    /// <summary>Q[state, action] — the brain. Starts at all zeros: total ignorance.</summary>
    public double[,] Q { get; private set; }

    public int StateCount { get; }
    public int ActionCount { get; }

    /// <summary>Learning rate α: how far each update moves the old estimate toward the new evidence.</summary>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>Discount factor γ: how much future reward is worth compared to immediate reward.</summary>
    public double DiscountFactor { get; set; } = 0.95;

    /// <summary>Exploration rate ε: probability of trying a random action instead of the best-known one.</summary>
    public double Epsilon { get; set; } = 1.0;

    public bool HasLearned { get; private set; }

    private readonly Random _random;

    public QLearningAgent(int stateCount, int actionCount, int? seed = null)
    {
        StateCount = stateCount;
        ActionCount = actionCount;
        Q = new double[stateCount, actionCount];
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Epsilon-greedy: usually exploit what we know, sometimes explore at random.
    /// Without the random part the agent can get stuck on the first path it finds.
    /// </summary>
    public int ChooseAction(int state)
    {
        if (_random.NextDouble() < Epsilon)
            return _random.Next(ActionCount);
        return BestAction(state);
    }

    /// <summary>Purely greedy choice — used when testing what has actually been learned.</summary>
    public int BestAction(int state)
    {
        var best = 0;
        for (var a = 1; a < ActionCount; a++)
            if (Q[state, a] > Q[state, best])
                best = a;
        return best;
    }

    public double BestValue(int state)
    {
        var best = Q[state, 0];
        for (var a = 1; a < ActionCount; a++)
            if (Q[state, a] > best)
                best = Q[state, a];
        return best;
    }

    public double[] QValues(int state)
    {
        var values = new double[ActionCount];
        for (var a = 0; a < ActionCount; a++)
            values[a] = Q[state, a];
        return values;
    }

    /// <summary>
    /// The Q-learning update — the one line of math the whole project exists to teach:
    ///   Q(s,a) += α · (r + γ·max_a' Q(s',a') − Q(s,a))
    /// "Move my old guess toward (what I just got) + (the best I think I can do from where I landed)."
    /// </summary>
    public void Update(int state, int action, double reward, int nextState, bool done)
    {
        // If the episode ended there is no future — the target is just the reward.
        var futureValue = done ? 0.0 : BestValue(nextState);
        var target = reward + DiscountFactor * futureValue;
        Q[state, action] += LearningRate * (target - Q[state, action]);
        HasLearned = true;
    }

    public void ResetKnowledge()
    {
        Q = new double[StateCount, ActionCount];
        HasLearned = false;
    }

    /// <summary>
    /// Serialize the brain to JSON. Open the file: the learning is literally just numbers.
    /// </summary>
    public string ToJson()
    {
        var rows = new double[StateCount][];
        for (var s = 0; s < StateCount; s++)
        {
            rows[s] = new double[ActionCount];
            for (var a = 0; a < ActionCount; a++)
                rows[s][a] = Q[s, a];
        }
        var dto = new BrainDto
        {
            StateCount = StateCount,
            ActionCount = ActionCount,
            LearningRate = LearningRate,
            DiscountFactor = DiscountFactor,
            Epsilon = Epsilon,
            Q = rows,
        };
        return JsonSerializer.Serialize(dto, JsonOptions);
    }

    public static QLearningAgent FromJson(string json, int? seed = null)
    {
        var dto = JsonSerializer.Deserialize<BrainDto>(json, JsonOptions)
                  ?? throw new ArgumentException("Invalid brain JSON.");
        var agent = new QLearningAgent(dto.StateCount, dto.ActionCount, seed)
        {
            LearningRate = dto.LearningRate,
            DiscountFactor = dto.DiscountFactor,
            Epsilon = dto.Epsilon,
            HasLearned = true, // a saved brain is a trained brain
        };
        for (var s = 0; s < dto.StateCount; s++)
            for (var a = 0; a < dto.ActionCount; a++)
                agent.Q[s, a] = dto.Q[s][a];
        return agent;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    private class BrainDto
    {
        public int StateCount { get; set; }
        public int ActionCount { get; set; }
        public double LearningRate { get; set; }
        public double DiscountFactor { get; set; }
        public double Epsilon { get; set; }
        public double[][] Q { get; set; } = [];
    }
}
