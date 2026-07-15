using System.Text.Json;

namespace MazeRL.Core;

/// <summary>
/// Deep Q-Network agent: same job as the Q-table, but the table is replaced by
/// a small neural network that maps state → 4 Q-values. Uses the two classic
/// DQN stabilizers:
///  • experience replay — learn from a shuffled memory of past moves, not just the last one;
///  • a target network — a frozen copy of the net provides the learning targets,
///    synced only occasionally, so the net isn't chasing its own moving output.
/// </summary>
public class DqnAgent : IAgent
{
    /// <summary>The online network — the brain being trained (and visualized).</summary>
    public NeuralNetwork Network { get; }

    private readonly NeuralNetwork _targetNetwork;

    public int StateCount { get; }
    public int ActionCount { get; }

    public double Epsilon { get; set; } = 1.0;
    public double DiscountFactor { get; set; } = 0.95;

    /// <summary>A fresh net outputs random Q-values everywhere — only claim knowledge after the first gradient step.</summary>
    public bool HasLearned { get; private set; }

    /// <summary>How many remembered transitions each gradient step learns from.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>Do a gradient step every N environment steps (1 = every step).</summary>
    public int TrainEvery { get; set; } = 2;

    /// <summary>Copy online → target network every N gradient steps.</summary>
    public int TargetSyncEvery { get; set; } = 250;

    /// <summary>Don't start training until the replay memory has at least this many samples.</summary>
    public int WarmupSize { get; set; } = 200;

    /// <summary>Exponential moving average of the batch loss — what the loss chart shows.</summary>
    public double AverageLoss { get; private set; }

    /// <summary>How many gradient steps have been taken so far.</summary>
    public int TrainSteps { get; private set; }

    public int ReplayCount => _replayCount;

    // Replay memory: a fixed-size ring buffer of (s, a, r, s', done).
    private readonly (int State, int Action, double Reward, int NextState, bool Done)[] _replay;
    private int _replayNext, _replayCount;

    private int _envSteps;
    private readonly Random _random;

    public DqnAgent(int stateCount, int actionCount, int[]? hiddenLayers = null,
        int replayCapacity = 10_000, int? seed = null)
    {
        StateCount = stateCount;
        ActionCount = actionCount;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        hiddenLayers ??= [16, 16];
        int[] sizes = [stateCount, .. hiddenLayers, actionCount];
        Network = new NeuralNetwork(sizes, seed);
        _targetNetwork = new NeuralNetwork(sizes, seed);
        _targetNetwork.CopyFrom(Network);

        _replay = new (int, int, double, int, bool)[replayCapacity];
    }

    /// <summary>The state the network actually sees: a one-hot vector, one slot per maze cell.</summary>
    public double[] Encode(int state)
    {
        var x = new double[StateCount];
        x[state] = 1.0;
        return x;
    }

    public int ChooseAction(int state)
    {
        if (_random.NextDouble() < Epsilon)
            return _random.Next(ActionCount);
        return BestAction(state);
    }

    public int BestAction(int state) => ArgMax(Network.Predict(Encode(state)));

    public double BestValue(int state) => Network.Predict(Encode(state)).Max();

    public double[] QValues(int state) => Network.Predict(Encode(state));

    /// <summary>
    /// Learning, DQN style: remember the transition, then train on a random
    /// batch of memories. The target for each memory is the same formula the
    /// Q-table uses — r + γ·max Q(s′) — but computed by the frozen target net.
    /// </summary>
    public void Update(int state, int action, double reward, int nextState, bool done)
    {
        _replay[_replayNext] = (state, action, reward, nextState, done);
        _replayNext = (_replayNext + 1) % _replay.Length;
        if (_replayCount < _replay.Length) _replayCount++;

        _envSteps++;
        if (_replayCount < WarmupSize || _envSteps % TrainEvery != 0)
            return;

        TrainOnBatch();
    }

    private void TrainOnBatch()
    {
        var n = Math.Min(BatchSize, _replayCount);
        var inputs = new double[n][];
        var actions = new int[n];
        var targets = new double[n];

        for (var i = 0; i < n; i++)
        {
            var (s, a, r, s2, done) = _replay[_random.Next(_replayCount)];
            inputs[i] = Encode(s);
            actions[i] = a;
            // If the episode ended there is no future — the target is just the reward.
            var future = done ? 0.0 : _targetNetwork.Predict(Encode(s2)).Max();
            targets[i] = r + DiscountFactor * future;
        }

        var loss = Network.TrainBatch(inputs, actions, targets);
        AverageLoss = TrainSteps == 0 ? loss : 0.98 * AverageLoss + 0.02 * loss;
        TrainSteps++;
        HasLearned = true;

        if (TrainSteps % TargetSyncEvery == 0)
            _targetNetwork.CopyFrom(Network);
    }

    private static int ArgMax(double[] values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }

    // ---- Serialization: the "brain" is the network's weights and biases ----

    public string ToJson()
    {
        var dto = new BrainDto
        {
            StateCount = StateCount,
            ActionCount = ActionCount,
            LayerSizes = Network.LayerSizes,
            Epsilon = Epsilon,
            DiscountFactor = DiscountFactor,
            Weights = Network.Weights.Select(l => l.Select(row => row.ToArray()).ToArray()).ToArray(),
            Biases = Network.Biases.Select(b => b.ToArray()).ToArray(),
        };
        return JsonSerializer.Serialize(dto, JsonOptions);
    }

    public static DqnAgent FromJson(string json, int? seed = null)
    {
        var dto = JsonSerializer.Deserialize<BrainDto>(json, JsonOptions)
                  ?? throw new ArgumentException("Invalid brain JSON.");
        var hidden = dto.LayerSizes[1..^1];
        var agent = new DqnAgent(dto.StateCount, dto.ActionCount, hidden, seed: seed)
        {
            Epsilon = dto.Epsilon,
            DiscountFactor = dto.DiscountFactor,
            HasLearned = true, // a saved brain is a trained brain
        };
        for (var l = 0; l < dto.Weights.Length; l++)
        {
            for (var to = 0; to < dto.Weights[l].Length; to++)
            {
                Array.Copy(dto.Weights[l][to], agent.Network.Weights[l][to], dto.Weights[l][to].Length);
                agent.Network.Biases[l][to] = dto.Biases[l][to];
            }
        }
        agent._targetNetwork.CopyFrom(agent.Network);
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
        public int[] LayerSizes { get; set; } = [];
        public double Epsilon { get; set; }
        public double DiscountFactor { get; set; }
        public double[][][] Weights { get; set; } = [];
        public double[][] Biases { get; set; } = [];
    }
}
