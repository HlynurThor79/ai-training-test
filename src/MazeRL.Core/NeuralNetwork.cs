namespace MazeRL.Core;

/// <summary>
/// A tiny fully-connected neural network written from scratch — no libraries.
/// ReLU hidden layers, linear output, trained by backpropagation with Adam.
///
/// Shapes: Weights[l][to][from] connects layer l to layer l+1;
/// Biases[l][to] is added to node "to" of layer l+1.
/// </summary>
public class NeuralNetwork
{
    public int[] LayerSizes { get; }
    public double[][][] Weights { get; }
    public double[][] Biases { get; }

    /// <summary>Adam step size. The only knob you usually need to touch.</summary>
    public double LearningRate { get; set; } = 0.002;

    // Adam state: first/second moment estimates per parameter, and the step counter.
    private readonly double[][][] _mW, _vW;
    private readonly double[][] _mB, _vB;
    private int _adamStep;
    private const double Beta1 = 0.9, Beta2 = 0.999, AdamEps = 1e-8;

    private readonly Random _random;

    public NeuralNetwork(int[] layerSizes, int? seed = null)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("Need at least an input and an output layer.");

        LayerSizes = layerSizes;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        var links = layerSizes.Length - 1;
        Weights = new double[links][][];
        Biases = new double[links][];
        _mW = new double[links][][];
        _vW = new double[links][][];
        _mB = new double[links][];
        _vB = new double[links][];

        for (var l = 0; l < links; l++)
        {
            var fanIn = layerSizes[l];
            var fanOut = layerSizes[l + 1];
            // He initialization: keeps signal variance stable through ReLU layers.
            var scale = Math.Sqrt(2.0 / fanIn);

            Weights[l] = new double[fanOut][];
            _mW[l] = new double[fanOut][];
            _vW[l] = new double[fanOut][];
            Biases[l] = new double[fanOut];
            _mB[l] = new double[fanOut];
            _vB[l] = new double[fanOut];

            for (var to = 0; to < fanOut; to++)
            {
                Weights[l][to] = new double[fanIn];
                _mW[l][to] = new double[fanIn];
                _vW[l][to] = new double[fanIn];
                for (var from = 0; from < fanIn; from++)
                    Weights[l][to][from] = NextGaussian() * scale;
            }
        }
    }

    /// <summary>
    /// Run the network forward. Returns the activations of EVERY layer
    /// (index 0 = input, last = output) so callers can visualize the inside.
    /// Hidden layers use ReLU; the output layer is linear (Q-values can be negative).
    /// </summary>
    public double[][] Forward(double[] input)
    {
        var acts = new double[LayerSizes.Length][];
        acts[0] = input;

        for (var l = 0; l < Weights.Length; l++)
        {
            var prev = acts[l];
            var next = new double[LayerSizes[l + 1]];
            var isOutput = l == Weights.Length - 1;

            for (var to = 0; to < next.Length; to++)
            {
                var sum = Biases[l][to];
                var w = Weights[l][to];
                for (var from = 0; from < prev.Length; from++)
                    sum += w[from] * prev[from];
                next[to] = isOutput ? sum : Math.Max(0, sum); // ReLU on hidden layers
            }
            acts[l + 1] = next;
        }
        return acts;
    }

    /// <summary>Convenience: just the output layer.</summary>
    public double[] Predict(double[] input) => Forward(input)[^1];

    /// <summary>
    /// One gradient step on a batch. For each sample only ONE output node has a
    /// training target (the action that was actually taken) — the others produce
    /// no gradient, exactly like DQN. Returns the mean squared error of the batch.
    /// </summary>
    public double TrainBatch(double[][] inputs, int[] actions, double[] targets)
    {
        var links = Weights.Length;

        // Gradient accumulators, same shapes as the parameters.
        var gradW = new double[links][][];
        var gradB = new double[links][];
        for (var l = 0; l < links; l++)
        {
            gradW[l] = new double[LayerSizes[l + 1]][];
            gradB[l] = new double[LayerSizes[l + 1]];
            for (var to = 0; to < LayerSizes[l + 1]; to++)
                gradW[l][to] = new double[LayerSizes[l]];
        }

        var totalLoss = 0.0;

        for (var n = 0; n < inputs.Length; n++)
        {
            var acts = Forward(inputs[n]);
            var output = acts[^1];

            var error = output[actions[n]] - targets[n];
            totalLoss += error * error;

            // delta = ∂loss/∂(node pre-activation), walked backward layer by layer.
            var delta = new double[output.Length];
            delta[actions[n]] = error; // d(½e²)/d(out) = e; only the taken action learns

            for (var l = links - 1; l >= 0; l--)
            {
                var prev = acts[l];
                var prevDelta = new double[prev.Length];

                for (var to = 0; to < delta.Length; to++)
                {
                    if (delta[to] == 0) continue; // sparse deltas make the output layer cheap
                    var d = delta[to];
                    gradB[l][to] += d;
                    var w = Weights[l][to];
                    var g = gradW[l][to];
                    for (var from = 0; from < prev.Length; from++)
                    {
                        g[from] += d * prev[from];
                        prevDelta[from] += d * w[from];
                    }
                }

                // ReLU derivative: gradient only flows through nodes that were active.
                if (l > 0)
                    for (var i = 0; i < prevDelta.Length; i++)
                        if (acts[l][i] <= 0)
                            prevDelta[i] = 0;

                delta = prevDelta;
            }
        }

        ApplyAdam(gradW, gradB, inputs.Length);
        return totalLoss / inputs.Length;
    }

    private void ApplyAdam(double[][][] gradW, double[][] gradB, int batchSize)
    {
        _adamStep++;
        // Bias-corrected learning rate (standard Adam trick, folded into one factor).
        var alpha = LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, _adamStep)) / (1 - Math.Pow(Beta1, _adamStep));

        for (var l = 0; l < Weights.Length; l++)
        {
            for (var to = 0; to < Weights[l].Length; to++)
            {
                for (var from = 0; from < Weights[l][to].Length; from++)
                {
                    var g = gradW[l][to][from] / batchSize;
                    var m = _mW[l][to][from] = Beta1 * _mW[l][to][from] + (1 - Beta1) * g;
                    var v = _vW[l][to][from] = Beta2 * _vW[l][to][from] + (1 - Beta2) * g * g;
                    Weights[l][to][from] -= alpha * m / (Math.Sqrt(v) + AdamEps);
                }

                var gb = gradB[l][to] / batchSize;
                var mb = _mB[l][to] = Beta1 * _mB[l][to] + (1 - Beta1) * gb;
                var vb = _vB[l][to] = Beta2 * _vB[l][to] + (1 - Beta2) * gb * gb;
                Biases[l][to] -= alpha * mb / (Math.Sqrt(vb) + AdamEps);
            }
        }
    }

    /// <summary>Copy all weights from another same-shaped network (target-network sync).</summary>
    public void CopyFrom(NeuralNetwork other)
    {
        for (var l = 0; l < Weights.Length; l++)
        {
            for (var to = 0; to < Weights[l].Length; to++)
            {
                Array.Copy(other.Weights[l][to], Weights[l][to], Weights[l][to].Length);
                Biases[l][to] = other.Biases[l][to];
            }
        }
    }

    private double NextGaussian()
    {
        // Box–Muller: turn two uniform randoms into one normally-distributed one.
        var u1 = 1.0 - _random.NextDouble();
        var u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
