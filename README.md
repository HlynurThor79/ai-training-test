# MazeRL — reinforcement learning from scratch in C#

A ball learns to solve a maze using reinforcement learning, written from the ground up — no ML libraries. Two interchangeable brains: **tabular Q-learning** (a lookup table) and a **DQN** (a tiny hand-written neural network trained by backpropagation). The goal is to understand how an AI agent actually learns.

## Run it

**Console demo** — trains and prints the learning story to the terminal:

```bash
dotnet run --project src/MazeRL.Console          # tabular Q-learning
dotnet run --project src/MazeRL.Console -- --dqn # neural-network (DQN) agent
```

You'll watch the agent go from random wandering to solving the maze optimally, then see its learned policy (arrows), a Q-value heatmap, and the final path. The trained "brain" is saved to `brain.json` — open it and see that the learning is literally just numbers.

**GUI playground** (Avalonia — runs on Linux/WSLg and Windows):

```bash
dotnet run --project src/MazeRL.UI
```

- **Pick a brain**: switch between the Q-table and the neural net (DQN) in the toolbar
- **Watch it learn**: start training and see the Q-value heatmap spread backwards from the goal while the policy arrows organize themselves into a path; speed slider goes from 1 to 500 episodes per tick
- **Watch the network train** (DQN mode): the bottom panel shows the net live — the one-hot input as a mini-maze, hidden nodes glowing with activation, edges colored by weight sign (green +, blue −) and sized by magnitude, the four Q-value outputs, and a falling loss curve. During a test run the input follows the ball, so you see the net "think" as it walks
- **Test run**: pauses training and animates the ball following the greedy policy — pure "show me what you know"
- **Draw mazes** (⚙ Settings): drag to paint/erase walls, place start and goal
- **Inspect the brain**: click any cell to see its four raw Q-values (and feed that cell into the network view); save/load the brain as JSON
- **Experiment**: wall off the learned path mid-training and watch the agent adapt

On WSL you need two small X11 libraries Avalonia depends on: `sudo apt install libice6 libsm6`.

## How the learning works

The agent's entire intelligence is a **Q-table**: for every cell in the maze and every action (up/down/left/right), one number estimating "how much total reward will I get if I do this here?" It starts at all zeros (total ignorance).

After *every single step* the table is nudged toward reality with the Q-learning update:

```
Q(s,a) += α · (reward + γ · max Q(s',a') − Q(s,a))
```

In words: *"move my old guess toward (what I just got) plus (the best I think I can do from where I landed)."* Reward flows backwards from the goal, one cell per visit, until the whole table points home.

Exploration is **epsilon-greedy**: with probability ε the agent acts randomly (discovering new routes), otherwise it does the best thing it knows. ε starts at 1.0 (pure random) and decays toward 0.05 as the agent gains confidence.

## The neural-net agent (DQN)

The DQN agent does the same job as the Q-table, but the table is replaced by a small fully-connected network (one-hot cell → 16 → 16 → 4 Q-values) written from scratch: ReLU hidden layers, backpropagation, Adam. Same update target — `r + γ · max Q(s′)` — but learned by gradient descent instead of a table nudge, with the two classic DQN stabilizers:

- **Experience replay**: transitions go into a memory buffer and training samples random batches from it, so the net isn't fed correlated consecutive steps
- **Target network**: a frozen copy of the net computes the learning targets and is re-synced only every 250 gradient steps, so the net isn't chasing its own moving output

## Rewards — the only teaching signal

| Event | Reward |
|---|---|
| Normal step | −1 (encourages short paths) |
| Bumping a wall | −5 |
| Reaching the goal | +100 |

Nobody tells the agent where the goal is or what walls are. It infers everything from these numbers.

## Project structure

```
src/
  MazeRL.Core/        # no UI dependencies — the learning engine
    IEnvironment.cs      # the state/action/reward contract (Gym-style)
    Maze.cs              # grid, walls, start/goal + JSON save/load
    MazeEnvironment.cs   # wraps a Maze as an RL environment; owns the rewards
    IAgent.cs            # what any brain must do — table and net are interchangeable
    QLearningAgent.cs    # the Q-table, action selection, the update rule
    NeuralNetwork.cs     # tiny MLP: forward pass, backprop, Adam — no libraries
    DqnAgent.cs          # DQN: replay buffer + target network around the MLP
    Trainer.cs           # episode loop, epsilon decay, test runs
  MazeRL.Console/     # console demo proving learning works (--dqn for the net)
  MazeRL.UI/          # Avalonia GUI: maze editor, live heatmap, training controls
    MazeBoard.cs         # renders maze + heatmap + arrows + ball; click/drag editing
    NetworkView.cs       # live picture of the DQN: activations, weights, Q-outputs
    LineChart.cs         # learning curve + loss curve
    MainWindow.axaml     # layout: toolbar, board, viz strip, collapsible settings
```

`IEnvironment` is deliberately generic — a slippery-floor maze, a maze with a patrolling monster, or any other world can be dropped in without touching the agent.

## Roadmap

- [x] Core engine + console proof that learning works
- [x] GUI (Avalonia): draw mazes, watch the heatmap/policy evolve live, speed slider, stats chart
- [x] Tiny hand-written neural network replacing the Q-table (DQN) + live network visualization
- [ ] Slippery-floor variant (stochastic environment)
- [ ] Moving obstacle (state-space explosion lesson)
