# MazeRL — reinforcement learning from scratch in C#

A ball learns to solve a maze using **tabular Q-learning**, written from the ground up — no ML libraries, no neural networks. The goal is to understand how an AI agent actually learns.

## Run it

```bash
dotnet run --project src/MazeRL.Console
```

You'll watch the agent go from random wandering to solving the maze optimally, then see its learned policy (arrows), a Q-value heatmap, and the final path. The trained "brain" is saved to `brain.json` — open it and see that the learning is literally just numbers.

## How the learning works

The agent's entire intelligence is a **Q-table**: for every cell in the maze and every action (up/down/left/right), one number estimating "how much total reward will I get if I do this here?" It starts at all zeros (total ignorance).

After *every single step* the table is nudged toward reality with the Q-learning update:

```
Q(s,a) += α · (reward + γ · max Q(s',a') − Q(s,a))
```

In words: *"move my old guess toward (what I just got) plus (the best I think I can do from where I landed)."* Reward flows backwards from the goal, one cell per visit, until the whole table points home.

Exploration is **epsilon-greedy**: with probability ε the agent acts randomly (discovering new routes), otherwise it does the best thing it knows. ε starts at 1.0 (pure random) and decays toward 0.05 as the agent gains confidence.

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
    QLearningAgent.cs    # the Q-table, action selection, the update rule
    Trainer.cs           # episode loop, epsilon decay, test runs
  MazeRL.Console/     # console demo proving learning works
```

`IEnvironment` is deliberately generic — a slippery-floor maze, a maze with a patrolling monster, or any other world can be dropped in without touching the agent.

## Roadmap

- [x] Core engine + console proof that learning works
- [ ] GUI (Avalonia): draw mazes, watch the heatmap/policy evolve live, speed slider, stats chart
- [ ] Slippery-floor variant (stochastic environment)
- [ ] Moving obstacle (state-space explosion lesson)
- [ ] Tiny hand-written neural network replacing the Q-table (DQN territory)
