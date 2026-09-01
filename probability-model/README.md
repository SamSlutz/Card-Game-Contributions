# Card Game Probability Modeling

An exact probability solver for evaluating fixed play policies in card game
combo lines, applied here to a Hearthstone Rogue build. Instead of
estimating outcomes via Monte Carlo simulation, this computes the exact
probability of reaching a target end state, given a deterministic policy
and a random shuffle.

## Why this shape of problem

Monte Carlo simulation can estimate a combo's win rate. What it can't
easily give you is the mathematical structure underneath; how the
probability mass actually distributes across the shuffle, or why a given
policy is or isn't near-optimal. That's the part I find interesting,
and it's the reason this project exists as an exact solver rather than a simulator.

## The idea

For a fixed policy (a deterministic function of game state → next action),
every distinct shuffle of the deck induces exactly one play trace and one
terminal state. There's no branching from choice, only from the
randomness of the draw. That means "probability of success" is a property
of the permutation space, not something we need to sample our way
toward.

`policy_automaton.py` is the generic engine: given a policy and a set of
card definitions (cost, condition, effect), it computes exact outcome
probabilities via a memoized backward recursion over compressed game
states, thus, many distinct card orderings collapse onto the same internal
state, so this scales better than enumerating permutations directly,
while still being provably equivalent to brute-force enumeration on small
cases (see `solve_bruteforce`, used to validate `solve_exact`).

## Files

| File | What it is |
|---|---|
| `policy_automaton.py` | Generic solver: game state, effect primitives (`Draw`, `Tutor`, `AddMana`, etc.), and multiple solve modes (`solve_exact`, `solve_bruteforce`, `solve_optimal`, `solve_with_progress`) |
| `miracle_rogue.py` | Deck definition, card effects, and hand-coded policy for a specific Miracle Rogue (Pressure Points) build |
| `yuge.py` | Run script - sets up a starting hand/deck and solves for probability of reaching the win condition |
| `archive/` | Earlier drafts, kept for reference |

## Status

The framework runs and cross-validates correctly against brute force
enumeration on small test decks. At full deck size, exact runs are slow
enough that I haven't yet confirmed whether that's simply the size of the
state space or a policy inefficiency, that's the current open question
I'm working through.
