"""
policy_automaton.py -- the frame. No game-specific content lives here.

Mental model:

    permutation of the deck  --->  automaton (policy + card rules)  --->  terminal state

A "policy" is a deterministic function of (state, legal_actions) -> chosen
action or None (stop). Given a fixed policy, every distinct shuffle of the
deck induces exactly one play trace and one terminal state -- there are no
branches from choice, only from the randomness of the shuffle. Success (or
any other predicate) is a property of that terminal state, and its
probability is a property of the permutation space, not of a search.

Two solvers, guaranteed to agree, for two different purposes:

  solve_exact(...)       Efficient. Hand/deck are tracked as per-card-type
                          counts, not literal card identities. Many distinct
                          deck orderings collapse onto the same internal
                          state and share one memoized computation. This is
                          valid because, for a uniformly random shuffle,
                          every distinct card-TYPE sequence is equally
                          likely -- a fixed multiset of counts is realized
                          by exactly prod(n_i!) underlying full permutations
                          regardless of which pattern it is, so probability
                          mass per pattern is constant. This function
                          computes precisely the same number solve_bruteforce
                          would, just without enumerating permutations.

  solve_bruteforce(...)  Ground truth. Literally enumerates every distinct
                          card-type ordering of the supplied deck, runs the
                          automaton on each one once, averages. Only usable
                          on small decks (this exists to validate a new
                          deck/policy pair against solve_exact, not to be
                          the real engine).

  solve_optimal(...)     A third mode, kept for when you DO want to ask
                          "what's the best achievable value" rather than
                          evaluate one fixed policy -- same machinery, but
                          replaces the policy's single choice with a max
                          over all legal actions. Only takes one scalar
                          objective (not a dict of predicates), since
                          "optimal for A" and "optimal for B" can require
                          different play.

You define: GameState's `flags` schema, CardDef.cost/condition/effect for
each card, and a policy function. Everything else here is generic.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional, Dict, List, Tuple, Any
from itertools import permutations as _permutations
import heapq
import itertools as _itertools


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameState:
    mana: int
    hand: Tuple[Tuple[str, int], ...]
    deck: Tuple[Tuple[str, int], ...]
    flags: Tuple[Tuple[str, Any], ...]

    def hand_count(self, name: str) -> int:
        return dict(self.hand).get(name, 0)

    def deck_count(self, name: str) -> int:
        return dict(self.deck).get(name, 0)

    def deck_total(self) -> int:
        return sum(c for _, c in self.deck)

    def flag(self, name: str, default=None):
        return dict(self.flags).get(name, default)

    def with_mana(self, new_mana: int) -> "GameState":
        return GameState(new_mana, self.hand, self.deck, self.flags)

    def with_hand_delta(self, name: str, delta: int, hand_cap: Optional[int] = None) -> "GameState":
        d = dict(self.hand)
        newcount = d.get(name, 0) + delta
        if newcount < 0:
            raise ValueError(f"hand count for {name} went negative")
        if hand_cap is not None and delta > 0:
            total_other = sum(c for n, c in d.items() if n != name)
            if total_other + newcount > hand_cap:
                newcount = d.get(name, 0)  # burned: doesn't actually join hand
        if newcount == 0:
            d.pop(name, None)
        else:
            d[name] = newcount
        return GameState(self.mana, tuple(sorted(d.items())), self.deck, self.flags)

    def with_deck_delta(self, name: str, delta: int) -> "GameState":
        d = dict(self.deck)
        newcount = d.get(name, 0) + delta
        if newcount < 0:
            raise ValueError(f"deck count for {name} went negative")
        if newcount == 0:
            d.pop(name, None)
        else:
            d[name] = newcount
        return GameState(self.mana, self.hand, tuple(sorted(d.items())), self.flags)

    def with_flag(self, name: str, value: Any) -> "GameState":
        d = dict(self.flags)
        if value is None:
            d.pop(name, None)
        else:
            d[name] = value
        return GameState(self.mana, self.hand, self.deck, tuple(sorted(d.items())))


def make_state(mana: int, hand: Dict[str, int], deck: Dict[str, int], flags: Dict[str, Any] = None) -> GameState:
    flags = flags or {}
    return GameState(
        mana,
        tuple(sorted((k, v) for k, v in hand.items() if v > 0)),
        tuple(sorted((k, v) for k, v in deck.items() if v > 0)),
        tuple(sorted(flags.items())),
    )


# ---------------------------------------------------------------------------
# Effect primitives: the vocabulary cards are built from. Both interpreters
# below know how to execute each of these. A card's `effect` is just a list
# of them, applied in order.
# ---------------------------------------------------------------------------

class Draw:
    """Draw the next card off the top of the deck."""
    def __init__(self, hand_cap: int = 10):
        self.hand_cap = hand_cap


class DrawN:
    """Draw N cards at once, as a single joint multivariate-hypergeometric
    outcome distribution -- NOT N sequential Draw()s. Sequential Draw()s
    branch out and merge back N times before the outer memoized recursion
    ever sees the result, which is pure wasted work (draws commute: there's
    no decision or state-dependence between them, so nothing is lost by
    computing the joint distribution directly)."""
    def __init__(self, n: int, hand_cap: int = 10):
        self.n = n
        self.hand_cap = hand_cap


class Tutor:
    """Search the deck (in shuffle order) for the first card matching
    predicate(name), draw it, skip/ignore everything before it."""
    def __init__(self, predicate: Callable[[str], bool], hand_cap: int = 10):
        self.predicate = predicate
        self.hand_cap = hand_cap


class AddHandCard:
    """Add a specific card straight to hand, not from the deck (e.g. a
    generated token, a copy from an effect)."""
    def __init__(self, name: str, count: int = 1, hand_cap: int = 10):
        self.name, self.count, self.hand_cap = name, count, hand_cap


class SetFlag:
    def __init__(self, name: str, value_fn: Callable[[GameState], Any]):
        self.name, self.value_fn = name, value_fn


class AddMana:
    def __init__(self, delta_fn: Callable[[GameState], int]):
        self.delta_fn = delta_fn


class TransformInHand:
    """If trigger_name is currently in hand, replace one copy of it with
    into_name_fn(state) (e.g. a Shadow-of-Demise-style transform)."""
    def __init__(self, trigger_name: str, into_name_fn: Callable[[GameState], str]):
        self.trigger_name, self.into_name_fn = trigger_name, into_name_fn


class Custom:
    """Escape hatch: an arbitrary deterministic state -> state transform for
    anything the primitives above don't cover."""
    def __init__(self, fn: Callable[[GameState], GameState]):
        self.fn = fn


# ---------------------------------------------------------------------------
# Card definition
# ---------------------------------------------------------------------------

@dataclass
class CardDef:
    name: str
    cost: Callable[[GameState], Optional[int]]
    condition: Callable[[GameState], bool] = lambda s: True
    effect: List = field(default_factory=list)  # list of primitive ops above


# ---------------------------------------------------------------------------
# Interpreter 1: exact / stochastic (state-compressed)
# ---------------------------------------------------------------------------

def _multivariate_hypergeometric(deck: Tuple[Tuple[str, int], ...], k: int):
    """All ways to draw k cards from `deck` (a tuple of (name,count)) in one
    joint step, as (Fraction probability, {name: drawn_count}) pairs. This
    is the direct joint distribution -- the thing N sequential Draw()s
    compute the hard way, by branching out to it and back."""
    from math import comb
    names = [n for n, c in deck]
    counts = [c for n, c in deck]
    total = sum(counts)
    if total == 0 or k == 0:
        return [(Fraction(1), {})]
    k = min(k, total)
    results = []

    def recurse(idx, remaining, draws, ways):
        if idx == len(names):
            if remaining == 0:
                results.append((ways, dict(draws)))
            return
        # prune: can't possibly reach `remaining` with what's left
        max_possible = sum(counts[idx:])
        if max_possible < remaining:
            return
        max_take = min(counts[idx], remaining)
        for take in range(0, max_take + 1):
            if take:
                draws[names[idx]] = take
            recurse(idx + 1, remaining - take, draws, ways * comb(counts[idx], take))
            draws.pop(names[idx], None)

    recurse(0, k, {}, 1)
    total_ways = comb(total, k)
    return [(Fraction(ways, total_ways), d) for ways, d in results if ways > 0]


def _exec_op_exact(state: GameState, op) -> List[Tuple[Fraction, GameState]]:
    if isinstance(op, Draw):
        total = state.deck_total()
        if total == 0:
            return [(Fraction(1), state)]
        out = []
        for name, count in state.deck:
            p = Fraction(count, total)
            out.append((p, state.with_deck_delta(name, -1).with_hand_delta(name, +1, hand_cap=op.hand_cap)))
        return out
    if isinstance(op, DrawN):
        combos = _multivariate_hypergeometric(state.deck, op.n)
        out = []
        for p, drawn in combos:
            s2 = state
            for name, cnt in drawn.items():
                s2 = s2.with_deck_delta(name, -cnt).with_hand_delta(name, +cnt, hand_cap=op.hand_cap)
            out.append((p, s2))
        return out
    if isinstance(op, Tutor):
        matches = [(n, c) for n, c in state.deck if op.predicate(n)]
        total = sum(c for _, c in matches)
        if total == 0:
            return [(Fraction(1), state)]
        out = []
        for name, count in matches:
            p = Fraction(count, total)
            out.append((p, state.with_deck_delta(name, -1).with_hand_delta(name, +1, hand_cap=op.hand_cap)))
        return out
    if isinstance(op, AddHandCard):
        return [(Fraction(1), state.with_hand_delta(op.name, op.count, hand_cap=op.hand_cap))]
    if isinstance(op, SetFlag):
        return [(Fraction(1), state.with_flag(op.name, op.value_fn(state)))]
    if isinstance(op, AddMana):
        return [(Fraction(1), state.with_mana(state.mana + op.delta_fn(state)))]
    if isinstance(op, TransformInHand):
        if state.hand_count(op.trigger_name) > 0:
            into = op.into_name_fn(state)
            return [(Fraction(1), state.with_hand_delta(op.trigger_name, -1).with_hand_delta(into, +1, hand_cap=10))]
        return [(Fraction(1), state)]
    if isinstance(op, Custom):
        return [(Fraction(1), op.fn(state))]
    raise TypeError(f"unknown effect op {op!r}")


def _apply_exact(state: GameState, ops: List) -> List[Tuple[Fraction, GameState]]:
    branches = [(Fraction(1), state)]
    for op in ops:
        new_branches = []
        for p, s in branches:
            for p2, s2 in _exec_op_exact(s, op):
                new_branches.append((p * p2, s2))
        branches = new_branches
    return branches


# ---------------------------------------------------------------------------
# Interpreter 2: deterministic single trace (drives a literal deck order,
# via a shared mutable cursor -- used by solve_bruteforce)
# ---------------------------------------------------------------------------

def _exec_op_deterministic(state: GameState, op, cursor: List[str]) -> GameState:
    if isinstance(op, Draw):
        if not cursor:
            return state
        name = cursor.pop(0)
        return state.with_deck_delta(name, -1).with_hand_delta(name, +1, hand_cap=op.hand_cap)
    if isinstance(op, DrawN):
        for _ in range(op.n):
            if not cursor:
                break
            name = cursor.pop(0)
            state = state.with_deck_delta(name, -1).with_hand_delta(name, +1, hand_cap=op.hand_cap)
        return state
    if isinstance(op, Tutor):
        for i, name in enumerate(cursor):
            if op.predicate(name):
                cursor.pop(i)
                return state.with_deck_delta(name, -1).with_hand_delta(name, +1, hand_cap=op.hand_cap)
        return state
    if isinstance(op, AddHandCard):
        return state.with_hand_delta(op.name, op.count, hand_cap=op.hand_cap)
    if isinstance(op, SetFlag):
        return state.with_flag(op.name, op.value_fn(state))
    if isinstance(op, AddMana):
        return state.with_mana(state.mana + op.delta_fn(state))
    if isinstance(op, TransformInHand):
        if state.hand_count(op.trigger_name) > 0:
            into = op.into_name_fn(state)
            return state.with_hand_delta(op.trigger_name, -1).with_hand_delta(into, +1, hand_cap=10)
        return state
    if isinstance(op, Custom):
        return op.fn(state)
    raise TypeError(f"unknown effect op {op!r}")


def _apply_deterministic(state: GameState, ops: List, cursor: List[str]) -> GameState:
    for op in ops:
        state = _exec_op_deterministic(state, op, cursor)
    return state


# ---------------------------------------------------------------------------
# The automaton
# ---------------------------------------------------------------------------

Policy = Callable[[GameState, List[str]], Optional[str]]
Predicate = Callable[[GameState], bool]


class PolicyAutomaton:
    def __init__(self, card_defs: Dict[str, CardDef], policy: Policy):
        self.card_defs = card_defs
        self.policy = policy

    def legal_actions(self, state: GameState) -> List[str]:
        out = []
        for name, count in state.hand:
            if count <= 0 or name not in self.card_defs:
                continue
            cdef = self.card_defs[name]
            cost = cdef.cost(state)
            if cost is None or cost > state.mana:
                continue
            if not cdef.condition(state):
                continue
            out.append(name)
        return out

    def _pay_and_remove(self, state: GameState, name: str) -> GameState:
        cdef = self.card_defs[name]
        cost = cdef.cost(state)
        return state.with_mana(state.mana - cost).with_hand_delta(name, -1)

    # ---- exact, single scalar predicate, with optional pruning -----------
    def solve_exact_scalar(
        self,
        start_state: GameState,
        success_fn: Callable[[GameState], bool],
        bound_fn: Optional[Callable[[GameState], float]] = None,
        bound_threshold: Optional[float] = None,
        progress_every: Optional[int] = None,
    ) -> float:
        """Same backward memoized recursion as solve_exact, but for a single
        boolean predicate (less per-node overhead than the dict version),
        with an optional sound pruning bound: if bound_fn(state) is below
        bound_threshold, that whole subtree is resolved as failure (0.0)
        without ever recursing into it.

        progress_every: if set, prints a cheap (calls, memo size) ticker --
        NOT true probability bounds (that needs the slower forward solver,
        solve_with_progress) -- just enough to see it's alive and how fast
        the memo is growing relative to raw calls (a big calls/memo ratio is
        exactly the redundant-branching symptom DrawN is meant to fix)."""
        memo: Dict[GameState, float] = {}
        calls = [0]

        def rec(state: GameState) -> float:
            if state in memo:
                return memo[state]
            calls[0] += 1
            if progress_every and calls[0] % progress_every == 0:
                print(f"{calls[0]:,} calls | {len(memo):,} memo states")
            if bound_fn is not None and bound_threshold is not None and bound_fn(state) < bound_threshold:
                memo[state] = 0.0
                return 0.0
            actions = self.legal_actions(state)
            chosen = self.policy(state, actions) if actions else None
            if chosen is None:
                val = 1.0 if success_fn(state) else 0.0
            else:
                s2 = self._pay_and_remove(state, chosen)
                val = sum(float(p) * rec(s3) for p, s3 in _apply_exact(s2, self.card_defs[chosen].effect))
            memo[state] = val
            return val

        return rec(start_state)

    # ---- exact (fast, state-compressed) --------------------------------
    def solve_exact(self, start_state: GameState, predicates: Dict[str, Predicate]) -> Dict[str, float]:
        memo: Dict[GameState, Dict[str, float]] = {}

        def rec(state: GameState) -> Dict[str, float]:
            if state in memo:
                return memo[state]
            actions = self.legal_actions(state)
            chosen = self.policy(state, actions) if actions else None
            if chosen is None:
                result = {k: (1.0 if pred(state) else 0.0) for k, pred in predicates.items()}
            else:
                s2 = self._pay_and_remove(state, chosen)
                branches = _apply_exact(s2, self.card_defs[chosen].effect)
                result = {k: 0.0 for k in predicates}
                for p, s3 in branches:
                    sub = rec(s3)
                    for k in predicates:
                        result[k] += float(p) * sub[k]
            memo[state] = result
            return result

        return rec(start_state)

    # ---- literal ground truth (small decks only) ------------------------
    def run_single_trace(self, start_state: GameState, deck_sequence: List[str]) -> GameState:
        cursor = list(deck_sequence)
        state = start_state
        while True:
            actions = self.legal_actions(state)
            chosen = self.policy(state, actions) if actions else None
            if chosen is None:
                return state
            state = self._pay_and_remove(state, chosen)
            state = _apply_deterministic(state, self.card_defs[chosen].effect, cursor)

    def solve_bruteforce(self, start_state: GameState, deck_multiset: List[str],
                          predicates: Dict[str, Predicate]) -> Dict[str, float]:
        distinct = set(_permutations(deck_multiset))
        totals = {k: 0.0 for k in predicates}
        for perm in distinct:
            final = self.run_single_trace(start_state, list(perm))
            for k, pred in predicates.items():
                if pred(final):
                    totals[k] += 1
        n = len(distinct)
        return {k: v / n for k, v in totals.items()}

    # ---- optimal play (max over actions instead of a fixed policy) ------
    def solve_optimal(self, start_state: GameState, success_fn: Callable[[GameState], float]) -> float:
        memo: Dict[GameState, float] = {}

        def rec(state: GameState) -> float:
            if state in memo:
                return memo[state]
            best = success_fn(state)  # stopping now is always allowed
            for name in self.legal_actions(state):
                s2 = self._pay_and_remove(state, name)
                ev = sum(float(p) * rec(s3) for p, s3 in _apply_exact(s2, self.card_defs[name].effect))
                if ev > best:
                    best = ev
            memo[state] = best
            return best

        return rec(start_state)

    # ---- forward/topological with live bounds and optional pruning -------
    def solve_with_progress(
        self,
        start_state: GameState,
        success_fn: Callable[[GameState], bool],
        bound_fn: Optional[Callable[[GameState], float]] = None,
        bound_threshold: Optional[float] = None,
        report_every: int = 10_000,
    ) -> float:
        """Same answer as solve_exact (single scalar predicate version), but
        computed as a FORWARD pass over states in topological order (by
        deck_total descending -- every card effect either draws, which
        strictly shrinks the deck, or is Ethereal Augmerchant, which is the
        one card that can leave deck_total unchanged; that's still a valid
        partial order via a (deck_total, augmerchant_played) tiebreak, so
        there are no cycles to worry about).

        Because we process in topological order, probability mass reaching
        each state is *fully* accumulated before that state is ever popped
        and expanded -- so at any point during the run, every unit of mass
        that has reached a terminal state is done for good, giving genuine
        running bounds:
            lower_bound = success_mass
            upper_bound = success_mass + (1 - completed_mass)   [optimistic:
                          assumes everything still in flight succeeds]

        `bound_fn`/`bound_threshold`: if bound_fn(state) < bound_threshold
        at a non-terminal state, that branch is mathematically guaranteed to
        fail no matter what happens afterward (bound_fn must be a genuine
        upper bound on achievable value), so it's resolved as a failure
        immediately instead of being expanded further.
        """
        frontier: Dict[GameState, Fraction] = {start_state: Fraction(1)}
        heap = [(-start_state.deck_total(), bool(start_state.flag("augmerchant_played")), 0, start_state)]
        counter = _itertools.count(1)
        processed = set()
        completed_mass = Fraction(0)
        success_mass = Fraction(0)
        calls = 0

        while heap:
            _, _, _, state = heapq.heappop(heap)
            if state in processed:
                continue
            processed.add(state)
            mass = frontier.get(state)
            if not mass:
                continue
            calls += 1

            pruned = bound_fn is not None and bound_threshold is not None and bound_fn(state) < bound_threshold
            actions = [] if pruned else self.legal_actions(state)
            chosen = self.policy(state, actions) if actions else None

            if pruned or chosen is None:
                completed_mass += mass
                if (not pruned) and success_fn(state):
                    success_mass += mass
            else:
                s2 = self._pay_and_remove(state, chosen)
                for p, s3 in _apply_exact(s2, self.card_defs[chosen].effect):
                    frontier[s3] = frontier.get(s3, Fraction(0)) + mass * p
                    heapq.heappush(
                        heap,
                        (-s3.deck_total(), bool(s3.flag("augmerchant_played")), next(counter), s3),
                    )

            if calls % report_every == 0:
                remaining = Fraction(1) - completed_mass
                lo, hi = float(success_mass), float(success_mass + remaining)
                print(f"{calls:,} states popped | {len(frontier):,} in frontier | "
                      f"bounds [{lo:.6f}, {hi:.6f}] | completed mass {float(completed_mass):.6f}")

        return float(success_mass)
