"""
Miracle Rogue (Pressure Points build), turn-5-onward combo, Auctioneer
already resolved (its -4 mana is baked into the starting state, not a
playable card here). Record Scratcher / Blackwater Cutlass are NOT in this
list at all -- Pressure Points is the discount engine instead.

SIMPLIFICATIONS CARRIED OVER FROM THE SPEC (explicit, not hidden):
  - Shadow of Demise -> identical CardDef to Counterfeit Coin. No transform
    tracked; it's just treated as a 3rd coin.
  - Gone Fishin' -> flat "draw 2" (no Dredge choice).
  - Garrote -> NOT in CARD_DEFS at all. Never played as a modeled action.
    The win conditions are pure terminal-state predicates on (deck_total,
    mana) -- see TIER predicates at the bottom. This matches "ignore the
    shuffle mechanic, don't track damage."
  - No hand-size cap tracked.
  - Ethereal Augmerchant's battlecry damage / Spell Damage grant: untracked.
  - Pressure Points' own damage effect: untracked (only the cost-reduction
    half of its text matters here).

ASSUMPTION I ADDED (flagged, not from your spec): Ethereal Augmerchant gets
a fallback priority slot (4c below) for the case where it's in hand but
Shadowstep is NOT in hand and Dig for Treasure's window has already closed
(Augmerchant already found). Otherwise it would never get played outside
the Shadowstep case, which can't be intended given it's a fixed line item
in your mana ledger.
"""

from typing import List, Optional
from policy_automaton import (
    GameState, CardDef, PolicyAutomaton, make_state,
    Draw, DrawN, Tutor, AddMana, SetFlag, Custom,
)

# Framework default hand_cap is 10 (real Hearthstone hand limit). This deck
# is explicitly modeled WITHOUT tracking hand space (per spec: "there are
# clever ways to play around the hand space issue prior to playing the
# auctioneer") -- so every Draw()/Tutor()/DrawN() below must be passed this
# explicitly, or it silently reverts to the framework's default cap of 10
# and starts burning cards that shouldn't be burned.
NO_CAP = 999

# ---------------------------------------------------------------------------
# Deck (30 cards) -- counts, for reference / building the starting state
# ---------------------------------------------------------------------------

FULL_COUNTS = {
    "Backstab": 2, "Counterfeit Coin": 2, "Preparation": 2, "Ransack": 2,
    "Shadow of Demise": 1, "Shadowstep": 2,
    "Brain Freeze": 1, "Cold Blood": 2, "Dig for Treasure": 2,
    "Ethereal Augmerchant": 1, "Ghostly Strike": 2, "Gone Fishin'": 2,
    "Oh, Manager!": 2, "Swindle": 2,
    "Pressure Points": 2,
    # Garrote (2x) and Black Market Auctioneer (1x) deliberately excluded --
    # Auctioneer is already resolved (baked into starting flags/mana),
    # Garrote is never a modeled action (see predicates at bottom).
}
assert sum(FULL_COUNTS.values()) == 27, sum(FULL_COUNTS.values())
# 27 + Garrote(2) + Auctioneer(1) = 30 total in the real deck.

BASE_COST = {
    "Backstab": 0, "Counterfeit Coin": 0, "Preparation": 0, "Ransack": 0,
    "Shadow of Demise": 0, "Shadowstep": 0,
    "Brain Freeze": 1, "Cold Blood": 1, "Dig for Treasure": 1,
    "Ethereal Augmerchant": 1, "Ghostly Strike": 1, "Gone Fishin'": 1,
    "Oh, Manager!": 2, "Swindle": 2,
    "Pressure Points": 3,
}

NAME_TYPE = {  # for Dig for Treasure's tutor -- only Augmerchant is a minion here
    name: ("minion" if name == "Ethereal Augmerchant" else "spell")
    for name in BASE_COST
}

# The six Combo-keyword cards Pressure Points can discount
COMBO_CARDS = ["Cold Blood", "Brain Freeze", "Ghostly Strike", "Gone Fishin'", "Oh, Manager!", "Swindle"]
# Of those, all but Swindle are mana-neutral-or-better once discounted
MANA_NEUTRAL_COMBO_CARDS = ["Cold Blood", "Brain Freeze", "Ghostly Strike", "Gone Fishin'", "Oh, Manager!"]


def pp_name(name: str) -> str:
    return name + " (PP)"


# ---------------------------------------------------------------------------
# cost/effect helpers
# ---------------------------------------------------------------------------

def prep_discounted_cost(base: int):
    """Preparation auto-applies to whatever's cast next (matches real rules
    -- not optional/selective); -2, floored at 0."""
    def cost(state: GameState):
        return max(0, base - 2) if state.flag("prep_charge", 0) > 0 else base
    return cost


def spell_ops(core_ops: list) -> list:
    """Every spell here triggers the (already-resolved) Auctioneer's draw
    trigger, on top of whatever it does itself."""
    return core_ops + [Draw(hand_cap=NO_CAP)]


def consume_prep_if_used(name: str):
    """
    Consume Preparation when a non-zero-base-cost card is played.

    If this card was the specific card Preparation designated as its
    mandatory next target, also clear prep_target so normal policy resumes
    afterward.
    """
    def fn(state: GameState) -> GameState:
        if (
            state.flag("prep_charge", 0) > 0
            and BASE_COST.get(name, 0) > 0
        ):
            state = state.with_flag(
                "prep_charge",
                state.flag("prep_charge", 0) - 1,
            )

        if state.flag("prep_target") == name:
            state = state.with_flag("prep_target", None)

        return state

    return fn


def make_simple_spell(name: str, extra_ops: list = None) -> CardDef:
    extra_ops = extra_ops or []
    return CardDef(
        name=name,
        cost=prep_discounted_cost(BASE_COST[name]),
        effect=[Custom(consume_prep_if_used(name))] + spell_ops(extra_ops),
    )


# ---------------------------------------------------------------------------
# Card definitions
# ---------------------------------------------------------------------------

CARD_DEFS = {}

CARD_DEFS["Backstab"] = make_simple_spell("Backstab")
CARD_DEFS["Ransack"] = make_simple_spell("Ransack")
CARD_DEFS["Counterfeit Coin"] = make_simple_spell("Counterfeit Coin", [AddMana(lambda s: 1)])
CARD_DEFS["Shadow of Demise"] = make_simple_spell("Shadow of Demise", [AddMana(lambda s: 1)])  # treated as a 3rd coin
CARD_DEFS["Preparation"] = make_simple_spell(
    "Preparation",
    [
        SetFlag(
            "prep_charge",
            lambda s: s.flag("prep_charge", 0) + 1
        ),
        SetFlag(
            "prep_target",
            lambda s: choose_prep_target(s)
        ),
    ],
)

CARD_DEFS["Shadowstep"] = CardDef(
    name="Shadowstep",
    cost=prep_discounted_cost(BASE_COST["Shadowstep"]),
    condition=lambda s: bool(s.flag("augmerchant_in_play")),
    effect=[Custom(consume_prep_if_used("Shadowstep"))] + spell_ops([]),
)

CARD_DEFS["Ethereal Augmerchant"] = CardDef(
    name="Ethereal Augmerchant",
    cost=prep_discounted_cost(BASE_COST["Ethereal Augmerchant"]),  # a minion, but keep prep-discount plumbing unused (base handles it)
    effect=[
        Custom(consume_prep_if_used("Ethereal Augmerchant")),
        SetFlag("augmerchant_in_play", lambda s: True),
        SetFlag("augmerchant_played", lambda s: True),
    ],  # NOTE: no Draw() -- it's a minion, doesn't trigger the Auctioneer
)

CARD_DEFS["Dig for Treasure"] = make_simple_spell(
    "Dig for Treasure", [Tutor(lambda n: NAME_TYPE.get(n) == "minion", hand_cap=NO_CAP)]
)
CARD_DEFS["Ghostly Strike"] = make_simple_spell("Ghostly Strike", [Draw(hand_cap=NO_CAP)])  # Combo: draw a card
CARD_DEFS["Gone Fishin'"] = make_simple_spell("Gone Fishin'", [DrawN(2, hand_cap=NO_CAP)])  # batched, not 2 sequential Draw()s
CARD_DEFS["Cold Blood"] = make_simple_spell("Cold Blood")  # no draw/mana effect of its own
CARD_DEFS["Brain Freeze"] = make_simple_spell("Brain Freeze")  # damage/freeze untracked
CARD_DEFS["Oh, Manager!"] = make_simple_spell("Oh, Manager!", [AddMana(lambda s: 1)])  # Combo: Get a Coin
CARD_DEFS["Swindle"] = make_simple_spell(
    "Swindle",
    [Tutor(lambda n: NAME_TYPE.get(n) == "spell", hand_cap=NO_CAP),
     Tutor(lambda n: NAME_TYPE.get(n) == "minion", hand_cap=NO_CAP)],
)

# --- Pressure Points-discounted virtual copies (own CardDefs, cheaper cost) ---
for _name in COMBO_CARDS:
    CARD_DEFS[pp_name(_name)] = CardDef(
        name=pp_name(_name),
        cost=prep_discounted_cost(max(0, BASE_COST[_name] - 1)),
        effect=CARD_DEFS[_name].effect,  # identical effect, just cheaper
    )


def _pressure_points_transform(state: GameState) -> GameState:
    for name in COMBO_CARDS:
        c = state.hand_count(name)
        if c > 0:
            state = state.with_hand_delta(name, -c)
            state = state.with_hand_delta(pp_name(name), +c)
    return state


CARD_DEFS["Pressure Points"] = CardDef(
    name="Pressure Points",
    cost=prep_discounted_cost(BASE_COST["Pressure Points"]),
    condition=lambda s: not s.flag("pp_used"),  # hard cap: second copy is permanently illegal
    effect=[
        Custom(consume_prep_if_used("Pressure Points")),
        Custom(_pressure_points_transform),
        SetFlag("pp_used", lambda s: True),
    ] + spell_ops([]),
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

ZERO_COST = [
    "Counterfeit Coin",
    "Shadow of Demise",
    "Backstab",
    "Ransack",
    "Shadowstep",
]

OH_MANAGER_NAMES = ["Oh, Manager!", pp_name("Oh, Manager!")]
SWINDLE_NAMES = ["Swindle", pp_name("Swindle")]
COLD_BLOOD_NAMES = ["Cold Blood", pp_name("Cold Blood")]
BRAIN_FREEZE_NAMES = ["Brain Freeze", pp_name("Brain Freeze")]
GONE_FISHIN_NAMES = ["Gone Fishin'", pp_name("Gone Fishin'")]
GHOSTLY_STRIKE_NAMES = ["Ghostly Strike", pp_name("Ghostly Strike")]


def _first_legal(names, legal):
    for n in names:
        if n in legal:
            return n
    return None


# Preparation targets, in descending priority.
#
# Pressure Points is handled separately because it is only the preferred
# target when we have at least 3 mana-neutral Combo cards in hand.
#
# Augmerchant is deliberately absent: it is a minion.
# Dig for Treasure is deliberately absent: it is not a priority target.
PREP_TARGETS = [
    "Oh, Manager!",
    "Gone Fishin'",
    "Ghostly Strike",
    "Cold Blood",
    "Brain Freeze",
    "Swindle",
]


def choose_prep_target(state: GameState) -> Optional[str]:
    """
    Decide what the immediately-following play should be if we play
    Preparation now.

    Pressure Points is the highest-value target whenever:
      - PP has not already been used,
      - at least 3 mana-neutral Combo cards are in hand,
      - PP is in hand,
      - and we have at least 1 mana, so the Prepped PP (3 -> 1) is
        actually playable immediately.

    Otherwise choose the highest-priority ordinary spell that Preparation
    can make free.

    This function is called BEFORE Preparation is played, so the chosen
    target is stored in the state and becomes mandatory on the next
    policy evaluation.
    """

    eligible_combo_count = sum(
        state.hand_count(n)
        for n in MANA_NEUTRAL_COMBO_CARDS
    )

    # Pressure Points is the most valuable Prep target.
    if (
        not state.flag("pp_used")
        and eligible_combo_count >= 3
        and state.hand_count("Pressure Points") > 0
        and state.mana >= 1
    ):
        return "Pressure Points"

    # Otherwise use Prep on the highest-priority normal spell available.
    for name in PREP_TARGETS:
        if state.hand_count(name) > 0:
            return name

    return None


def policy(state: GameState, legal_actions: List[str]) -> Optional[str]:
    legal = set(legal_actions)

    # ---------------------------------------------------------------
    # 0. Mandatory Preparation target
    #
    # Once Preparation has been played, its intended target MUST be
    # played next. This deliberately comes before the zero-cost tier.
    #
    # Example:
    #   Preparation -> draws Ransack
    #
    # We still play the intended Prep target first; we do NOT play the
    # newly drawn Ransack before it.
    # ---------------------------------------------------------------
    prep_target = state.flag("prep_target")

    if prep_target is not None:
        if prep_target in legal:
            return prep_target

        # This should normally be impossible because choose_prep_target()
        # only selects a target that was in hand when Preparation was cast.
        #
        # If it somehow becomes illegal, stop rather than silently playing
        # something else and violating the "Prep target next" rule.
        return None

    # ---------------------------------------------------------------
    # 1. Naturally 0-cost cards
    # ---------------------------------------------------------------
    zc = _first_legal(ZERO_COST, legal)
    if zc:
        return zc

    # ---------------------------------------------------------------
    # 2. Preparation
    #
    # If there is a useful Prep target, Preparation takes priority over
    # the normal card-playing order.
    #
    # Pressure Points is preferred whenever it has >=3 qualifying Combo
    # cards in hand.
    # ---------------------------------------------------------------
    if state.flag("prep_charge", 0) == 0 and "Preparation" in legal:
        prep_target = choose_prep_target(state)

        if prep_target is not None:
            return "Preparation"

    # ---------------------------------------------------------------
    # 3. Normal priority
    #
    # Dig for Treasure is intentionally NOT played here.
    #
    # Augmerchant gets its special priority because the Shadowstep /
    # Augmerchant line is part of the intended combo.
    # ---------------------------------------------------------------
    if (
        state.hand_count("Shadowstep") > 0
        and "Ethereal Augmerchant" in legal
    ):
        return "Ethereal Augmerchant"

    if "Ethereal Augmerchant" in legal:
        return "Ethereal Augmerchant"

    # ---------------------------------------------------------------
    # 4. Draw cards
    # ---------------------------------------------------------------
    if _first_legal(GONE_FISHIN_NAMES, legal):
        return _first_legal(GONE_FISHIN_NAMES, legal)

    if _first_legal(GHOSTLY_STRIKE_NAMES, legal):
        return _first_legal(GHOSTLY_STRIKE_NAMES, legal)

    # ---------------------------------------------------------------
    # 5. Remaining Combo cards
    # ---------------------------------------------------------------
    if _first_legal(OH_MANAGER_NAMES, legal):
        return _first_legal(OH_MANAGER_NAMES, legal)

    if _first_legal(COLD_BLOOD_NAMES, legal):
        return _first_legal(COLD_BLOOD_NAMES, legal)

    if _first_legal(BRAIN_FREEZE_NAMES, legal):
        return _first_legal(BRAIN_FREEZE_NAMES, legal)

    if _first_legal(SWINDLE_NAMES, legal):
        return _first_legal(SWINDLE_NAMES, legal)

    # ---------------------------------------------------------------
    # 6. Pressure Points
    #
    # PP is deliberately very low in the ordinary priority order.
    # It is only played here if:
    #   - it wasn't already used,
    #   - >=3 qualifying Combo cards are in hand,
    #   - and all higher-priority actions have been exhausted.
    # ---------------------------------------------------------------
    eligible_combo_count = sum(
        state.hand_count(n)
        for n in MANA_NEUTRAL_COMBO_CARDS
    )

    if (
        not state.flag("pp_used")
        and eligible_combo_count >= 3
        and "Pressure Points" in legal
    ):
        return "Pressure Points"

    return None

# ---------------------------------------------------------------------------
# Outcome predicates (checked at the terminal state)
# ---------------------------------------------------------------------------

def tier1_both_garrotes(state: GameState) -> bool:
    return state.deck_total() <= 2 and state.mana >= 4


def tier2_one_garrote(state: GameState) -> bool:
    return (state.deck_total() <= 1 and state.mana >= 2) and not tier1_both_garrotes(state)


def tier3_failure(state: GameState) -> bool:
    return not tier1_both_garrotes(state) and not tier2_one_garrote(state)


PREDICATES = {"tier1": tier1_both_garrotes, "tier2": tier2_one_garrote, "tier3": tier3_failure}

AUTOMATON = PolicyAutomaton(CARD_DEFS, policy)


# ---------------------------------------------------------------------------
# Mana-bound pruning: a SOUND (never over-prunes) upper bound on achievable
# end-of-turn mana from any given state. Deliberately loose rather than
# tight -- it only counts remaining INCOME, never subtracts future costs
# (Augmerchant, Pressure Points), because whether those costs are ever
# actually paid depends on the random draws and isn't guaranteed. A loose
# bound can only under-prune (safe), never incorrectly cut off a branch
# that could still reach the target -- tightening it further is possible
# later if this doesn't cut enough of the tree.
# ---------------------------------------------------------------------------

def mana_upper_bound(state: GameState) -> float:
    coin_sources = 0
    for name in ("Counterfeit Coin", "Shadow of Demise"):
        coin_sources += state.hand_count(name) + state.deck_count(name)
    oh_manager_sources = 0
    for name in ("Oh, Manager!", pp_name("Oh, Manager!")):
        oh_manager_sources += state.hand_count(name) + state.deck_count(name)
    preps_remaining = (
        state.flag("prep_charge", 0)
        + state.hand_count("Preparation")
        + state.deck_count("Preparation")
    )
    return state.mana + coin_sources + oh_manager_sources + 2 * preps_remaining


# For the tier1-only heavy run: a single scalar success predicate instead of
# the 3-way dict, to cut per-node overhead at this scale.
def success_tier1(state: GameState) -> bool:
    return tier1_both_garrotes(state)

