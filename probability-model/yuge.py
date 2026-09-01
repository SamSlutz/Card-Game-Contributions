from miracle_rogue import AUTOMATON, PREDICATES, make_state, FULL_COUNTS, success_tier1, mana_upper_bound

hand = {
    "Counterfeit Coin": 1,
    "Preparation": 1,
    "Gone Fishin'": 1,
    "Oh, Manager!": 1,
    "Ransack": 2,
    "Ghostly Strike": 1
}

deck = FULL_COUNTS.copy()
for name, c in hand.items():
    deck[name] -= c
    if deck[name] == 0:
        del deck[name]

start = make_state(
    mana=5,
    hand=hand,
    deck=deck,
    flags={
        "pp_used": False,
        "prep_charge": 0,
        "prep_target": None,
        "augmerchant_played": False,
        "augmerchant_in_play": False,
    },
)

print("Starting...")
result = AUTOMATON.solve_exact_scalar(start, success_tier1, bound_fn=mana_upper_bound, bound_threshold=4, progress_every=1_000_000)
print("Done!")
print(result)