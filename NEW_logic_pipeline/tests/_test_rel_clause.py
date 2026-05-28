from NEW_logic_pipeline import build_skeleton

cases = [
    ("Laura met John with Sarah.", "should be FACT"),
    ("Students with approval access the lab.", "should be RULE"),
    ("Students who practice daily succeed.", "should be RULE"),
    ("Students without approval cannot access the lab.", "should be RULE"),
    ("The cat sat with the dog.", "should be FACT"),
]

for text, expected in cases:
    s = build_skeleton("P1", text)
    print(f'  {s.kind:15s}  {expected:20s}  "{text}"')
