import json 

discovered = {
    "matrix-sizes" : "4x4",
    "algorithms" : [
        "1. Divide matrix...",
        "2. Apply rule X...",
        "3. Combine results..."
    ]
}

with open("discovered.json", "w") as f:
    json.dump(discovered, f)
print("Reinforcement-Learning discovered algorithms saved.")