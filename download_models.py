import wandb

api = wandb.Api()

PROJECT = "davidsc2-university-of-illinois-urbana-champaign/harm models"

MODELS = [
    "bert_harm_binary",
    "bert_obfus_binary",
    "roberta_harm_binary",
    "roberta_obfus_binary",
    "deberta_harm_binary",
    "deberta_obfus_binary",
]

for name in MODELS:
    print(f"Downloading {name}...")
    artifact = api.artifact(f"{PROJECT}/{name}:latest")
    artifact.download(root=f"models/{name}")
    print(f"  Saved to models/{name}")

print("\nAll models downloaded.")
