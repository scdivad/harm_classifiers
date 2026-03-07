"""Download model artifacts from Weights & Biases.

Reads artifact paths from a text file (one per line). Each line can be:
  - Full path:  entity/project/name:version
  - Short name: name (uses default project)

Optionally specify a local directory after the artifact path:
  entity/project/name:version  models/my_model

If no local directory is given, saves to models/<artifact_name>.

Usage:
  python download_models.py                    # uses models.txt
  python download_models.py my_artifacts.txt   # custom file
"""
import os
import sys
import wandb

DEFAULT_PROJECT = "davidsc2-university-of-illinois-urbana-champaign/harm models"
DEFAULT_FILE = "models.txt"

def main():
    models_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    if not os.path.exists(models_file):
        print(f"Error: '{models_file}' not found")
        print(f"Usage: python download_models.py [models_file]")
        sys.exit(1)

    api = wandb.Api()
    os.makedirs("models", exist_ok=True)

    with open(models_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            artifact_path = parts[0]

            # If no slashes, assume it's a short name under the default project
            if "/" not in artifact_path:
                artifact_path = f"{DEFAULT_PROJECT}/{artifact_path}"
            # Add :latest if no version specified
            if ":" not in artifact_path:
                artifact_path += ":latest"

            # Determine local save directory
            if len(parts) >= 2:
                save_dir = parts[1]
            else:
                # Use artifact name (without version) as directory
                artifact_name = artifact_path.rsplit("/", 1)[-1].split(":")[0]
                save_dir = f"models/{artifact_name}"

            print(f"Downloading {artifact_path}")
            print(f"  -> {save_dir}")
            artifact = api.artifact(artifact_path)
            artifact.download(root=save_dir)
            print(f"  Done.")

    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()
