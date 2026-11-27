"""Log reports/drift_reference.json to W&B as the artifact delhi-drift-reference."""
import wandb

ENTITY = "anikaub20-rv-college"
PROJECT = "delhi-bus-delay"


def main() -> None:
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="monitoring", name="drift-reference")
    art = wandb.Artifact(
        "delhi-drift-reference",
        type="drift-reference",
        description="Per-feature reference stats from training days 1-61 (delhi-data-v2).",
    )
    art.add_file("reports/drift_reference.json")
    run.log_artifact(art)
    run.finish()
    print("uploaded delhi-drift-reference")


if __name__ == "__main__":
    main()
