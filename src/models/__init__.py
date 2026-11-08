"""Model training, evaluation, tracking, quality gate, and registry modules."""

def __getattr__(name: str):
    if name in ("load_model", "register_model", "promote_to_production", "register_production_model", "ProductionModel"):
        from src.models import registry
        if name == "register_production_model":
            return getattr(registry, "register_model")
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["load_model", "register_model", "promote_to_production", "register_production_model", "ProductionModel"]
