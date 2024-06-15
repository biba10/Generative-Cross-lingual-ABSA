from dataclasses import dataclass, field


@dataclass(kw_only=True, frozen=True, slots=True)
class Slots:
    """Slots for a single sentence used for evaluation of sequence-to-sequence models."""
    # ACD contains the aspect categories
    acd: set[str] = field(default_factory=set)
    # ATE contains the aspect terms
    ate: set[str] = field(default_factory=set)
    # ACTE contains the aspect categories and aspect terms
    acte: set[tuple[str, str]] = field(default_factory=set)
    # TASD contains the aspect categories, aspect terms and sentiment
    tasd: set[tuple[str, str, str]] = field(default_factory=set)
    # E2E contains the aspect terms and sentiment
    e2e: set[tuple[str, str]] = field(default_factory=set)
    # ACSA contains the aspect categories and sentiment
    acsa: set[tuple[str, str]] = field(default_factory=set)
