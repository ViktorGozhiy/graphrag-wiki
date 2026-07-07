# ABOUTME: Closed schema of node and relation types for knowledge-graph extraction over the corpus.
# ABOUTME: Renders a prompt block that steers the extractor to this fixed vocabulary.

NODE_TYPES = {
    "Person": "A specific person: emperor, general, writer, official, or religious figure.",
    "Place": "A geographic location: city, province, region, or territory.",
    "Organization": (
        "A formal body or social order: the Senate, the army, a legion, "
        "or a social class such as the equites."
    ),
    "Event": "A datable occurrence or period: war, battle, plague, crisis, reign, or era.",
    "Law": "A specific law, edict, decree, or legal code.",
    "Concept": (
        "An abstract institution, right, practice, or idea: citizenship, "
        "taxation, slavery, or the imperial cult."
    ),
}

ANY = list(NODE_TYPES)

RELATION_TYPES = {
    "RULED": {
        "sources": ["Person", "Organization"],
        "targets": ["Place", "Organization"],
        "description": "The person or body governed or held authority over the place or body.",
    },
    "MEMBER_OF": {
        "sources": ["Person"],
        "targets": ["Organization"],
        "description": "The person belonged to the body or social order.",
    },
    "PART_OF": {
        "sources": ["Organization", "Place"],
        "targets": ["Organization", "Place"],
        "description": "The source is structurally contained within the target.",
    },
    "LOCATED_IN": {
        "sources": ["Place", "Event", "Organization"],
        "targets": ["Place"],
        "description": "The source is situated in the target place.",
    },
    "ISSUED": {
        "sources": ["Person"],
        "targets": ["Law"],
        "description": "The person enacted or promulgated the law or edict.",
    },
    "GRANTED": {
        "sources": ["Law", "Person"],
        "targets": ["Concept"],
        "description": "The source conferred the right, status, or concept.",
    },
    "FOUNDED": {
        "sources": ["Person"],
        "targets": ["Place", "Organization"],
        "description": "The person established the place or body.",
    },
    "PARTICIPATED_IN": {
        "sources": ["Person", "Organization"],
        "targets": ["Event"],
        "description": "The source took part in the event.",
    },
    "SUCCEEDED": {
        "sources": ["Person"],
        "targets": ["Person"],
        "description": "The source followed the target in office or succession.",
    },
    "INFLUENCED": {
        "sources": ANY,
        "targets": ANY,
        "description": "The source affected, changed, or contributed to the target.",
    },
}


def _endpoints(node_types):
    if set(node_types) == set(NODE_TYPES):
        return "any"
    return " | ".join(node_types)


def schema_prompt_block():
    """Render the schema as a text block for the extraction prompt."""
    lines = ["Node types:"]
    for name, description in NODE_TYPES.items():
        lines.append(f"- {name}: {description}")
    lines.append("")
    lines.append("Relation types (source -> target):")
    for name, spec in RELATION_TYPES.items():
        arrow = f"{_endpoints(spec['sources'])} -> {_endpoints(spec['targets'])}"
        lines.append(f"- {name} ({arrow}): {spec['description']}")
    return "\n".join(lines)
