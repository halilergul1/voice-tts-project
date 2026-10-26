"""Turkish label text written into training manifests: style prefixes and inline non-verbal tags.

VoxCPM2 reads a parenthesised instruction at the start of the text; inline square-bracket tags mark a non-verbal
sound at that position in the audio. Both are plain text for the model, so they must be in the corpus language.
"""
from __future__ import annotations

STYLE_PREFIX = {
    "crying": "(ağlayarak)", "shouting": "(bağırarak)", "whispering": "(fısıldayarak)", "laughing": "(gülerek)",
    "screaming": "(çığlık atarak)", "pained": "(acı içinde)", "scared": "(korkarak)", "angry": "(öfkeyle)",
    "happy": "(neşeyle)", "sad": "(üzgün bir sesle)", "surprised": "(şaşkınlıkla)", "sarcastic": "(alaycı bir tonla)",
    "excited": "(heyecanla)", "pleading": "(yalvararak)", "breathless": "(nefes nefese)", "sighing": "(iç çekerek)",
    "stuttering": "(kekeleyerek)",
}
EVENT_TAG = {
    "laugh": "[gülme]", "cry": "[ağlama]", "breath": "[nefes]", "sigh": "[iç çekme]", "cough": "[öksürük]",
    "scream": "[çığlık]", "moan": "[inleme]", "yawn": "[esneme]", "sniff": "[burun çekme]", "snore": "[horlama]",
    "throat_clear": "[boğaz temizleme]", "shout": "[bağırış]",
}
# sounds that occur right next to a spoken line and are worth re-cutting a clip for (snoring is not speech-adjacent)
EXTENDABLE_TAGS = {"laugh", "cry", "breath", "sigh", "cough", "scream", "moan", "yawn", "sniff", "throat_clear", "shout"}


def tag_text(text: str, tag: str, edge: str) -> str:
    """'Hadi!' + laugh at the start -> '[gülme] Hadi!'; at the end -> 'Hadi! [gülme]'."""
    t = EVENT_TAG[tag]
    text = text.strip()
    return f"{t} {text}" if edge == "start" else f"{text} {t}"

# styles that must not be trained as plain text in Stage 1 (every style that has a prefix)
STRONG_STYLES = frozenset(STYLE_PREFIX)
_EVIDENCE_RANK = {"laugh_syllables": 3, "tagger": 2, "clean": 1, None: 0}


def allowed_labels(styles: list[str], tags: list[str], min_style: int, min_tag: int) -> tuple[set[str], set[str]]:
    """Styles / event tags that occur often enough in a slice to be worth a label (rarer ones stay metadata)."""
    from collections import Counter

    cs, ct = Counter(styles), Counter(tags)
    return ({s for s, n in cs.items() if n >= min_style and s in STYLE_PREFIX},
            {t for t, n in ct.items() if n >= min_tag and t in EVENT_TAG})


def labelled_text(text: str, tags: dict | None, styles_ok: set[str], events_ok: set[str]) -> str:
    """Training text with a Turkish style prefix and inline tags at a merged clip's pauses.

    tags: the clip's soundtags.jsonl row (style list; events with where "gap:k" carry the character offset of the
    pause). Only styles in styles_ok and tags in events_ok are written; events the subtitle attributes to another
    person (who_match False) and events without a known position are never written."""
    out = text
    if not tags:
        return out
    inserts = []
    for e in tags.get("events", []):
        if (e.get("where") or "").startswith("gap:") and e.get("char") is not None and e.get("who_match") is not False \
                and e.get("tag") in events_ok and e["tag"] in EVENT_TAG:
            inserts.append((int(e["char"]), EVENT_TAG[e["tag"]]))
    for char, tag in sorted(inserts, reverse=True):
        out = f"{out[:char]} {tag}{out[char:]}"
    styles = [s for s in STYLE_PREFIX if s in tags.get("style", []) and s in styles_ok]
    if styles:
        out = f"({', '.join(STYLE_PREFIX[s][1:-1] for s in styles)}) {out}"
    return out


def oversample(rows: list[dict], repeat: int, max_share: float, seed: int = 0) -> list[dict]:
    """Rows with row["labelled"] True repeated `repeat` times in total, capped so that labelled rows stay at or
    under max_share of the result; extra copies are appended after the original rows."""
    import random

    lab = [r for r in rows if r.get("labelled")]
    if repeat <= 1 or not lab:
        return list(rows)
    n_plain = len(rows) - len(lab)
    cap = int(max_share * n_plain / ((1.0 - max_share) * len(lab))) if max_share < 1.0 else repeat
    r = max(1, min(repeat, cap))
    extra = [dict(x) for x in lab for _ in range(r - 1)]
    random.Random(seed).shuffle(extra)
    return list(rows) + extra
