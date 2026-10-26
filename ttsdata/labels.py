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
