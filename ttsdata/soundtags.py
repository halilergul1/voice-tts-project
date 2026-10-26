"""Mine the sound descriptions of accessible (SDH) subtitles as free emotion / non-verbal labels.

Engelsiz subtitles describe sounds in parentheses: standalone cues such as "(Gülme sesi)", "(Hatice ağlama sesi)",
"(Eşref iç çekti)", manner labels glued to a speech line such as "(Ağlayarak) Neden?", and noise cues such as
"(Gerilim müziği)", "(Fonda kahkaha)", "(Üst üste konuşmalar)", "(***)". The segmenter drops all of them; this module
keeps them with their timestamps (mine_cues), classifies them (classify), and attaches them to dataset clips
(attach): a speaking style for the clip, non-verbal events inside / between / next to the clip, and flags for
noise that makes a clip a poor training example.
"""
from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass

from .subtitles import MUSIC_CHARS, PAREN_BLOCK_RE, TAG_RE, Cue

# ------------------------------------------------------------------------------------------- vocabulary
# vocal events: exact inflected forms (or prefixes ending in '*'), so names such as Gülden / Ayşegül never match
VOCAL_FORMS = {
    "laugh": ["gülme", "gülmesi", "gülmeler", "gülmeleri", "gülüyor", "gülüyorlar", "güldü", "güldüler", "gülüşme",
              "gülüşmeler", "gülüşüyorlar", "kahkaha*", "kıkırda*", "kikirde*"],
    "cry": ["ağlama", "ağlaması", "ağlamalar", "ağlıyor", "ağlıyorlar", "ağladı", "ağladılar", "ağlamaya", "hıçkırık*",
            "hıçkırıyor", "hıçkırdı", "hüngür*"],
    "breath": ["nefes*", "soluyor", "soluma", "soluk"],
    "sigh": ["iç çek*", "of çek*"],
    "cough": ["öksür*"],
    "scream": ["çığlık*", "haykırıyor", "haykırdı", "haykırdılar", "haykırış*"],
    "shout": ["bağırma", "bağırması", "bağırmalar", "bağırıyor", "bağırıyorlar", "bağırdı", "bağırdılar", "bağırış*"],
    "moan": ["inleme", "inlemesi", "inliyor", "inliyorlar", "inledi", "inlemeler", "inilti*"],
    "yawn": ["esneme", "esniyor", "esnedi"],
    "sniff": ["burnunu çek*"],
    "hum": ["mırıldanıyor", "mırıldandı", "mırıltı*"],
    "whistle": ["ıslık*"],
    "snore": ["horluyor", "horladı", "horlama*", "horultu*"],
    "throat_clear": ["boğazını temizl*", "boğaz temizle*"],
    "retch": ["öğürüyor", "öğürdü"],
}
# speaking manner: adverbial / adjectival forms that describe how a line is spoken
MANNER_FORMS = {
    "crying": ["ağlayarak", "ağlamaklı", "hıçkırarak", "hüngür"],
    "laughing": ["gülerek", "kahkahayla", "gülüşerek"],
    "smiling": ["gülümseyerek"],
    "shouting": ["bağırarak", "bağırırcasına"],
    "screaming": ["haykırarak", "çığlıkla"],
    "whispering": ["fısıl*", "fısıltı*", "fısıldar*"],
    "pained": ["inleyerek", "acıyla", "acı içinde"],
    "sighing": ["iç çekerek"],
    "breathless": ["nefes nefese", "soluk soluğa"],
    "scared": ["korkarak", "korkuyla", "korkmuş", "ürkek", "ürkerek"],
    "trembling": ["titreyerek", "titrek"],
    "angry": ["sinirle", "sinirli", "sinirlenerek", "öfkeyle", "öfkeli", "kızgın", "kızgınlıkla", "kızarak", "hiddetle"],
    "happy": ["sevinçle", "neşeyle", "neşeli", "mutlu", "coşkuyla"],
    "sad": ["üzgün", "üzüntüyle", "hüzünle", "hüzünlü", "buruk"],
    "surprised": ["şaşkın", "şaşkınlıkla", "hayretle"],
    "sarcastic": ["alaycı", "alayla", "alay ederek", "iğneleyici"],
    "excited": ["heyecanla", "heyecanlı"],
    "pleading": ["yalvararak"],
    "stuttering": ["kekeleyerek"],
    "drunk": ["sarhoş"],
    "dialect": ["ağzıyla", "şivesiyle"],
}
# a vocal event that labels a speech line describes how the line is spoken
VOCAL_TO_MANNER = {"laugh": "laughing", "cry": "crying", "breath": "breathless", "sigh": "sighing", "cough": "coughing",
                   "scream": "screaming", "shout": "shouting", "moan": "pained", "yawn": "yawning", "sniff": "sniffling",
                   "hum": "humming", "whistle": "whistling", "snore": "sleepy", "throat_clear": "throat_clearing",
                   "retch": "retching"}
_L = r"(?<!\w)"  # a vocabulary stem must start a word: "cep" must not fire inside "Recep"
MUSIC_RE = re.compile(_L + r"(?:müzi|şarkı|melodi|jenerik|ezgi|ağıt|türkü|keman|piyano|gitar|bağlama|ney\b|davul|zurna"
                      r"|ilahi|marş|enstrüman|çalmaya|söylüyor|söylen|okunuyor|ezan|mevlit|orkestra)|[\"“”„«»]")
SFX_RE = re.compile(_L + r"(?:telefon|kapı|ayak sesi|ayak sesleri|silah|kurşun|araba|korna|siren|zil|rüzgar|rüzgâr|yağmur|gök"
                    r"|köpek|kuş|kedi|at kişne|cam|su sesi|motor|fren|tabak|bardak|çatal|kaşık|tık|vur|patla|çarp|kırıl"
                    r"|arama tonu|mesaj|bildirim|alarm|saat|televizyon|radyo|anons|ortam|trafik|dalga|deniz|ateş|çıtır"
                    r"|kayıt|tuş|klavye|kâğıt|kağıt|zarf|anahtar|kilit|fermuar|sandalye|masa|cep|ton\b|tonu|düdü"
                    r"|cihaz|makine|vızıl|asansör|namlu|mermi|geçiş)")
CHANNEL_RE = re.compile(_L + r"(?:anons\b|tv\b|televizyon|radyo|telsiz|hoparlör|megafon|telefondan|telefondaki)")
# a label before a line is usually a speaker name: only call it music / sfx / channel when it names a sound
SOUND_MARKER_RE = re.compile(_L + r"(?:ses|sesi|sesleri|sesiyle|çal(?:ıyor|ıyorlar|dı|maya|an|ar)|müzi\w*|şarkı\w*|jenerik"
                             r"|türkü\w*|ağıt|devam ediyor|geldi|açıl\w*|kapan\w*|vurul\w*|vurma\w*|vuruyor|zili|tonu"
                             r"|anons|tv|televizyon|radyo|telsiz|hoparlör|megafon|telefondan|telefondaki)(?!\w)|[\"“”„«»]")
# "(Erkek sesi)", "(Nisan'ın sesi)": a voice, i.e. a speaker label, not a sound effect
VOICE_RE = re.compile(r"^(?:(?:erkek|kadın|çocuk|adam|kız|bebek|yaşlı|genç|\w+['’](?:ın|in|un|ün|nın|nin|nun|nün))\s+)+"
                      r"ses(?:i)?$")
ADVERBIAL_RE = re.compile(r"(?:arak|erek|la|le)$")         # ağlayarak, fısıltıyla, sinirle
FINITE_RE = re.compile(r"(?:yor|yorlar|[dt][ıiuü](?:lar|ler)?)$")  # ağlıyor, güldü, öksürdü, iç çekti
NOMINAL_RE = re.compile(r"(?:me|ma|mesi|ması|meler|malar|meleri|maları|ük|ık|ükler|ıklar)$")
# sounds that interrupt speech rather than colour it: an event in any verb form ("Öksürüyor" is still a cough)
DISCRETE_TAGS = {"cough", "throat_clear", "sniff", "yawn", "snore", "whistle", "retch"}
NOUN_STEMS = ("kahkaha", "hıçkırık", "inilti", "nefes", "öksürük", "çığlık", "ıslık", "horultu", "mırıltı", "soluk")
CENSOR_RE = re.compile(r"^(?:[*\s]+|bi+p[!.]*|bip sesi)$")
FOREIGN_RE = re.compile(r"\b(almanca|fransızca|ingilizce|rusça|arapça|farsça|yunanca|ermenice|italyanca|ispanyolca|kürtçe"
                        r"|lazca|çince|japonca|korece|bulgarca|sırpça|boşnakça|gürcüce|yabancı dil\w*)\b")
UNINTELLIGIBLE_RE = re.compile(r"anlaşılma(yan|z)|anlaşılmıyor|duyulmuyor|duyulmayan|seçilemiyor")
OVERLAP_RE = re.compile(r"üst üste|konuşmalar|kalabalık|uğultu|gürültü|hep bir ağızdan|bağrışmalar|tezahürat|alkış")
BACKGROUND_RE = re.compile(r"^(fonda|uzaktan|arkada|dışarıdan|içeriden)\b")
ADVERB_WORDS = {"derin", "hızlı", "hızla", "uzun", "kısa", "zorla", "derince", "sessizce", "yavaşça", "hafifçe", "içten",
                "acı", "kesik", "kesik kesik", "sürekli", "yine", "tekrar", "birden", "aniden", "yüksek", "alçak",
                "burundan", "burnundan", "ağzından", "ağızdan", "göğsünden", "zor", "rahat", "sık", "sık sık"}
ROLE_WORDS = {"ses", "dış", "iç", "ve"}
_TOKEN_RE = re.compile(r"[\wçğıöşüâîûÇĞİÖŞÜÂÎÛ'*]+", re.UNICODE)


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")


def _compile(forms: dict[str, list[str]]) -> list[tuple[str, re.Pattern]]:
    out = []
    for tag, fs in forms.items():
        for f in fs:
            body = re.escape(f[:-1]) + r"[\wçğıöşü]*" if f.endswith("*") else re.escape(f)
            out.append((tag, re.compile(rf"(?<![\wçğıöşü]){body}(?![\wçğıöşü])")))
    return out


_VOCAL = _compile(VOCAL_FORMS)
_MANNER = _compile(MANNER_FORMS)


def _first(patterns: list[tuple[str, re.Pattern]], low: str) -> tuple[str, re.Match] | None:
    best = None
    for tag, p in patterns:
        m = p.search(low)
        if m and (best is None or m.start() < best[1].start()):
            best = (tag, m)
    return best


@dataclass
class Desc:
    kind: str  # vocal | manner | background | music | sfx | overlap | censored | silence | foreign | unintelligible
    #            | channel | other
    tag: str | None = None
    who: str | None = None
    manner: str | None = None
    nominal: bool = False     # a sound noun ("Öksürme sesi", "Kahkaha"): a discrete event, not a way of speaking


@dataclass
class Event:
    start: float
    end: float
    desc: str
    kind: str
    tag: str | None
    who: str | None
    manner: str | None
    position: str             # standalone (cue without speech) | lead (label before a line) | inline (inside a line)
    nominal: bool = False

    def to_json(self) -> dict:
        return asdict(self)


def _who(original: str, upto: int) -> str | None:
    """Name-like tokens before the keyword: 'Hatice', 'Afra ses', 'Erkek 2', 'Afra ve Nisan'; else None."""
    toks = original[:upto].strip(" ,;:-").split()
    if not toks or len(toks) > 4 or not (toks[0][:1].isupper() or toks[0].isdigit()):
        return None
    for t in toks:
        low = tr_lower(t)
        if low in ADVERB_WORDS or _first(_MANNER, low):
            return None
        if not (t[:1].isupper() or t.isdigit() or low in ROLE_WORDS) or not re.fullmatch(r"[\wçğıöşüÇĞİÖŞÜÂÎÛâîû']+", t):
            return None
    return " ".join(toks)


def _char_offset(original: str, low: str, pos: int) -> int:
    """Map an offset in tr_lower(original) back to original (lowercasing 'İ' may change length)."""
    if len(low) == len(original):
        return pos
    acc = 0
    for i, ch in enumerate(original):
        if acc >= pos:
            return i
        acc += len(tr_lower(ch))
    return len(original)


def _is_nominal(low: str, matched: str) -> bool:
    if re.search(r"(?<!\w)ses(?:i|leri)?(?!\w)", low):
        return True
    last, first = matched.split()[-1], matched.split()[0]
    return bool(NOMINAL_RE.search(last)) or (first.startswith(NOUN_STEMS) and not FINITE_RE.search(last))


def classify(desc: str, position: str = "standalone") -> Desc:
    """Classify one parenthetical. position "lead" / "inline" (a label on a speech line) is stricter, because such
    labels are usually speaker names: a lone capitalised word must be an adverb ("Ağlayarak") or a finite verb
    ("Ağlıyor") to count as a sound or manner, and music / sfx / channel need a sound word ("sesi", "çalıyor")."""
    d = re.sub(r"\s+", " ", desc).strip()
    low = tr_lower(d)
    if not low:
        return Desc("other")
    if "***" in d or CENSOR_RE.match(low):
        return Desc("censored")
    labelled = position != "standalone"
    if labelled and len(d.split()) == 1 and d[:1].isupper() and not (ADVERBIAL_RE.search(low) or FINITE_RE.search(low)):
        return Desc("other")
    bg = BACKGROUND_RE.match(low)
    if bg:
        rest = low[bg.end():].strip()
        if MUSIC_RE.search(rest):
            return Desc("music")
        if OVERLAP_RE.search(rest) or "konuşma" in rest:
            return Desc("overlap")
        v = _first(_VOCAL, rest)
        return Desc("background", v[0] if v else ("applause" if "alkış" in rest else None))
    if FOREIGN_RE.search(low):
        return Desc("foreign")
    if UNINTELLIGIBLE_RE.search(low):
        return Desc("unintelligible")
    if VOICE_RE.match(low):
        return Desc("other")
    names_a_sound = not labelled or bool(SOUND_MARKER_RE.search(low))
    if names_a_sound and CHANNEL_RE.search(low):
        return Desc("channel")
    if names_a_sound and MUSIC_RE.search(low):
        return Desc("music")
    if low.startswith("sessizlik"):
        return Desc("silence")
    if OVERLAP_RE.search(low):
        return Desc("overlap")
    v = _first(_VOCAL, low)
    m = _first(_MANNER, low)
    if v and not (m and m[1].start() <= v[1].start() < m[1].end()):
        manner = m[0] if m and m[1].end() <= v[1].start() else None
        who = None if manner else _who(d, _char_offset(d, low, v[1].start()))
        return Desc("vocal", v[0], who, manner, v[0] in DISCRETE_TAGS or _is_nominal(low, v[1].group(0)))
    if m:
        return Desc("manner", m[0], _who(d, _char_offset(d, low, m[1].start())))
    if names_a_sound and (SFX_RE.search(low) or re.search(r"(?<!\w)ses(?:i|leri)(?!\w)", low)):
        return Desc("sfx")
    return Desc("other")


def hint_style(hint: str | None) -> tuple[str | None, list[str], list[str]]:
    """Split a subtitle speaker label into (speaker name, speaking styles, flags)."""
    if not hint:
        return None, [], []
    c = classify(hint, position="lead")
    if c.kind == "censored":
        return None, [], ["censored"]
    if c.kind == "vocal" and c.nominal:
        return c.who, [], []  # "(Gülme sesi)": a sound before the line; mine_cues reports it as an event
    if c.kind == "vocal":
        styles = {VOCAL_TO_MANNER[c.tag]} | ({c.manner} if c.manner else set())
        return c.who, sorted(styles), []
    if c.kind == "manner":
        return c.who, [c.tag], []
    return hint, [], []


# ------------------------------------------------------------------------------------------- mining
def mine_cues(cues: list[Cue]) -> list[Event]:
    """Every sound description of a subtitle file, classified, with its cue time and position."""
    out: list[Event] = []
    for cue in cues:
        lines = [html.unescape(TAG_RE.sub("", l)).strip() for l in cue.lines]
        lines = [l for l in lines if l]
        if not lines:
            continue
        joined = "\n".join(lines)
        if any(ch in joined for ch in MUSIC_CHARS):
            out.append(Event(cue.start, cue.end, "♪", "music", None, None, None, "standalone"))
            continue
        blocks = PAREN_BLOCK_RE.findall(joined)
        if not blocks:
            continue
        rest = PAREN_BLOCK_RE.sub(" ", joined).strip(" \n-–—.,!?:;")
        for b in blocks:
            desc = re.sub(r"\s+", " ", b[1:-1]).strip()
            if not rest:
                position = "standalone"
            elif any(re.sub(r"^[-–—]\s*", "", l).startswith(b) for l in lines):
                position = "lead"
            else:
                position = "inline"
            c = classify(desc, position=position)
            if c.kind in ("other", "silence") and position != "standalone":
                continue  # speaker names and the like
            if c.kind == "other":
                continue  # visual descriptions
            out.append(Event(cue.start, cue.end, desc, c.kind, c.tag, c.who, c.manner, position, c.nominal))
    return out


# ------------------------------------------------------------------------------------------- attaching
FLAG_OF_KIND = {"music": "music", "sfx": "sfx", "overlap": "overlap", "censored": "censored", "silence": "silence",
                "foreign": "foreign_speech", "unintelligible": "unintelligible", "channel": "channel"}


def _same_person(a: str | None, b: str | None) -> bool | None:
    if not a or not b:
        return None
    return tr_lower(a).split()[0] == tr_lower(b).split()[0]


def _own_cue(ev: Event, a: float, b: float) -> bool:
    """A label on a speech line belongs to the clip cut from that line: it covers half of the shorter of the two."""
    ov = min(b, ev.end) - max(a, ev.start)
    return ov >= 0.5 * min(max(ev.end - ev.start, 1e-6), b - a)


def _styles_of(a: float, b: float, hint: str | None, events: list[Event]) -> set[str]:
    """How the speech in [a, b] is delivered: the line's own label (hint) plus manner labels on its cue."""
    styles = set(hint_style(hint)[1])
    for ev in events:
        speaks = ev.kind == "manner" or (ev.kind == "vocal" and ev.position != "standalone" and not ev.nominal)
        if speaks and _own_cue(ev, a, b):
            styles.add(ev.tag if ev.kind == "manner" else VOCAL_TO_MANNER[ev.tag])
            if ev.manner:
                styles.add(ev.manner)
    return styles


def attach(clip: dict, events: list[Event], adjacent: float = 1.5, gap_tol: float = 0.3) -> dict:
    """Tags of one clip from the episode's events.

    clip: {"id", "orig_start", "orig_end", "speaker_hint"} plus, for a merged clip, "member_spans" (episode-time
    [start, end] per member), "member_text_ends" (character offset in "text" where each member's text ends) and
    "member_hints". A merged clip gets a style only when every member has it ("style_members" lists them all).
    Times "t" / "dt" are rounded to 0.1 s: subtitle cues are only that precise (often less).
    """
    a, b = float(clip["orig_start"]), float(clip["orig_end"])
    name, _, flags = hint_style(clip.get("speaker_hint"))
    flags = set(flags)
    spans, ends = clip.get("member_spans") or [], clip.get("member_text_ends") or []
    if spans:
        hints = clip.get("member_hints") or [None] * len(spans)
        per = [sorted(_styles_of(sa, sb, h, events)) for (sa, sb), h in zip(spans, hints)]
        styles = set(per[0]).intersection(*map(set, per[1:])) if per else set()
        for h in hints:
            flags |= set(hint_style(h)[2])
    else:
        per = None
        styles = _styles_of(a, b, clip.get("speaker_hint"), events)
    found = []
    for ev in events:
        s, e = ev.start, ev.end
        ov = min(b, e) - max(a, s)
        if ev.kind in FLAG_OF_KIND or ev.kind == "background":
            if ov > 0:
                flags.add(FLAG_OF_KIND.get(ev.kind) or f"bg_{ev.tag or 'other'}")
            continue
        if ev.kind == "manner":
            continue  # a way of speaking: handled by _styles_of
        item = {"tag": ev.tag, "where": None, "start": s, "end": e, "t": None, "dt": None, "char": None, "in_clip": False,
                "who": ev.who, "who_match": _same_person(ev.who, name), "manner": ev.manner, "desc": ev.desc}
        if ev.position != "standalone":
            if ev.nominal and _own_cue(ev, a, b):
                item.update(where="lead")  # "(Öksürme sesi) Ben onu mu diyorum?": the sound comes before the words
                found.append(item)
            continue
        mid = (s + e) / 2
        gap = next((k for k in range(len(spans) - 1) if spans[k][1] - gap_tol <= mid <= spans[k + 1][0] + gap_tol), None)
        if gap is not None:
            item.update(where=f"gap:{gap}", t=round(max(0.0, s - a), 1), in_clip=True,
                        char=ends[gap] if gap < len(ends) else None)
        elif a <= mid <= b and ov >= min(0.3, 0.5 * (e - s)):  # most of the sound falls inside the clip
            item.update(where="inside", t=round(min(max(0.0, s - a), b - a), 1), in_clip=True)
        elif mid < a and a - e <= adjacent:  # mostly before the clip (may graze its first moments)
            item.update(where="before", dt=round(max(0.0, a - e), 1))
        elif mid > b and s - b <= adjacent:  # mostly after it (a laugh cue that starts on the last word)
            item.update(where="after", dt=round(max(0.0, s - b), 1))
        else:
            continue
        found.append(item)
    out = {"id": clip["id"], "style": sorted(styles), "flags": sorted(flags), "events": found}
    if per is not None:
        out["style_members"] = per
    return out


def count_unique_events(tags: list[dict], uid_of: dict[str, str]) -> dict[tuple[str, str], int]:
    """Unique events per (tag, "in clip" | "adjacent"); an event next to two clips counts once."""
    seen: dict[tuple[str, str], set] = {}
    for t in tags:
        for e in t["events"]:
            kind = "in clip" if e["in_clip"] else "adjacent"
            seen.setdefault((e["tag"], kind), set()).add((uid_of[t["id"]], e["start"], e["tag"]))
    return {k: len(v) for k, v in seen.items()}


# ------------------------------------------------------------------------------------------- driver
def _events_near(events: list[Event], starts: list[float], max_len: float, a: float, b: float, pad: float) -> list[Event]:
    import bisect

    lo = bisect.bisect_left(starts, a - pad - max_len)
    hi = bisect.bisect_right(starts, b + pad)
    return events[lo:hi]


def tag_rows(rows: list[dict], events_by_uid: dict[str, list[Event]], adjacent: float) -> list[dict]:
    """attach() for every row, using a time window into each episode's sorted events; rows without tags are left out."""
    out = []
    prepared = {}
    for uid, evs in events_by_uid.items():
        evs = sorted(evs, key=lambda e: e.start)
        prepared[uid] = (evs, [e.start for e in evs], max((e.end - e.start for e in evs), default=0.0))
    for r in rows:
        evs, starts, max_len = prepared.get(r["uid"], ([], [], 0.0))
        t = attach(r, _events_near(evs, starts, max_len, r["orig_start"], r["orig_end"], adjacent), adjacent)
        if t["style"] or t["flags"] or t["events"]:
            out.append(t)
    return out


def _cut_raw(raw: Path, start: float, end: float, dst: Path, sr: int = 24000) -> bool:
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{end - start:.3f}", "-i", str(raw),
           "-ac", "1", "-ar", str(sr), str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def run(cfg, listen: int = 4, seed: int = 1) -> dict:
    """Mine every episode's subtitles, tag the dataset (and data/dataset_long when present), write a report."""
    import json
    import os
    import random
    from collections import Counter, defaultdict
    from pathlib import Path

    from .speakers import load_rows
    from .subtitles import parse

    roots = [("dataset", cfg.dataset_dir, load_rows(cfg))]
    long_meta = cfg.root / "dataset_long" / "metadata.jsonl"
    if long_meta.exists():
        roots.append(("dataset_long", long_meta.parent,
                      [json.loads(l) for l in long_meta.read_text(encoding="utf-8").splitlines() if l.strip()]))
    episodes = sorted({(r["source"], r["uid"]) for _, _, rs in roots for r in rs})
    events_by_uid: dict[str, list[Event]] = {}
    raw_of: dict[str, Path] = {}
    missing_subs: list[str] = []
    for source, uid in episodes:
        ep_dir = cfg.raw_dir / source / uid
        subs = next(ep_dir.glob("subs.*"), None)
        if subs is None:
            print(f"[warn] {uid}: no subtitle file in {ep_dir}; its clips get no tags")
            missing_subs.append(uid)
        events_by_uid[uid] = mine_cues(parse(subs)) if subs else []
        raw = next(ep_dir.glob("audio.*"), None)
        if raw is not None:
            raw_of[uid] = raw
    with (cfg.dataset_dir / "sound_events.jsonl").open("w", encoding="utf-8") as f:
        for uid, evs in events_by_uid.items():
            for e in evs:
                f.write(json.dumps({"uid": uid, **e.to_json()}, ensure_ascii=False) + "\n")

    all_events = [e for evs in events_by_uid.values() for e in evs]
    n_ep = max(1, len(events_by_uid))
    source_of = {uid: source for source, uid in episodes}
    lines = ["# Subtitle sound descriptions (`ttsdata soundtags`)", "",
             f"{len(all_events)} descriptions mined from {len(events_by_uid)} episodes "
             f"(all of them, with times: `sound_events.jsonl`).", "",
             "| kind | count | per episode | top tags / examples |", "|---|---|---|---|"]
    by_kind = defaultdict(list)
    for e in all_events:
        by_kind[e.kind].append(e)
    for kind, evs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        tags = Counter(e.tag for e in evs if e.tag)
        top = ", ".join(f"{t} {n}" for t, n in tags.most_common(8)) or ", ".join(
            f"{d} {n}" for d, n in Counter(tr_lower(e.desc) for e in evs).most_common(4))
        lines.append(f"| {kind} | {len(evs)} | {len(evs) / n_ep:.1f} | {top} |")

    lines += ["", "Standalone vocal events per episode, by source (the TRT catalogue is the one to scale from):", "",
              "| source | episodes | vocal events / episode | top tags (count) |", "|---|---|---|---|"]
    for source in sorted({s for s, _ in episodes}):
        uids = [u for s, u in episodes if s == source]
        vocal = [e for u in uids for e in events_by_uid[u] if e.kind == "vocal" and e.position == "standalone"]
        top = ", ".join(f"{t} {n}" for t, n in Counter(e.tag for e in vocal).most_common(6))
        lines.append(f"| {source} | {len(uids)} | {len(vocal) / max(1, len(uids)):.1f} | {top} |")

    lines += ["", "Timing: subtitle cues are only roughly aligned to the sound (a clip's start differs from its cue's "
              "start by more than 0.5 s for about a fifth of clips), so `inside` / `before` / `after` are estimates and "
              "event times are rounded to 0.1 s. Lead sounds (`where: lead`, e.g. `(Öksürme sesi) Ben...`) come before "
              "the words and are usually trimmed off the clip."]
    if missing_subs:
        lines += ["", f"Episodes without a subtitle file (no tags): {len(missing_subs)}: {', '.join(missing_subs[:10])}"]
    summary = {"episodes": len(events_by_uid), "events": len(all_events), "missing_subs": missing_subs,
               "events_by_kind": {k: len(v) for k, v in by_kind.items()}}
    picks: list[tuple[str, str, Path, str]] = []  # (group, name, source wav, note)
    rng = random.Random(seed)
    for name, root, rs in roots:
        tags = tag_rows(rs, events_by_uid, cfg.soundtag_adjacent)
        with (root / "soundtags.jsonl").open("w", encoding="utf-8") as f:
            for t in tags:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        row_of = {r["id"]: r for r in rs}
        hours = lambda ids: sum(row_of[i]["duration"] for i in ids) / 3600  # noqa: E731
        style = defaultdict(list)
        flags = defaultdict(list)
        in_clip = defaultdict(list)
        for t in tags:
            for s in t["style"]:
                style[s].append(t["id"])
            for fl in t["flags"]:
                flags[fl].append(t["id"])
            for e in t["events"]:
                if e["in_clip"]:
                    in_clip[e["tag"]].append((t["id"], e))
        lines += ["", f"## `{name}/` ({len(rs)} clips)", "",
                  f"`{name}/soundtags.jsonl`: {len(tags)} clips carry a style, an event or a flag.", "",
                  "Speaking style of the clip itself (from the subtitle label of its own line):", "",
                  "| style | clips | hours |", "|---|---|---|"]
        lines += [f"| {s} | {len(v)} | {hours(v):.2f} |" for s, v in sorted(style.items(), key=lambda kv: -len(kv[1]))] \
            or ["| none | | |"]
        lines += ["", "Unique non-verbal events (`in clip`: the sound is inside the clip or in a merged clip's pause; "
                  f"`adjacent`: within {cfg.soundtag_adjacent} s before / after a clip, so not in its audio yet):", "",
                  "| event | in clip | adjacent |", "|---|---|---|"]
        ev_where = count_unique_events(tags, {r["id"]: r["uid"] for r in rs})
        n_of = lambda tag, kind: ev_where.get((tag, kind), 0)  # noqa: E731
        for tag in sorted({k[0] for k in ev_where}, key=lambda t: -(n_of(t, "in clip") + n_of(t, "adjacent"))):
            lines.append(f"| {tag} | {n_of(tag, 'in clip')} | {n_of(tag, 'adjacent')} |")
        lines += ["", "Flags (noise that makes a clip a weaker training example):", "", "| flag | clips | hours |",
                  "|---|---|---|"]
        lines += [f"| {fl} | {len(v)} | {hours(v):.2f} |" for fl, v in sorted(flags.items(), key=lambda kv: -len(kv[1]))]
        summary[name] = {"tagged": len(tags), "style": {k: len(v) for k, v in style.items()},
                         "flags": {k: len(v) for k, v in flags.items()},
                         "events": {f"{k[0]}:{k[1]}": v for k, v in ev_where.items()}}
        wav_root = root / "wavs"
        for s, ids in style.items():
            for cid in rng.sample(ids, min(listen, len(ids))):
                picks.append((f"style_{s}", cid, wav_root / f"{cid}.wav", row_of[cid]["text"]))
        for tag, items in in_clip.items():
            gap_first = sorted(items, key=lambda x: (not x[1]["where"].startswith("gap"), rng.random()))
            for cid, e in gap_first[:listen]:
                picks.append((f"event_{tag}_{e['where'].split(':')[0]}", cid, wav_root / f"{cid}.wav",
                              f"{e['desc']} at {e['t']} s · {row_of[cid]['text']}"))

    listen_dir = cfg.dataset_dir / "listen_tags"
    listen_dir.mkdir(exist_ok=True)
    for p in listen_dir.iterdir():
        if p.is_symlink() or p.suffix == ".wav":
            p.unlink()
    idx = ["# Sound-tag listening sample", "", "| file | what | text / description |", "|---|---|---|"]
    k = 0
    for group, cid, src, note in picks:
        name = f"{k:03d}_{group}_{cid}.wav"
        if src.exists():
            os.symlink(os.path.relpath(src, listen_dir), listen_dir / name)
            idx.append(f"| `{name}` | {group} | {note} |")
            k += 1
    raw_events = defaultdict(list)
    for uid, evs in events_by_uid.items():
        for e in evs:
            if e.kind == "vocal" and e.position == "standalone" and uid in raw_of:
                raw_events[e.tag].append((uid, e))
    for tag, items in sorted(raw_events.items()):
        for uid, e in rng.sample(items, min(listen, len(items))):
            name = f"{k:03d}_raw_{tag}_{uid}_{e.start:.1f}.wav"
            if _cut_raw(raw_of[uid], e.start - 0.3, e.end + 0.3, listen_dir / name):
                idx.append(f"| `{name}` | raw subtitle event: {tag} | ({e.desc}) cue {e.start:.1f}–{e.end:.1f} s, cut ±0.3 s |")
                k += 1
    (listen_dir / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    lines += ["", f"Listening sample: `{listen_dir}/index.md` ({k} files: styled clips, clips with events, and raw "
              "cuts of standalone event cues to check how well subtitle times match the sound).", ""]
    (cfg.dataset_dir / "SOUNDTAGS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summary
