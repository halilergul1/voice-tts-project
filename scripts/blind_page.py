"""Self-contained listening pages (audio inlined as mp3 data URIs) and the tally of the listener's answers.

  blind_page.py ab --ours data/eval/RUN_A/wavs --theirs data/eval/RUN_B/wavs --name-theirs "untrained" \
      --sentences data/eval/r7_sentences.txt --n 20 --seed 7 --title "Blind A/B stage 2 round 1" --out page.html
  blind_page.py style --style angry=DIR --style happy=DIR --style sad=DIR --style neutral=DIR --n-per 8 --seed 1 \
      --sentences data/eval/style_neutral.txt --ask-natural --title "Which delivery?" --out page.html
  blind_page.py tally --key page.key.json --answers "ab: 1:A 2:same 3:B"

Wav files are named by sentence index (0000.wav, ...), as eval_tts.py writes them. The answer key goes to
<out>.key.json next to the page and never into the page; the page's "Copy answers" button gives a line the listener
pastes back, which `tally` scores. Picks survive a reload (localStorage, per page). Needs ffmpeg.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttsdata.blindpage import ab_key, parse_answers, style_items, tally_ab, tally_style  # noqa: E402

CSS = """:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e4e3dd;--acc:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#33322f;--acc:#7cc2ad}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:800px;margin:auto}h1{font-size:22px;margin:0 0 4px}.mut{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.row{display:flex;gap:16px;flex-wrap:wrap;margin-top:6px}.pl{display:flex;flex-direction:column;font-size:13px}
audio{height:32px;width:240px;max-width:100%}label{cursor:pointer}.tag{color:var(--acc);font-size:13px}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
#out{width:100%;min-height:60px;font:12px monospace;background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:6px;margin-top:8px;box-sizing:border-box}"""

# groups: list of (answer-line label, radio-name prefix); N cards each
JS = """const N=__N__;const G=__G__;const S=__S__;
function load(){try{return JSON.parse(localStorage.getItem(S)||'{}')}catch(e){return {}}}
function save(a){try{localStorage.setItem(S,JSON.stringify(a))}catch(e){}}
for(const[k,v]of Object.entries(load())){const el=document.querySelector(`input[name="${k}"][value="${v}"]`);if(el)el.checked=true}
document.addEventListener('change',e=>{if(e.target.name){const b=load();b[e.target.name]=e.target.value;save(b)}});
document.getElementById('copy').onclick=()=>{const b=load();let n=0;const parts=[];
for(const[label,p]of G){const s=[];for(let i=1;i<=N;i++){const v=b[p+i]||'-';if(v!=='-'&&p===G[0][1])n++;s.push(i+':'+v)}
parts.push(label+': '+s.join(' '))}
const t=parts.join(' | ');document.getElementById('out').value=t;
document.getElementById('done').textContent=n+' of '+N+' answered';try{navigator.clipboard.writeText(t)}catch(e){}};"""


def b64(p: Path) -> str:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"],
                       capture_output=True, check=True)
    return base64.b64encode(r.stdout).decode()


def audio(p: Path, label: str = "") -> str:
    head = f"<b>{html.escape(label)}</b>" if label else ""
    return f"<div class=pl>{head}<audio controls preload=none src='data:audio/mpeg;base64,{b64(p)}'></audio></div>"


def radios(name: str, options: list[tuple[str, str]]) -> str:
    return "".join(f"<label><input type=radio name={name} value=\"{html.escape(v)}\"> {html.escape(t)}</label>"
                   for v, t in options)


def write_page(out: Path, title: str, intro: str, cards: list[str], groups: list[tuple[str, str]], key: dict) -> None:
    js = (JS.replace("__N__", str(len(cards))).replace("__G__", json.dumps(groups))
          .replace("__S__", json.dumps(f"blind_page:{out.stem}")))
    page = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><main><h1>{html.escape(title)}</h1>"
            f"<p class=mut>{html.escape(intro)} Your picks stay in this browser if you reload. At the end press Copy "
            f"answers and paste the text back in the chat.</p>{''.join(cards)}"
            f"<p><button id=copy>Copy answers</button> <span id=done class=mut></span></p>"
            f"<textarea id=out readonly></textarea><script>{js}</script></main>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    out.with_suffix(".key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB, {len(cards)} cards; key {out.with_suffix('.key.json')}")


def read_sentences(p: str | None) -> list[str]:
    return [l.strip() for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()] if p else []


def cmd_ab(a) -> None:
    sentences = read_sentences(a.sentences)
    n = a.n or len(sentences)
    pairs = []
    for j in range(n):
        ours, theirs = Path(a.ours) / f"{j:04d}.wav", Path(a.theirs) / f"{j + a.theirs_offset:04d}.wav"
        for p in (ours, theirs):
            if not p.exists():
                raise SystemExit(f"missing {p}")
        pairs.append((j, ours, theirs))
    random.Random(a.seed).shuffle(pairs)
    sides = ab_key(n, a.seed)
    cards, key = [], []
    for i, ((j, ours, theirs), side) in enumerate(zip(pairs, sides), 1):
        first, second = (ours, theirs) if side == "A" else (theirs, ours)
        text = html.escape(sentences[j]) if j < len(sentences) else ""
        cards.append(f"<div class=card><div class=mut>{i} / {n}</div><div>{text}</div>"
                     f"<div class=row>{audio(first, 'A')}{audio(second, 'B')}</div><div class=row>"
                     + radios(f"q{i}", [("A", "A better"), ("B", "B better"), ("same", "same")]) + "</div></div>")
        key.append({"card": i, "sentence": j, "ours": side})
    intro = a.intro or (f"{n} sentences. One side is {a.name_theirs}, the other is ours; which side is A changes at "
                        "random. Pick the one that sounds better overall (natural Turkish, clear voice, no robotic or "
                        "thin sound), or 'same'.")
    write_page(Path(a.out), a.title, intro, cards, [("ab", "q")],
               {"kind": "ab", "name_theirs": a.name_theirs, "ours": a.ours, "theirs": a.theirs, "cards": key})


def cmd_style(a) -> None:
    dirs = dict(s.split("=", 1) for s in a.style)
    by = {name: [str(p) for p in sorted(Path(d).glob("*.wav"))] for name, d in dirs.items()}
    items = style_items(by, a.n_per, a.seed)
    sentences = read_sentences(a.sentences)
    names = list(dirs)
    cards, groups = [], [("style", "q")] + ([("natural", "n")] if a.ask_natural else [])
    for i, it in enumerate(items, 1):
        stem = Path(it["wav"]).stem
        text = sentences[int(stem)] if stem.isdigit() and int(stem) < len(sentences) else ""
        shown = f" <span class=tag>{html.escape(it['style'])}</span>" if a.show_labels else ""
        nat = ("<div class=row><span class=mut>natural?</span>" + radios(f"n{i}", [("yes", "yes"), ("no", "no")])
               + "</div>") if a.ask_natural else ""
        cards.append(f"<div class=card><div class=mut>{i} / {len(items)}{shown}</div><div>{html.escape(text)}</div>"
                     f"<div class=row>{audio(Path(it['wav']))}</div><div class=row>"
                     + radios(f"q{i}", [(s, s) for s in names]) + f"</div>{nat}</div>")
    intro = a.intro or ("Each clip asks for one delivery. Pick the one you hear"
                        + (" and say whether it sounds natural." if a.ask_natural else "."))
    write_page(Path(a.out), a.title, intro, cards, groups,
               {"kind": "style", "styles": names, "dirs": dirs, "items": items})


def cmd_tally(a) -> None:
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    sections = {}
    for part in a.answers.split("|"):
        label, _, rest = part.partition(":")
        sections[label.strip()] = parse_answers(rest)
    if key["kind"] == "ab":
        ans = sections.get("ab", {})
        res = tally_ab([c["ours"] for c in key["cards"]], [ans.get(c["card"], "-") for c in key["cards"]])
        res["name_theirs"] = key["name_theirs"]
    else:
        ans = sections.get("style", {})
        items = key["items"]
        res = tally_style(items, [ans.get(i, "-") for i in range(1, len(items) + 1)])
        res["confusion"] = {f"{t} -> {p}": n for (t, p), n in sorted(res["confusion"].items())}
        if "natural" in sections:
            nat = sections["natural"]
            per = {}
            for i, it in enumerate(items, 1):
                if nat.get(i) in ("yes", "no"):
                    d = per.setdefault(it["style"], {"yes": 0, "n": 0})
                    d["n"] += 1
                    d["yes"] += nat[i] == "yes"
            res["natural"] = per
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ab = sub.add_parser("ab", help="blind A/B page: ours vs another system, same sentences")
    ab.add_argument("--ours", required=True)
    ab.add_argument("--theirs", required=True)
    ab.add_argument("--name-theirs", required=True)
    ab.add_argument("--theirs-offset", type=int, default=0, help="their wav index = sentence index + offset")
    ab.add_argument("--n", type=int, default=0, help="pairs (default: every sentence)")
    st = sub.add_parser("style", help="style (or tag) identification page")
    st.add_argument("--style", action="append", required=True, help="NAME=DIR, repeat; NAME is also the answer")
    st.add_argument("--n-per", type=int, default=8)
    st.add_argument("--ask-natural", action="store_true")
    st.add_argument("--show-labels", action="store_true", help="not blind: print each clip's style (data checks)")
    for p in (ab, st):
        p.add_argument("--sentences", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--title", required=True)
        p.add_argument("--intro", default=None)
        p.add_argument("--out", required=True)
    ta = sub.add_parser("tally", help="score a pasted answer line against the page's key")
    ta.add_argument("--key", required=True)
    ta.add_argument("--answers", required=True)
    a = ap.parse_args()
    {"ab": cmd_ab, "style": cmd_style, "tally": cmd_tally}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
