"""Fetch raw episodes (audio + Turkish subtitles) into data/raw/<source>/<uid>/.

Each raw directory holds:
  audio.<ext>   best available audio stream, untouched
  subs.<vtt|srt> human-made Turkish subtitles
  info.json     title, series, episode, url, duration, source
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests

SERIES_RE = re.compile(r"\s*(\d+)\s*\.?\s*Bölüm.*$", re.IGNORECASE)


def series_from_title(title: str) -> tuple[str, int | None]:
    """'Payitaht 'Abdülhamid' Engelsiz 31.Bölüm' -> ('Payitaht Abdülhamid', 31)."""
    m = re.search(r"(\d+)\s*\.?\s*Bölüm", title, re.IGNORECASE)
    ep = int(m.group(1)) if m else None
    s = re.sub(r"\(.*?\)", " ", title)
    s = re.sub(r"\d+\s*\.?\s*Bölüm", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bEngelsiz\b|\bTRT\b|-", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[\"'“”‘’]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, ep


def _write_info(d: Path, **info) -> None:
    (d / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def is_complete(d: Path) -> bool:
    return (d / "info.json").exists() and any(d.glob("audio.*")) and any(d.glob("subs.*"))


# --------------------------------------------------------------------------- TRT / YouTube
def _js_runtime() -> str:
    """yt-dlp needs a JavaScript runtime for YouTube: prefer deno (its default), else node."""
    for name in ("deno", "node"):
        if shutil.which(name):
            return name
    return "deno"


def download_youtube(video_id: str, raw_dir: Path, js_runtime: str | None = None, sleep: float = 2.0) -> Path | None:
    import yt_dlp

    js_runtime = js_runtime or _js_runtime()

    out = raw_dir / "trt" / video_id
    if is_complete(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    probe_opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "js_runtimes": {js_runtime: {}}}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    subs = info.get("subtitles") or {}
    tr_tracks = [k for k in subs if k == "tr" or k.startswith("tr-")]
    if not tr_tracks:
        print(f"[skip] {video_id}: no human-made Turkish subtitles ({info.get('title')})")
        shutil.rmtree(out, ignore_errors=True)
        return None
    lang = "tr" if "tr" in tr_tracks else tr_tracks[0]
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "js_runtimes": {js_runtime: {}},
        "format": "bestaudio[acodec=opus]/bestaudio/best",
        "outtmpl": str(out / "audio.%(ext)s"),
        "writesubtitles": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "retries": 10,
        "fragment_retries": 10,
        "sleep_interval": sleep,
        "max_sleep_interval": sleep + 3,
        "concurrent_fragment_downloads": 4,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    sub_files = sorted(out.glob("audio.*.vtt"))
    if not sub_files:
        print(f"[fail] {video_id}: subtitle file missing after download")
        return None
    sub_files[0].rename(out / "subs.vtt")
    for extra in sub_files[1:]:
        extra.unlink()
    series, ep = series_from_title(info.get("title", ""))
    _write_info(out, source="trt", uid=video_id, title=info.get("title"), series=series, episode=ep,
                url=info.get("webpage_url", url), duration=info.get("duration"), channel=info.get("channel"),
                subtitle_lang=lang)
    return out


# --------------------------------------------------------------------------- Kanal D / duhnet CDN
def download_kanald(series_slug: str, series_name: str, episode: int, mp4_url: str, raw_dir: Path,
                    srt_url: str | None = None) -> Path | None:
    """Pull the AAC track out of the CDN mp4 without storing the video; fetch the sibling .srt."""
    uid = f"{series_slug}_ep{episode:03d}"
    out = raw_dir / "kanald" / uid
    if is_complete(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    srt_url = srt_url or re.sub(r"\.mp4$", ".srt", mp4_url)

    r = requests.get(srt_url, timeout=60)
    if r.status_code != 200 or b"-->" not in r.content:
        print(f"[skip] {uid}: no subtitle at {srt_url} (HTTP {r.status_code})")
        shutil.rmtree(out, ignore_errors=True)
        return None
    (out / "subs.srt").write_bytes(r.content)

    audio = out / "audio.m4a"
    tmp = out / "audio.part.m4a"
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_on_network_error", "1", "-reconnect_delay_max", "30",
           "-i", mp4_url, "-vn", "-c:a", "copy", str(tmp)]
    for attempt in range(3):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1_000_000:
            break
        print(f"[retry {attempt + 1}] {uid}: ffmpeg failed: {res.stderr.strip()[:200]}")
        time.sleep(5)
    else:
        print(f"[fail] {uid}: could not fetch audio")
        return None
    tmp.rename(audio)
    dur = probe_duration(audio)
    _write_info(out, source="kanald", uid=uid, title=f"{series_name} {episode}. Bölüm", series=series_name,
                episode=episode, url=mp4_url, duration=dur)
    return out


def probe_duration(path: Path) -> float | None:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return None


def iter_raw(raw_dir: Path):
    for src in sorted(p for p in raw_dir.iterdir() if p.is_dir()) if raw_dir.exists() else []:
        for d in sorted(p for p in src.iterdir() if p.is_dir()):
            if is_complete(d):
                yield d
