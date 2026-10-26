from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    root: Path = Path("data")
    sample_rate: int = 24000

    # segment cutting: a generous window around each cue; exact edges come from word alignment later
    window_pad: float = 1.00
    trim_margin_before: float = 0.12
    trim_margin_after: float = 0.20
    merge_gap: float = 0.60       # max silence between two cues joined by "..." continuation
    merge_gap_comma: float = 0.30 # max gap when previous cue ends with a comma / no punctuation
    max_merged_dur: float = 15.0

    # VAD (cheap no-speech detection on the window; not used for trimming)
    vad_threshold: float = 0.5

    # dataset filters
    min_dur: float = 1.0
    max_dur: float = 15.0
    min_speech_ratio: float = 0.40  # VAD speech seconds in the window / cue length; mostly-music cues skip ASR
    min_chars: int = 3
    min_cps: float = 2.5          # characters per second (text / audio duration); slow emotional lines sit ~3
    max_cps: float = 30.0
    max_cer: float = 0.20         # ASR transcript (aligned span) vs subtitle text
    min_word_prob: float = 0.0    # mean Whisper word probability over the span (0 disables)
    max_no_speech_prob: float = 0.6
    require_asr: bool = True

    # edge snapping at build time (VAD on the trimmed clip): Whisper dates the first word from the end of the
    # preceding silence, which leaves up to ~1 s of lead-in; cap non-speech at the edges to these values
    lead_silence_max: float = 0.15
    trail_silence_max: float = 0.25
    snap_margin: float = 0.10

    # output
    target_lufs: float = -20.0
    peak_ceiling: float = 0.99
    val_fraction: float = 0.02
    seed: int = 1234

    # ASR backend: "auto" picks mlx on Apple Silicon, faster-whisper (CTranslate2, CUDA/CPU) elsewhere
    whisper_backend: str = "auto"
    whisper_model: str = "auto"      # auto -> mlx-community/whisper-large-v3-turbo | deepdml/faster-whisper-large-v3-turbo-ct2
    whisper_device: str = "auto"     # faster-whisper only: cuda / cpu / auto
    whisper_compute: str = "auto"    # faster-whisper only: float16 / int8_float16 / int8 / auto
    # packed inference: several clips concatenated (with silence gaps) into one <=30 s Whisper window,
    # then words are handed back to their clip by time. 0 disables (one Whisper call per clip).
    pack_seconds: float = 28.0
    pack_gap: float = 1.0
    pack_slack: float = 0.5          # words up to this far outside every clip still go to the nearest clip (gap/2)

    # speaker ids (ttsdata speakers): one embedding per clip, agglomerative clustering (average linkage, cosine
    # distance) within each episode, then episode clusters are merged across the series by centroid distance.
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_threshold: float = 0.70        # avg cosine distance between clips of one cluster, within an episode (hint purity knee)
    speaker_merge_threshold: float = 0.55  # cosine distance between episode-cluster centroids across the series (conservative)
    speaker_device: str = "auto"           # cuda / mps / cpu
    speaker_batch: int = 64                # clips per forward pass (sorted by length, zero-padded)

    # merged utterances (ttsdata merge-clips): consecutive same-voice subtitle lines joined into one longer clip
    merge_max_gap: float = 0.7             # seconds of pause allowed between two joined lines
    merge_min_cos: float = 0.4             # ECAPA cosine to the previous line and to the chain centroid (same voice)
    merge_cross_cluster_cos: float = 0.5   # ... when the two lines sit in different speaker clusters
    merge_after_question_cos: float = 0.55 # ... when the previous line ends with "?" (often answered by someone else)
    merge_max_dur: float = 15.0            # longest merged clip
    merge_max_cer: float = 0.10            # Whisper re-check of the merged clip against the joined subtitle text

    # subtitle sound descriptions (ttsdata soundtags)
    soundtag_adjacent: float = 1.5         # a vocal event this close before / after a clip is reported as adjacent

    # event-extended clips (ttsdata extend-events): the clip re-cut to include an adjacent laugh / cry / sigh
    event_pad: float = 0.2                 # seconds of margin around the event cue
    event_max_dur: float = 15.0            # longest extended clip
    event_max_bleed: int = 1               # extra Whisper tokens tolerated on the extended side (a laugh is not words)
    # verification of extended clips (ttsdata verify-events): Whisper + PANNs AudioSet tagger on the added audio
    event_speech_prob: float = 0.3         # words heard AND speech probability >= this: a narrator / another speaker
    event_tag_prob: float = 0.2            # the tag's own AudioSet class must reach this somewhere in the extension
    event_accept_clean: bool = True        # a clean extension (no words, no speech-like audio) counts as verified

    # coarse emotion labels (ttsdata emotions): emotion2vec+ (FunASR) per clip, calibrated against native ratings
    emotion_model: str = "iic/emotion2vec_plus_large"
    emotion_min_conf: float = 0.6          # top-class probability a clip needs to get a coarse label
    emotion_min_margin: float = 0.2        # ... and its lead over the runner-up coarse class
    emotion_device: str = "auto"           # cuda / cpu (FunASR has no MPS path)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def seg_dir(self) -> Path:
        return self.root / "segments"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"
