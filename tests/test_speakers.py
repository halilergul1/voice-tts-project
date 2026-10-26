import numpy as np

from ttsdata.speakers import (assign_speakers, centroids, cluster_embeddings, merge_clusters, normalize_hint,
                              purity_completeness, series_slug, speaker_ids)

RNG = np.random.default_rng(0)
DIM = 32


def voices(n: int) -> np.ndarray:
    """n well-separated unit directions (random vectors in 32-d are nearly orthogonal)."""
    v = RNG.normal(size=(n, DIM))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def clips(voice: np.ndarray, n: int, noise: float = 0.3) -> np.ndarray:
    """n noisy copies of a unit voice vector; `noise` is the expected norm of the perturbation."""
    x = voice[None, :] + noise / np.sqrt(DIM) * RNG.normal(size=(n, DIM))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_cluster_embeddings_recovers_speakers():
    v = voices(3)
    emb = np.concatenate([clips(v[0], 20), clips(v[1], 15), clips(v[2], 5)])
    truth = np.array([0] * 20 + [1] * 15 + [2] * 5)
    lab = cluster_embeddings(emb, threshold=0.5)
    assert lab.max() + 1 == 3
    for t in range(3):
        assert len(set(lab[truth == t])) == 1  # every true speaker lands in exactly one cluster
    assert len(set(lab)) == 3
    assert cluster_embeddings(emb, threshold=1e-6).max() + 1 == len(emb)  # tiny threshold: all apart
    assert cluster_embeddings(emb, threshold=2.0).max() == 0  # huge threshold: one cluster


def test_cluster_embeddings_edge_cases():
    assert cluster_embeddings(np.zeros((0, DIM)), 0.5).tolist() == []
    assert cluster_embeddings(np.ones((1, DIM)), 0.5).tolist() == [0]
    lab = cluster_embeddings(np.concatenate([clips(voices(1)[0], 3), clips(voices(1)[0], 3)]), 0.5)
    assert lab[0] == 0  # labels numbered by first appearance


def test_centroids_are_unit_and_sized():
    v = voices(2)
    emb = np.concatenate([clips(v[0], 4), clips(v[1], 6)])
    c, s = centroids(emb, np.array([0] * 4 + [1] * 6))
    assert s.tolist() == [4, 6]
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0)
    assert c[0] @ v[0] > 0.9 and c[1] @ v[1] > 0.9


def test_merge_clusters_links_same_voice_across_episodes():
    v = voices(3)
    cents = np.stack([v[0], v[1], v[0] + 0.05 * RNG.normal(size=DIM), v[2]])
    sizes = np.array([30, 10, 20, 5])
    groups = ["ep1", "ep1", "ep2", "ep2"]
    lab = merge_clusters(cents, sizes, groups, threshold=0.3)
    assert lab[0] == lab[2]  # voice 0 recurs in ep2
    assert len(set(lab.tolist())) == 3
    assert lab[0] != lab[1] and lab[1] != lab[3] and lab[0] != lab[3]


def test_merge_clusters_never_merges_within_an_episode():
    v = voices(1)[0]
    # two near-identical clusters from the same episode, plus one from another episode close to both
    cents = np.stack([v, v + 0.01 * RNG.normal(size=DIM), v + 0.02 * RNG.normal(size=DIM)])
    lab = merge_clusters(cents, np.array([5, 5, 5]), ["a", "a", "b"], threshold=0.9)
    assert lab[0] != lab[1]  # same-episode pair stays apart ...
    assert (lab[2] == lab[0]) != (lab[2] == lab[1])  # ... and the other episode's cluster joins exactly one of them
    lab = merge_clusters(cents, np.array([5, 5, 5]), ["a", "a", "a"], threshold=2.0)
    assert len(set(lab.tolist())) == 3


def test_merge_clusters_respects_threshold_and_empty():
    v = voices(2)
    lab = merge_clusters(np.stack([v[0], v[1]]), np.array([1, 1]), ["a", "b"], threshold=0.1)
    assert lab.tolist() == [0, 1]
    assert merge_clusters(np.zeros((0, DIM)), np.zeros(0, int), [], 0.5).tolist() == []


def test_speaker_ids_numbered_by_size():
    ids = speaker_ids("esrefruya", np.array([2, 2, 2, 0, 1, 1]))
    assert ids == ["esrefruya_spk00"] * 3 + ["esrefruya_spk02"] + ["esrefruya_spk01"] * 2
    wide = speaker_ids("x", np.arange(120))
    assert wide[0] == "x_spk000" and wide[119] == "x_spk119"


def test_series_slug():
    assert series_slug("Eşref Rüya") == "esrefruya"
    assert series_slug("Payitaht Abdülhamid") == "payitahtabdulhamid"
    assert series_slug("Adını Sen Koy") == "adinisenkoy"
    assert series_slug("İlyas & Co. 2") == "ilyasco2"


def test_normalize_hint():
    assert normalize_hint("Nisan ses") == "nisan"
    assert normalize_hint("Abdülhamid dış ses") == "abdülhamid"
    assert normalize_hint("Adile iç ses") == "adile"
    assert normalize_hint("Bidâr") == normalize_hint("Bidar") == "bidar"
    assert normalize_hint("Erkek dış ses") is None and normalize_hint("Kadın") is None
    assert normalize_hint("***") is None and normalize_hint(None) is None and normalize_hint("") is None


def test_purity_completeness():
    m = purity_completeness(["c0", "c0", "c0", "c1", "c1", "c2", "c2"], ["a", "a", "b", "b", None, "c", "c"])
    # hinted rows: c0:{a,a,b} c1:{b} c2:{c,c}; hints: a:{c0,c0} b:{c0,c1} c:{c2,c2}
    assert m["n"] == 6 and m["hints"] == 3 and m["clusters"] == 3
    assert abs(m["purity"] - 5 / 6) < 1e-9
    assert abs(m["completeness"] - 5 / 6) < 1e-9
    assert purity_completeness([0, 1], [None, None])["n"] == 0


def test_assign_speakers_end_to_end():
    v = voices(3)
    rows, vectors = [], {}

    def add(uid, series, voice, n, hint=None):
        for e in clips(v[voice], n):
            cid = f"{uid}_{len(rows):05d}"
            rows.append({"id": cid, "uid": uid, "series": series, "speaker_hint": hint})
            vectors[cid] = e.astype(np.float32)

    add("s_ep001", "Show", 0, 20, "Ali")
    add("s_ep001", "Show", 1, 10)
    add("s_ep002", "Show", 0, 15, "Ali ses")
    add("s_ep002", "Show", 2, 8)
    add("other", "Other Show", 1, 6)
    assign = assign_speakers(rows, vectors, threshold=0.5, merge_threshold=0.3)
    assert set(assign) == set(vectors)
    show_ids = {assign[r["id"]][0] for r in rows if r["series"] == "Show"}
    assert show_ids == {"show_spk00", "show_spk01", "show_spk02"}
    ali = {assign[r["id"]][0] for r in rows if r["speaker_hint"]}
    assert ali == {"show_spk00"}  # recurring character keeps one id across episodes, and it is the biggest
    assert assign[rows[0]["id"]][1] == 35
    assert {assign[r["id"]][0] for r in rows if r["series"] == "Other Show"} == {"othershow_spk00"}
