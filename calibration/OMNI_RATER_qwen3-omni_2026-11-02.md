# Qwen3-Omni against emotion2vec+ on the hand-rated clips

Model `Qwen/Qwen3-Omni-30B-A3B-Instruct`, 268 clips with a rating and both scores. Rated: angry 57, happy 26, sad 40, neutral 137.

Top class over angry / happy / sad / neutral matches the rating: Qwen3-Omni 175/268, emotion2vec+ 172/268.

Precision of the k clips each model scores highest for a class (correct / k):

| class | k | Qwen3-Omni | emotion2vec+ |
|---|---|---|---|
| angry | 10 | 8/10 | 10/10 |
| angry | 20 | 16/20 | 20/20 |
| angry | 30 | 25/30 | 30/30 |
| angry | 40 | 32/40 | 38/40 |
| sad | 10 | 7/10 | 9/10 |
| sad | 20 | 13/20 | 16/20 |
| sad | 30 | 20/30 | 23/30 |
| sad | 40 | 23/40 | 27/40 |
| happy | 10 | 6/10 | 8/10 |
| happy | 20 | 8/20 | 14/20 |
| happy | 30 | 11/30 | 16/30 |
| happy | 40 | 12/40 | 20/40 |

Probability mass on the five class words: median 0.997, min 0.376.
