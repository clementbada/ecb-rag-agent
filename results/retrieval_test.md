# Retrieval scorecard

Split: **test** (18 questions with gold evidence). k = 5. Gold chunks = chunks covering at least 50% of a hand-verified evidence span, so variants with different chunk boundaries are scored on the same facts. Cost includes query embedding and rerank calls. MRR_ci95: 95% bootstrap interval over questions; configs whose intervals overlap are not separable on this set.

| variant              | mode          |   recall@5 |   precision@5 |   MRR | MRR_ci95     |   nDCG@5 |   doc_hit@5 |   latency_p50_ms |   cost_per_query_usd |
|:---------------------|:--------------|-----------:|--------------:|------:|:-------------|---------:|------------:|-----------------:|---------------------:|
| A_fixed200           | dense         |      0.500 |         0.111 | 0.374 | [0.19, 0.57] |    0.400 |       0.722 |          278.676 |                0.000 |
| A_fixed200           | bm25          |      0.639 |         0.144 | 0.491 | [0.29, 0.68] |    0.525 |       1.000 |           19.143 |                0.000 |
| A_fixed200           | hybrid        |      0.528 |         0.111 | 0.409 | [0.21, 0.61] |    0.435 |       0.889 |           16.676 |                0.000 |
| A_fixed200           | hybrid_rerank |      0.750 |         0.178 | 0.650 | [0.45, 0.83] |    0.657 |       0.944 |         1768.231 |                0.001 |
| B_fixed200_header    | dense         |      0.556 |         0.122 | 0.469 | [0.28, 0.68] |    0.474 |       0.778 |            0.422 |                0.000 |
| B_fixed200_header    | bm25          |      0.778 |         0.167 | 0.676 | [0.49, 0.84] |    0.684 |       0.944 |           12.949 |                0.000 |
| B_fixed200_header    | hybrid        |      0.750 |         0.156 | 0.565 | [0.38, 0.75] |    0.596 |       0.833 |           14.227 |                0.000 |
| B_fixed200_header    | hybrid_rerank |      0.944 |         0.222 | 0.875 | [0.72, 1.00] |    0.896 |       1.000 |         1876.157 |                0.001 |
| C_sentence250_header | dense         |      0.667 |         0.167 | 0.531 | [0.33, 0.73] |    0.528 |       0.722 |            0.411 |                0.000 |
| C_sentence250_header | bm25          |      0.852 |         0.222 | 0.713 | [0.52, 0.88] |    0.723 |       0.889 |            9.663 |                0.000 |
| C_sentence250_header | hybrid        |      0.741 |         0.200 | 0.685 | [0.48, 0.86] |    0.683 |       0.833 |           10.610 |                0.000 |
| C_sentence250_header | hybrid_rerank |      0.907 |         0.244 | 0.903 | [0.76, 1.00] |    0.887 |       0.944 |         1903.176 |                0.001 |
