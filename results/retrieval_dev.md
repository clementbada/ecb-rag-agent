# Retrieval scorecard

Split: **dev** (20 questions with gold evidence). k = 5. Gold chunks = chunks covering at least 50% of a hand-verified evidence span, so variants with different chunk boundaries are scored on the same facts. Cost includes query embedding and rerank calls.

| variant              | mode          |   recall@5 |   precision@5 |   MRR |   nDCG@5 |   doc_hit@5 |   latency_p50_ms |   cost_per_query_usd |
|:---------------------|:--------------|-----------:|--------------:|------:|---------:|------------:|-----------------:|---------------------:|
| A_fixed200           | dense         |      0.412 |         0.110 | 0.243 |    0.272 |       0.550 |          242.644 |                0.000 |
| A_fixed200           | bm25          |      0.613 |         0.160 | 0.403 |    0.444 |       0.750 |           14.458 |                0.000 |
| A_fixed200           | hybrid        |      0.588 |         0.140 | 0.381 |    0.417 |       0.750 |           15.573 |                0.000 |
| A_fixed200           | hybrid_rerank |      0.613 |         0.150 | 0.617 |    0.580 |       0.800 |         1739.254 |                0.001 |
| B_fixed200_header    | dense         |      0.550 |         0.150 | 0.406 |    0.403 |       0.700 |            0.413 |                0.000 |
| B_fixed200_header    | bm25          |      0.812 |         0.230 | 0.713 |    0.701 |       0.900 |           14.536 |                0.000 |
| B_fixed200_header    | hybrid        |      0.787 |         0.220 | 0.658 |    0.655 |       0.850 |           16.364 |                0.000 |
| B_fixed200_header    | hybrid_rerank |      0.863 |         0.240 | 0.910 |    0.865 |       0.950 |         1701.128 |                0.001 |
| C_sentence250_header | dense         |      0.400 |         0.110 | 0.300 |    0.302 |       0.550 |            0.403 |                0.000 |
| C_sentence250_header | bm25          |      0.900 |         0.230 | 0.685 |    0.734 |       0.950 |           10.537 |                0.000 |
| C_sentence250_header | hybrid        |      0.750 |         0.200 | 0.575 |    0.602 |       0.900 |           11.179 |                0.000 |
| C_sentence250_header | hybrid_rerank |      0.850 |         0.220 | 0.735 |    0.734 |       0.900 |         1716.841 |                0.001 |
