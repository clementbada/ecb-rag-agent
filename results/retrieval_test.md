# Retrieval scorecard

Split: **test** (18 questions with gold evidence). k = 5. Gold chunks = chunks covering at least 50% of a hand-verified evidence span, so variants with different chunk boundaries are scored on the same facts. Cost includes query embedding and rerank calls.

| variant              | mode          |   recall@5 |   precision@5 |   MRR |   nDCG@5 |   doc_hit@5 |   latency_p50_ms |   cost_per_query_usd |
|:---------------------|:--------------|-----------:|--------------:|------:|---------:|------------:|-----------------:|---------------------:|
| A_fixed200           | dense         |      0.500 |         0.111 | 0.374 |    0.400 |       0.722 |          244.009 |                0.000 |
| A_fixed200           | bm25          |      0.639 |         0.144 | 0.491 |    0.525 |       0.944 |           12.170 |                0.000 |
| A_fixed200           | hybrid        |      0.528 |         0.111 | 0.409 |    0.435 |       0.833 |           14.266 |                0.000 |
| A_fixed200           | hybrid_rerank |      0.694 |         0.156 | 0.694 |    0.652 |       0.944 |         1794.169 |                0.001 |
| B_fixed200_header    | dense         |      0.556 |         0.122 | 0.469 |    0.474 |       0.778 |            0.420 |                0.000 |
| B_fixed200_header    | bm25          |      0.778 |         0.167 | 0.676 |    0.684 |       0.889 |           12.777 |                0.000 |
| B_fixed200_header    | hybrid        |      0.750 |         0.156 | 0.565 |    0.596 |       0.833 |           14.388 |                0.000 |
| B_fixed200_header    | hybrid_rerank |      0.917 |         0.211 | 0.833 |    0.836 |       0.944 |         1804.669 |                0.001 |
| C_sentence250_header | dense         |      0.667 |         0.167 | 0.531 |    0.528 |       0.722 |            0.401 |                0.000 |
| C_sentence250_header | bm25          |      0.833 |         0.211 | 0.657 |    0.697 |       0.833 |            9.533 |                0.000 |
| C_sentence250_header | hybrid        |      0.722 |         0.189 | 0.667 |    0.670 |       0.778 |           10.230 |                0.000 |
| C_sentence250_header | hybrid_rerank |      0.889 |         0.233 | 0.678 |    0.709 |       0.944 |         1634.715 |                0.001 |
