# Agent evaluation

Split: **test** (held-out unless dev). Index: `B_fixed200_header`, retrieval: `hybrid_rerank`, model: `mistralai/mistral-small-2603`. Termination: {'final_answer': 21}. Cost includes agent calls, reranking and query embeddings. Faithfulness = share of answer claims supported by retrieved passages (LLM judge).

| system       |   n |   success_rate |   faithfulness |   evidence_recall |   mean_iterations |   mean_tool_calls |   hallucinated_call_rate |   tool_error_rate |   guard_blocked |   cost_per_answer_usd |   latency_p50_s |   latency_p95_s |
|:-------------|----:|---------------:|---------------:|------------------:|------------------:|------------------:|-------------------------:|------------------:|----------------:|----------------------:|----------------:|----------------:|
| agent        |  21 |          1.000 |          0.980 |             1.000 |             2.952 |             3.048 |                    0.000 |             0.031 |               1 |                 0.002 |           4.480 |          15.277 |
| rag_baseline |  21 |          0.857 |          0.969 |             0.944 |             1.000 |             0.000 |                    0.000 |             0.000 |               0 |                 0.001 |           4.562 |           5.734 |
