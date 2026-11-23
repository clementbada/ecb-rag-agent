# Agent evaluation

Split: **test** (held-out unless dev). Index: `B_fixed200_header`, retrieval: `hybrid_rerank`, model: `mistralai/mistral-small-2603`. Termination: {'final_answer': 21}. Cost includes agent calls, reranking and query embeddings. Faithfulness = share of answer claims supported by retrieved passages (LLM judge).

| system       |   n |   success_rate |   faithfulness |   evidence_recall |   mean_iterations |   mean_tool_calls |   hallucinated_call_rate |   tool_error_rate |   cost_per_answer_usd |   latency_p50_s |   latency_p95_s |
|:-------------|----:|---------------:|---------------:|------------------:|------------------:|------------------:|-------------------------:|------------------:|----------------------:|----------------:|----------------:|
| agent        |  21 |          1.000 |          0.941 |             1.000 |             2.476 |             2.524 |                    0.000 |             0.019 |                 0.002 |           3.893 |          10.894 |
| rag_baseline |  21 |          0.857 |          0.967 |             0.944 |             1.000 |             0.000 |                    0.000 |             0.000 |                 0.001 |           4.033 |           4.915 |
