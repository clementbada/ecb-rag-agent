"""Cost meter: the numbers behind 'cost per answer' (USD per 1M tokens: 0.15 in, 0.60 out, 0.10 embed)."""
from ecbrag.llm import Meter, usd_chat, usd_embed


def test_usd_chat_by_hand():
    assert abs(usd_chat(1_000_000, 1_000_000) - 0.75) < 1e-12
    assert abs(usd_chat(2_000, 300) - 0.00048) < 1e-12   # (2000*0.15 + 300*0.60) / 1e6


def test_meter_adds_chat_and_embeddings():
    m = Meter(chat_calls=3, chat_in=6_000, chat_out=900, embed_calls=1, embed_in=10_000)
    expected = usd_chat(6_000, 900) + usd_embed(10_000)          # 0.00144 + 0.001
    assert abs(m.usd() - 0.00244) < 1e-12 and abs(m.usd() - expected) < 1e-12
    assert m.snapshot()["usd"] == round(expected, 6)
