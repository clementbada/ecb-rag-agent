from ecbrag.chunking import fixed_windows, sentence_windows


def test_fixed_windows_count_and_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    spans = fixed_windows(text, size=200, overlap=50)
    assert len(spans) == 7  # ceil((1000 - 50) / 150) as in the course deck
    first, second = text[slice(*spans[0])].split(), text[slice(*spans[1])].split()
    assert first[-50:] == second[:50], "consecutive chunks must share exactly the overlap"
    assert text[slice(*spans[-1])].split()[-1] == "w999", "the last word must be covered"


def test_sentence_windows_never_cut_a_sentence():
    sents = [f"Sentence number {i} talks about inflation and wages in the euro area." for i in range(60)]
    text = " ".join(sents)
    spans = sentence_windows(text, size=50, overlap=1)
    for a, b in spans:
        chunk = text[a:b]
        assert chunk.startswith("Sentence") and chunk.endswith(".")
        assert len(chunk.split()) <= 50 + 12
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    # one-sentence overlap between consecutive chunks
    assert spans[1][0] < spans[0][1]
