"""Phase 1 self-check: parse + chunk DRA.pdf, assert non-empty, print stats. Runnable as a script."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.ingest import parse_pdf  # noqa: E402

DRA = pathlib.Path("/home/bombom/DRA.pdf")


def main():
    assert DRA.exists(), f"DRA.pdf missing at {DRA}"
    chunks = parse_pdf(DRA)
    assert len(chunks) > 0, "no chunks produced"
    sizes = [len(c.text) for c in chunks]
    print(f"chunks: {len(chunks)}")
    print(f"char size min/mean/max: {min(sizes)} / {sum(sizes)//len(sizes)} / {max(sizes)}")
    print("--- first chunk ---")
    print(chunks[0].text[:400])
    print("--- headings on first chunk ---")
    print(getattr(chunks[0].meta, "headings", None))


if __name__ == "__main__":
    main()
