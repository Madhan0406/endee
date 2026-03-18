"""CLI ingestion — ingest PDFs from a data/ folder into Endee."""

import os
from pipeline import load_model, extract_text_from_pdf, ingest
from endee_client import EndeeClient


def main():
    client = EndeeClient()

    if not client.is_healthy():
        print("✗ Endee is not reachable. Start it and retry.")
        return

    client.ensure_index()
    model = load_model()

    data_dir = "data"
    if not os.path.isdir(data_dir):
        print(f"✗ Directory '{data_dir}' not found. Create it and add PDFs.")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"✗ No PDF files found in '{data_dir}'.")
        return

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        print(f"\n📄 Processing: {filename}")

        text = extract_text_from_pdf(path)
        print(f"   Extracted {len(text)} characters")

        count = ingest(text, filename, model, client)
        if count > 0:
            print(f"   ✓ Stored {count} chunks")
        else:
            print(f"   ✗ Ingestion failed")

    print("\n✓ All done.")


if __name__ == "__main__":
    main()