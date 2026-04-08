import argparse
import sys
import os
import logging
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# For PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logging.basicConfig(level=logging.INFO)

MODEL_ID = "zai-org/GLM-OCR"


def load_images_from_pdf(pdf_path):
    """Convert each page of a PDF into a PIL Image."""
    if not PDF_SUPPORT:
        logging.error("PyMuPDF is required for PDF support. Install it with: pip install pymupdf")
        sys.exit(1)

    from PIL import Image
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page at 300 DPI for high quality OCR
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_num + 1, img))
    doc.close()
    return images


def run_ocr_on_image(model, processor, image, prompt="Text Recognition:"):
    """Run GLM-OCR inference on a single PIL Image using the official chat template format."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        images=[image],
    ).to(model.device)

    # Some models include token_type_ids that aren't needed
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    # Decode only the generated tokens (skip the input prompt tokens)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR locally using HuggingFace Transformers.")
    parser.add_argument("--document_path", type=str, default="testDocs/dixon51.pdf",
                        help="Path to the image or PDF to parse.")
    parser.add_argument("--output", type=str, default="./output",
                        help="Directory to save the output markdown.")
    parser.add_argument("--prompt", type=str, default="Text Recognition:",
                        help="Prompt to use for the OCR model.")

    args = parser.parse_args()

    if not os.path.exists(args.document_path):
        logging.error(f"File not found: {args.document_path}")
        sys.exit(1)

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.info(f"Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        logging.warning("CUDA not available! Running on CPU (this will be very slow).")

    # --- Model Loading ---
    logging.info(f"Loading model: {MODEL_ID} (first run will download ~2GB of weights)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    logging.info("Model loaded successfully!")

    # --- Process Document ---
    ext = os.path.splitext(args.document_path)[1].lower()
    all_results = []

    if ext == ".pdf":
        logging.info(f"Processing PDF: {args.document_path}")
        pages = load_images_from_pdf(args.document_path)
        logging.info(f"Found {len(pages)} page(s) in PDF.")
        for page_num, page_img in pages:
            logging.info(f"  Running OCR on page {page_num}/{len(pages)}...")
            text = run_ocr_on_image(model, processor, page_img, args.prompt)
            all_results.append(f"## Page {page_num}\n\n{text}")
    else:
        # Treat as a single image
        from PIL import Image
        logging.info(f"Processing image: {args.document_path}")
        image = Image.open(args.document_path).convert("RGB")
        text = run_ocr_on_image(model, processor, image, args.prompt)
        all_results.append(text)

    # --- Output ---
    final_output = "\n\n---\n\n".join(all_results)

    print("\n========== OCR RESULT ==========\n")
    print(final_output)
    print("\n================================\n")

    # Save to file
    os.makedirs(args.output, exist_ok=True)
    doc_basename = os.path.splitext(os.path.basename(args.document_path))[0]
    output_path = os.path.join(args.output, f"{doc_basename}_ocr.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
