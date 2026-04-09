import argparse
import sys
import os
import logging
import tempfile
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer

# For PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODEL_ID = "zai-org/GLM-OCR"

# Temp directory for saving page images from PDFs
TEMP_DIR = tempfile.mkdtemp(prefix="glm_ocr_")


def log_gpu_usage(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.info(f"[GPU {label}] Memory: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.1f} GB total")


def load_images_from_pdf(pdf_path):
    """Convert each page of a PDF into a temp PNG file."""
    if not PDF_SUPPORT:
        logging.error("PyMuPDF is required for PDF support. Install it with: pip install pymupdf")
        sys.exit(1)

    from PIL import Image
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page at 300 DPI for high quality OCR
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Save to temp file so the processor can load it via URL/path
        temp_path = os.path.join(TEMP_DIR, f"page_{page_num + 1}.png")
        img.save(temp_path)
        pages.append((page_num + 1, temp_path))
    doc.close()
    logging.info(f"PDF rendered to {len(pages)} temp image(s) in {TEMP_DIR}")
    return pages


def run_ocr_on_path(model, processor, image_path, prompt="Text Recognition:", stream=True):
    """Run GLM-OCR inference on an image file with optional token streaming."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
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
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    log_gpu_usage("before inference")

    # Set up streaming so tokens print to console as they are generated
    streamer = TextStreamer(processor, skip_prompt=True, skip_special_tokens=False) if stream else None

    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            streamer=streamer,
        )

    elapsed = time.time() - start_time
    num_generated = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = num_generated / elapsed if elapsed > 0 else 0
    logging.info(f"Generated {num_generated} tokens in {elapsed:.1f}s ({tokens_per_sec:.1f} tok/s)")

    log_gpu_usage("after inference")

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
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable live token streaming to console.")

    args = parser.parse_args()

    if not os.path.exists(args.document_path):
        logging.error(f"File not found: {args.document_path}")
        sys.exit(1)

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.info(f"✅ Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        logging.warning("=" * 60)
        logging.warning("⚠️  CUDA IS NOT AVAILABLE — RUNNING ON CPU!")
        logging.warning("   This will be EXTREMELY slow (10x-50x slower).")
        logging.warning("   Ensure your PyTorch install matches your CUDA driver.")
        logging.warning("   Try: pip install torch --index-url https://download.pytorch.org/whl/cu126")
        logging.warning("=" * 60)

    # --- Model Loading ---
    logging.info(f"Loading model: {MODEL_ID} ...")
    load_start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    load_elapsed = time.time() - load_start
    logging.info(f"✅ Model loaded in {load_elapsed:.1f}s")
    logging.info(f"   Model device: {model.device}")
    log_gpu_usage("after model load")

    # --- Process Document ---
    ext = os.path.splitext(args.document_path)[1].lower()
    all_results = []
    stream = not args.no_stream

    if ext == ".pdf":
        logging.info(f"Processing PDF: {args.document_path}")
        pages = load_images_from_pdf(args.document_path)
        total_pages = len(pages)
        logging.info(f"Found {total_pages} page(s) in PDF.")

        total_start = time.time()
        for page_num, page_path in pages:
            logging.info(f"━━━ Page {page_num}/{total_pages} ━━━")
            text = run_ocr_on_path(model, processor, page_path, args.prompt, stream=stream)
            all_results.append(f"## Page {page_num}\n\n{text}")
            logging.info(f"✅ Page {page_num}/{total_pages} complete")

        total_elapsed = time.time() - total_start
        logging.info(f"━━━ All {total_pages} pages processed in {total_elapsed:.1f}s ({total_elapsed/total_pages:.1f}s/page) ━━━")
    else:
        # Treat as a single image file — pass path directly
        logging.info(f"Processing image: {args.document_path}")
        image_path = os.path.abspath(args.document_path)
        text = run_ocr_on_path(model, processor, image_path, args.prompt, stream=stream)
        all_results.append(text)

    # --- Output ---
    final_output = "\n\n---\n\n".join(all_results)

    if not stream:
        print("\n========== OCR RESULT ==========\n")
        print(final_output)
        print("\n================================\n")

    # Save to file
    os.makedirs(args.output, exist_ok=True)
    doc_basename = os.path.splitext(os.path.basename(args.document_path))[0]
    output_path = os.path.join(args.output, f"{doc_basename}_ocr.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)
    logging.info(f"📄 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
