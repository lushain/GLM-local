import argparse
import sys
from glmocr import GlmOcr
import logging

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR locally using the Official SDK.")
    parser.add_argument("--document_path", type=str, default="testDocs/dixon51.pdf", help="Path to the image or PDF to parse.")
    parser.add_argument("--output", type=str, default="./output", help="Directory or path to save the output.")
    
    args = parser.parse_args()
    
    try:
        logging.info(f"Initializing GLM-OCR model (This might take a moment if weights need to be downloaded, or loaded to A100 VRAM)...")
        # By default this will try to use the self-hosted pipeline since we installed glmocr[selfhosted]
        # and it will automatically use CUDA if torch is installed correctly.
        with GlmOcr(mode="selfhosted") as ocr_parser:
            logging.info(f"Processing document: {args.document_path}")
            result = ocr_parser.parse(args.document_path)
            
            print("\n--- Parsed Markdown Result ---\n")
            print(result.markdown_result)
            print("\n------------------------------\n")
            
            # Save the layout and results 
            result.save(args.output)
            logging.info(f"Results successfully saved to {args.output}")
            
    except Exception as e:
        logging.error(f"Error during OCR extraction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
