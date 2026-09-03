#!/usr/bin/env python3
"""
Convert markdown documents to PDF format for RAG indexing.

This script uses markdown2 + pdfkit to preserve tables, formatting,
and structure from the source markdown files.

Each markdown file is converted to a separate PDF without overwriting
any existing documents.

Requirements:
  pip install markdown2 pdfkit
  wkhtmltopdf: https://wkhtmltopdf.org/
    - macOS: brew install --cask wkhtmltopdf
    - Ubuntu: sudo apt-get install wkhtmltopdf
    - Windows: download from https://wkhtmltopdf.org/
"""

import sys
from pathlib import Path

try:
    import markdown2
    import pdfkit
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install markdown2 pdfkit")
    sys.exit(1)


def convert_markdown_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """
    Convert a markdown file to PDF.
    
    Args:
        md_path: Path to input markdown file
        pdf_path: Path to output PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Reading: {md_path.name}...", end=" ", flush=True)
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        print(f"Converting to HTML...", end=" ", flush=True)
        html_content = markdown2.markdown(
            md_content,
            extras=[
                "tables",
                "fenced-code-blocks",
                "toc",
                "strike",
            ]
        )
        
        # Wrap in basic HTML structure for PDF generation
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.4; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                table, th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                h1, h2, h3 {{ color: #333; margin-top: 20px; }}
                code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }}
                blockquote {{ border-left: 4px solid #ddd; padding-left: 10px; color: #666; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        print(f"Writing PDF...", end=" ", flush=True)
        pdfkit.from_string(full_html, str(pdf_path))
        
        file_size = pdf_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ ({file_size:.2f} MB)")
        return True
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


def main():
    """Convert all markdown documents in the documents directory to PDF."""
    
    docs_dir = Path("documents")
    
    if not docs_dir.exists():
        print(f"Error: {docs_dir} directory not found.")
        sys.exit(1)
    
    # Map markdown files to PDF output filenames (separate from base policy PDF)
    conversions = [
        (
            docs_dir / "CA_Personal_Auto_Declarations_50_Policies.md",
            docs_dir / "CA_Personal_Auto_Declarations_50_Policies_SYNTHETIC.pdf"
        ),
        (
            docs_dir / "CA_Personal_Auto_Endorsements_50_Policies.md",
            docs_dir / "CA_Personal_Auto_Endorsements_50_Policies_SYNTHETIC.pdf"
        ),
        (
            docs_dir / "CA_Personal_Auto_Internal_Claims_FAQ.md",
            docs_dir / "CA_Personal_Auto_Internal_Claims_FAQ_SYNTHETIC.pdf"
        ),
    ]
    
    print("\n=== Markdown to PDF Conversion ===\n")
    print("Note: Existing base policy PDF is preserved.\n")
    
    success_count = 0
    for md_path, pdf_path in conversions:
        if not md_path.exists():
            print(f"Skipping: {md_path.name} (not found)")
            continue
        
        print(f"Converting: {md_path.name}")
        if convert_markdown_to_pdf(md_path, pdf_path):
            success_count += 1
            print(f"  Output: {pdf_path.name}\n")
    
    print(f"=== Conversion Complete ===")
    print(f"Successfully converted: {success_count}/{len(conversions)} files\n")
    
    if success_count == len(conversions):
        print("✓ All synthetic documents converted to PDF and ready for build_index.py")
        print(f"  Location: {docs_dir.resolve()}\n")
        return 0
    else:
        print("⚠ Some conversions failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
