"""
Generate a small sample PDF ("handbook.pdf") used by the PDF demos.

Run on its own, or just run the demo scripts — they call this automatically if
the PDF is missing.

    python examples/pdf-rag/make_sample_pdf.py
"""

import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Save the PDF in the project root's examples/ folder so both demos can find it.
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "handbook.pdf")
PDF_PATH = os.path.abspath(PDF_PATH)

# (heading, body) pairs. The heading structure is what the vectorless demo uses.
CONTENT = [
    ("LSPL Company Handbook", ""),
    ("1. Billing", ""),
    ("1.1 Refunds", "Annual plans are refundable within 30 days of purchase. "
                    "Monthly plans are not refundable but can be cancelled anytime."),
    ("1.2 Invoices", "Invoices are emailed on the first day of each month. "
                     "You can download past invoices from the billing dashboard."),
    ("2. Accounts", "This is fake type of account"),
    ("2.1 Passwords", "Reset your password from the account settings page. "
                      "Passwords must be at least 12 characters long."),
    ("2.2 Login Issues", "If you cannot log in, use the forgot-password link. "
                         "Accounts lock for 15 minutes after five failed attempts."),
    ("3. Support", "For Support contact us on pjp.comp@gmail.com or call us on 9090909090"),
    ("3.1 Contact", "Support is available 24/7 through chat and email. "
                    "Enterprise customers also get a dedicated account manager."),
]


def build_pdf(path=PDF_PATH):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=LETTER)
    flow = []
    for heading, body in CONTENT:
        style = "Title" if body == "" and heading == "Company Handbook" else "Heading2"
        flow.append(Paragraph(heading, styles[style]))
        if body:
            flow.append(Paragraph(body, styles["BodyText"]))
        flow.append(Spacer(1, 10))
    doc.build(flow)
    return path


if __name__ == "__main__":
    out = build_pdf()
    print(f"Created sample PDF at: {out}")
