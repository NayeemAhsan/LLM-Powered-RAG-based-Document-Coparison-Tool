from faker import Faker
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
import random
import os
from datetime import datetime

fake = Faker()

def generate_insurance_pdf(file_path, insured_name, carrier_name, address, premium, deductible, accidents, policy_year):
    c = canvas.Canvas(file_path, pagesize=LETTER)
    width, height = LETTER

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Insurance Policy Document")

    # Insured Name in body
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"This policy is issued to: {insured_name}")

    # Box for Carrier Name
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.lightgrey)
    c.rect(300, height - 140, 250, 40, fill=1)
    c.setFillColor(colors.black)
    c.drawString(310, height - 120, f"Insurance Carrier: {carrier_name}")

    # Table for Premium and Deductible
    table_data = [
        ["Yearly Premium", "Deductible", "Number of Accidents", "Policy Year"],
        [f"${premium:,.2f}", f"${deductible:,.2f}", str(accidents), str(policy_year)]
    ]
    table = Table(table_data, colWidths=[120]*4)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, height - 220)

    # Footer Address
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 100, f"Insured Address: {address}")

    c.save()

def generate_sample_documents(output_dir="data/insurance_docs", count=10):
    os.makedirs(output_dir, exist_ok=True)
    current_year = datetime.now().year
    last_year = current_year - 1

    for i in range(count):
        insured_name = fake.name()
        carrier_name = fake.company()
        address = fake.address().replace("\n", ", ")
        premium = round(random.uniform(600, 2500), 2)
        deductible = round(random.uniform(200, 1000), 2)
        accidents = random.randint(0, 3)

        # Alternate between last year and current year
        policy_year = last_year if i < count // 2 else current_year

        file_path = os.path.join(output_dir, f"insurance_policy_{i+1}.pdf")
        generate_insurance_pdf(file_path, insured_name, carrier_name, address, premium, deductible, accidents, policy_year)

    print(f"{count} insurance documents generated in: {output_dir}")

# Run
generate_sample_documents()
