def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    return obj
safe_structured = sanitize_for_json(structured)
safe_vehicle = sanitize_for_json(vehicle_info)


def create_pdf(report_text, annotated_images, structured, vehicle_info):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Add Unicode font
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.add_page()
    pdf.set_font("DejaVu", size=16)
    pdf.cell(0, 10, "Vehicle Damage Assessment Report", ln=True)

    pdf.set_font("DejaVu", size=11)
    pdf.ln(2)
    pdf.multi_cell(0, 6, report_text)

    pdf.ln(4)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(0, 8, "Detected Damages (Summary)", ln=True)

    pdf.set_font("DejaVu", size=10)

    if structured:
        for d in structured:
            pdf.cell(
                0,
                6,
                f"- {d['part_zone'].title()} | {d['damage_type']} | {d['severity']} | ₹{int(d['estimated_cost'])}",
                ln=True
            )
    else:
        pdf.cell(0, 6, "No damages detected.", ln=True)

    # Add annotated images
    for i, img in enumerate(annotated_images):
        pdf.add_page()
        pdf.set_font("DejaVu", size=12)
        pdf.cell(0, 8, f"Image {i+1}", ln=True)

        temp_path = f"temp_img_{i}.jpg"
        img_rgb = img.convert("RGB")
        img_rgb.save(temp_path, "JPEG")

        page_w = pdf.w - 2 * pdf.l_margin
        pil_w, pil_h = img_rgb.size
        ratio = page_w / pil_w
        new_h = pil_h * ratio

        pdf.image(temp_path, x=pdf.l_margin, y=pdf.get_y(), w=page_w)

        os.remove(temp_path)

    # Return PDF bytes directly
    return pdf.output(dest="S").encode("latin-1", "ignore")
