# app.py
import os
import io
import json
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from fpdf import FPDF
from dotenv import load_dotenv
from tqdm import tqdm

# optional YOLO
USE_ULTRALYTICS = True
try:
    if USE_ULTRALYTICS:
        from ultralytics import YOLO
except Exception as e:
    YOLO = None

# OpenAI
try:
    import openai
except Exception as e:
    openai = None

# load .env if present
load_dotenv()

st.set_page_config(page_title="Damage Report Generator — YOLO + LLM", layout="wide")

# ---------------------------
# Utilities
# ---------------------------

def load_yolo_model(model_path: str = "yolov8n.pt"):
    """
    Load a YOLO model. If model_path is 'damage.pt' it's assumed to be a fine-tuned damage detector.
    Otherwise, ultralytics will download the standard weight automatically.
    """
    if YOLO is None:
        st.warning("Ultralytics not installed. Falling back to heuristic detection.")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model at {model_path}: {e}")
        return None

def pil_to_cv2(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    return arr[:, :, ::-1].copy()  # RGB to BGR

def cv2_to_pil(img_cv2):
    return Image.fromarray(img_cv2[:, :, ::-1])

# Heuristic fallback (same as earlier)
def detect_damage_heuristic(pil_img, min_area_ratio=0.002):
    img = pil_to_cv2(pil_img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min_area_ratio * w * h):
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        bbox_area = cw * ch
        solidity = area / (bbox_area + 1e-9)
        score = min(0.99, (area / (w*h)) * 10 + solidity * 0.3)
        dets.append({"bbox": (int(x), int(y), int(cw), int(ch)), "area_px": int(area), "score": float(score)})
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    return dets

def estimate_part_from_bbox(bbox, image_size):
    x, y, w, h = bbox
    img_w, img_h = image_size
    cx = x + w/2
    cy = y + h/2
    left_third = img_w/3
    right_third = img_w*2/3
    if cy < img_h * 0.25:
        return "roof"
    if cx < left_third:
        return "left_side"
    if cx > right_third:
        return "right_side"
    if cy > img_h * 0.6:
        return "rear"
    return "front"

def guess_damage_type(bbox, image):
    x, y, w, h = bbox
    img_cv = pil_to_cv2(image)
    roi = img_cv[y:y+h, x:x+w]
    if roi.size == 0:
        return "unknown"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    aspect = w / (h + 1e-9)
    if var_lap > 500:
        return "crack/broken"
    if aspect > 3 or aspect < 0.3:
        return "scratch"
    return "dent"

def estimate_area_cm2(area_px, image_size, vehicle_width_cm=175):
    img_w, img_h = image_size
    px_per_cm = img_w / vehicle_width_cm
    area_cm2 = (area_px) / (px_per_cm * px_per_cm + 1e-9)
    return float(area_cm2)

# simpler cost config for demo
COST_BASE = {"front":15000, "rear":14000, "left_side":10000, "right_side":10000, "roof":12000, "unknown":8000}
DAMAGE_MULTIPLIER = {"scratch":0.2, "dent":0.5, "crack/broken":0.9, "unknown":0.4}

def severity_from_area(area_cm2):
    if area_cm2 < 50:
        return "minor", 0.2
    if area_cm2 < 200:
        return "moderate", 0.5
    return "major", 0.9

def estimate_cost(part_zone, damage_type, area_cm2):
    base = COST_BASE.get(part_zone, COST_BASE["unknown"])
    multiplier = DAMAGE_MULTIPLIER.get(damage_type, DAMAGE_MULTIPLIER["unknown"])
    _, sev_score = severity_from_area(area_cm2)
    area_factor = 1 + min(3.0, area_cm2 / 300.0)
    est = base * (0.5 + multiplier * sev_score * 2.0) * area_factor
    low = est * 0.85
    high = est * 1.15
    return float(low), float(high), float(est)

# ---------------------------
# LLM report generation (OpenAI ChatCompletion)
# ---------------------------

def llm_generate_report_openai(structured_json, vehicle_info, api_key,deployment_name,endpoint, api_version):
    """
    Send structured JSON to OpenAI to create a polished assessor-style report.
    Requires OPENAI_API_KEY environment variable or provided arg.
    """
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

    system_prompt = (
        "You are a professional motor vehicle assessor and insurance report writer. "
        "Convert the structured damage JSON into a clear, professional damage assessment report "
        "suitable for a vehicle insurance claim. Be concise but complete. Include:"
        " - Short summary (1 para),"
        " - Detailed findings per damaged item (part/zone, damage type, severity, estimated area),"
        " - Repair recommendations,"
        " - Estimated cost ranges and a brief justification of cost,"
        " - Any safety-critical notes and next steps."
        " Do not hallucinate extra damaged parts that are not in the JSON. If JSON is empty, say 'no significant damage detected.'"
    )

    user_payload = {
        "structured":structured_json,
        "vehicle": vehicle_info
    }

    # Build a compact chat messages payload
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": "Here is the structured JSON and vehicle info:\n\n" + json.dumps(user_payload, indent=2)}
    ]

    # call ChatCompletion
    try:
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.1,
            max_tokens=900
        )
        text = resp.choices[0].message.content
        return text
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        # fallback: return a simple template
        return None

# ---------------------------
# PDF generation
# ---------------------------
def generate_simple_report_text(structured, vehicle_info):
    """
    Fallback report generator used when LLM fails.
    Creates a simple human-readable summary from structured detections.
    """
    lines = []
    lines.append("Vehicle Damage Assessment Report")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    make = vehicle_info.get("make", "-")
    model = vehicle_info.get("model", "-")
    year = vehicle_info.get("year", "-")
    reg = vehicle_info.get("registration", "-")

    lines.append(f"Vehicle: {make} {model} ({year})")
    lines.append(f"Registration: {reg}")
    lines.append("")

    if not structured:
        lines.append("No significant damage detected by automated analysis.")
        return "\n".join(lines)

    # Summary
    lines.append("Summary of Findings:")
    total_low = sum(d["estimated_cost_low"] for d in structured)
    total_high = sum(d["estimated_cost_high"] for d in structured)
    lines.append(f"- Estimated total repair cost: ₹{int(total_low)} – ₹{int(total_high)}")
    lines.append("")

    # Detailed damage items
    lines.append("Detailed Findings:")
    for i, d in enumerate(structured, 1):
        lines.append(f"{i}. {d['part_zone'].title()}")
        lines.append(f"   • Damage type: {d['damage_type']}")
        lines.append(f"   • Severity: {d['severity']} (score {d['severity_score']})")
        lines.append(f"   • Area: {d['area_cm2']} cm²")
        lines.append(f"   • Estimated cost: ₹{int(d['estimated_cost_low'])} – ₹{int(d['estimated_cost_high'])}")
        lines.append("")

    lines.append("Recommendations:")
    lines.append("- Review high severity damage first.")
    lines.append("- Consult a certified repair shop for an accurate final estimate.")
    lines.append("- Consider safety-critical replacements (lights, bumpers, panels).")

    return "\n".join(lines)

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
    return bytes(pdf.output(dest="S"))

# ---------------------------
# Streamlit App layout
# ---------------------------

st.title("🚗 Damage Report Generator — YOLO + LLM (OpenAI)")

with st.sidebar:
    st.header("Settings")
    openai_key_env = os.environ.get("AZURE_OPENAI_API_KEY", "")
    key_input = st.text_input("OpenAI API Key (or set OPENAI_API_KEY in .env)", type="password", value=openai_key_env)
    azure_endpoint = st.text_input("Azure OpenAI Endpoint", value=os.environ.get("AZURE_OPENAI_ENDPOINT",""))
    azure_deployment_name = st.text_input("Azure Deployment Name", value=os.environ.get("AZURE_OPENAI_MODEL",""))
    azure_api_version = st.text_input("Azure Api Version", value=os.environ.get("AZURE_OPENAI_API_VERSION",""))

    st.markdown("YOLO model:")
    yolom = st.text_input("YOLO model path or name (e.g., 'yolov8n.pt' or 'damage_best.pt')", value="yolov8n.pt")
    st.checkbox("Show debug logs", key="debug")

st.markdown("""
Upload multiple images (front/rear/left/right). For best results, include clear, well-lit close-up photos of damage areas.
If you have a fine-tuned YOLO damage model (e.g., `damage_best.pt`), provide its path above.
""")

uploaded = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg","png","jpeg"])
col1, col2 = st.columns(2)

if uploaded:
    images = []
    for f in uploaded:
        try:
            im = Image.open(f).convert("RGB")
            images.append(im)
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")

    if not images:
        st.warning("No images loaded.")
    else:
        with col1:
            st.subheader("Vehicle details")
            make = st.text_input("Make (e.g., Hyundai)", value="")
            model = st.text_input("Model (e.g., i20)", value="")
            year = st.text_input("Year", value="")
            registration = st.text_input("Registration", value="")
            vehicle_width_cm = st.number_input("Vehicle width (cm) — for area estimate", value=175.0)
            run_button = st.button("Run detection & generate report")

        if run_button:
            # load model (if available)
            model = None
            if YOLO is not None:
                try:
                    st.info(f"Loading YOLO model: {yolom} (this may take a few seconds)")
                    model = load_yolo_model(yolom)
                except Exception as e:
                    st.warning(f"Could not load YOLO: {e}")
                    model = None

            st.info("Running detection on images...")
            structured = []
            annotated_imgs = []
            prog = st.progress(0)
            for idx, img in enumerate(images):
                # detection via ultralytics if model loaded, else heuristic
                dets = []
                if model is not None:
                    try:
                        # run model (batch inference could be used)
                        res = model(img)  # ultralytics inference on PIL also supported
                        # res can contain boxes and masks
                        # We'll extract boxes (xyxy), scores and class
                        boxes = []
                        for r in res:
                            preds = r.boxes
                            if preds is None:
                                continue
                            for b in preds:
                                xyxy = b.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
                                score = float(b.conf[0].cpu().numpy())
                                x1,y1,x2,y2 = map(int, xyxy.tolist())
                                w = x2 - x1
                                h = y2 - y1
                                area = w*h
                                dets.append({"bbox": (x1,y1,w,h), "area_px": int(area), "score": score})
                    except Exception as e:
                        st.warning(f"YOLO inference failed on image {idx+1}: {e}")
                        dets = detect_damage_heuristic(img)
                else:
                    dets = detect_damage_heuristic(img)

                # build structured items
                img_w, img_h = img.size
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)
                for d in dets:
                    bbox = d["bbox"]
                    part_zone = estimate_part_from_bbox(bbox, (img_w, img_h))
                    d_type = guess_damage_type(bbox, img)
                    area_cm2 = estimate_area_cm2(d["area_px"], (img_w, img_h), vehicle_width_cm)
                    severity, sev_score = severity_from_area(area_cm2)
                    low, high, est = estimate_cost(part_zone, d_type, area_cm2)
                    structured_item = {
                        "image_index": idx,
                        "bbox": bbox,
                        "area_px": d["area_px"],
                        "area_cm2": round(area_cm2, 1),
                        "score": round(d["score"], 3),
                        "part_zone": part_zone,
                        "damage_type": d_type,
                        "severity": severity,
                        "severity_score": round(sev_score, 3),
                        "estimated_cost_low": round(low, 2),
                        "estimated_cost_high": round(high, 2),
                        "estimated_cost": round(est, 2)
                    }
                    structured.append(structured_item)
                    x, y, w, h = bbox
                    draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                    label = f"{d_type} {severity} ₹{int(est)}"
                    draw.text((x+4, max(0,y-18)), label, fill="yellow")
                annotated_imgs.append(draw_img)
                prog.progress(int((idx+1)/len(images)*100))
            st.success("Detection complete.")

            # small sanity: dedupe overlapping boxes (optional) - simple IOU filter
            # For simplicity we keep all detections; production should apply NMS and thresholding.

            # Show annotated images
            st.subheader("Annotated images")
            cols = st.columns(min(3, len(annotated_imgs)))
            for i, img in enumerate(annotated_imgs):
                with cols[i % len(cols)]:
                    st.image(img, caption=f"Image {i+1}", use_column_width=True)

            # show structured JSON
            st.subheader("Structured damage JSON")
            st.json(structured)

            # Generate LLM-polished report
            vehicle_info = {"make": make, "model": model, "year": year, "registration": registration}
            st.info("Generating polished report with OpenAI LLM...")
            try:
                llm_text = llm_generate_report_openai(
                     structured_json=structured,
                     vehicle_info=vehicle_info,
                     api_key=key_input,
                     deployment_name=azure_deployment_name,
                     endpoint=azure_endpoint,
                     api_version=azure_api_version 
                     )
                if llm_text is None:
                    raise Exception("LLM returned None; falling back to template text.")
            except Exception as e:
                st.error(f"LLM generation error: {e}")
                # fallback template
                llm_text = generate_simple_report_text(structured, vehicle_info) if 'generate_simple_report_text' in globals() else "LLM failed — see structured JSON."

            st.subheader("Polished LLM Report")
            st.text_area("Final Report", value=llm_text, height=400)

            # Create PDF
            pdf_bytes = create_pdf(llm_text, annotated_imgs, structured, vehicle_info)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="damage_report_llm.pdf", mime="application/pdf")

            # Download JSON
            st.download_button("Download structured JSON", data=json.dumps(structured, indent=2), file_name="structured_damage.json", mime="application/json")

else:
    st.info("Please upload images to begin.")

st.markdown("---")
st.caption("Notes: For best accuracy in damage detection, fine-tune a YOLO/DETR model on labelled damage images. LLM step uses OpenAI; you can optionally use other LLMs by changing the llm function.")
