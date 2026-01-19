import os
import sqlite3
from datetime import datetime
from io import BytesIO, StringIO
import csv
import base64

from flask import (Flask, g, redirect, render_template, request, session,
                   url_for, flash, make_response, jsonify)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge

import cv2
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO

# Example uploaded image path (for your own local testing only)
# SAMPLE_IMAGE_PATH = "/mnt/data/IMG20251114172047.jpg"

# --- USER-SPECIFIC: set tesseract executable path (Windows) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "parking.db")

TOTAL_SLOTS = 200
VISITOR_SLOTS = 30  # first 30 slots are visitor slots; remainder are regular

app = Flask(__name__)
app.secret_key = "change-this-secret"  # change in production

# Allow uploads up to 8 MB (increase if needed)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # bytes


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return (
        "<h1>Upload too large</h1>"
        "<p>The image you uploaded is too large. Try reducing quality "
        "or use the manual plate entry.</p>",
        413,
    )


####################
# YOLO model       #
####################
# Loads YOLOv8n model (will auto-download weights on first run)
YOLO_MODEL = YOLO("yolov8n.pt")

# COCO vehicle classes (standard ids)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def vehicle_present(img_bgr, conf_threshold=0.4):
    """
    Use YOLO to check if a vehicle is present in the frame.
    Returns True if at least one car/motorbike/bus/truck is detected.
    """
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return False

    results = YOLO_MODEL(img_rgb, verbose=False)
    if not results:
        return False

    r = results[0]
    if r.boxes is None:
        return False

    cls = r.boxes.cls.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    for c, cf in zip(cls, conf):
        if int(c) in VEHICLE_CLASS_IDS and cf >= conf_threshold:
            return True
    return False


####################
# Database helpers #
####################
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        must_init = not os.path.exists(DB_PATH)
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
        if must_init:
            init_db(db)
        else:
            ensure_history_table(db)
    return db


def init_db(db):
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password_hash TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE plots (
        id INTEGER PRIMARY KEY,
        plot_no INTEGER UNIQUE,
        type TEXT,
        status TEXT,
        plate TEXT,
        last_update TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE registrations (
        id INTEGER PRIMARY KEY,
        plate TEXT UNIQUE,
        owner_name TEXT,
        assigned_plot INTEGER
    )
    """)
    cur.execute("""
    CREATE TABLE history (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        action TEXT,
        plate TEXT,
        plot_no INTEGER,
        operator TEXT,
        notes TEXT
    )
    """)
    cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123")))
    for i in range(1, TOTAL_SLOTS + 1):
        ptype = "visitor" if i <= VISITOR_SLOTS else "regular"
        cur.execute("INSERT INTO plots (plot_no, type, status, plate, last_update) VALUES (?, ?, ?, ?, ?)",
                    (i, ptype, "EMPTY", None, None))
    db.commit()


def ensure_history_table(db):
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        action TEXT,
        plate TEXT,
        plot_no INTEGER,
        operator TEXT,
        notes TEXT
    )
    """)
    db.commit()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


####################
# Auth helpers     #
####################
def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*a, **kw)

    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username", "")
        pwd = request.form.get("password", "")
        db = get_db()
        cur = db.execute("SELECT id, password_hash FROM users WHERE username = ?", (uname,))
        row = cur.fetchone()
        if row and check_password_hash(row["password_hash"], pwd):
            session["user_id"] = row["id"]
            session["username"] = uname
            return redirect(url_for("main"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


####################
# Helper functions #
####################
def find_free_visitor_slot(db):
    cur = db.execute("SELECT plot_no FROM plots WHERE type='visitor' AND status='EMPTY' ORDER BY plot_no ASC")
    r = cur.fetchone()
    return r["plot_no"] if r else None


def find_registered_plot_for_plate(db, plate):
    cur = db.execute("SELECT assigned_plot FROM registrations WHERE plate = ?", (plate,))
    r = cur.fetchone()
    return r["assigned_plot"] if r else None


def mark_plot_in_use(db, plot_no, plate, operator=None, notes=None):
    ts = datetime.now().isoformat()
    db.execute("UPDATE plots SET status='IN USE', plate=?, last_update=? WHERE plot_no=?", (plate, ts, plot_no))
    db.execute("INSERT INTO history (timestamp, action, plate, plot_no, operator, notes) VALUES (?, ?, ?, ?, ?, ?)",
               (ts, "arrival", plate, plot_no, operator, notes))
    db.commit()


def mark_plot_empty(db, plot_no, operator=None, notes=None):
    ts = datetime.now().isoformat()
    cur = db.execute("SELECT plate FROM plots WHERE plot_no = ?", (plot_no,))
    row = cur.fetchone()
    plate = row["plate"] if row else None
    db.execute("UPDATE plots SET status='EMPTY', plate=NULL, last_update=? WHERE plot_no=?", (ts, plot_no))
    db.execute("INSERT INTO history (timestamp, action, plate, plot_no, operator, notes) VALUES (?, ?, ?, ?, ?, ?)",
               (ts, "departure", plate, plot_no, operator, notes))
    db.commit()


def get_plot_by_plate(db, plate):
    cur = db.execute("SELECT plot_no FROM plots WHERE plate = ? AND status='IN USE'", (plate,))
    row = cur.fetchone()
    return row["plot_no"] if row else None


####################
# Shared arrival/departure helpers (page flows) #
####################
def handle_arrival_plate(db, plate, operator=None):
    existing = get_plot_by_plate(db, plate)
    if existing:
        flash(f"Plate {plate} already IN USE in plot {existing}", "info")
        return redirect(url_for("main"))

    reg_plot = find_registered_plot_for_plate(db, plate)
    if reg_plot:
        cur = db.execute("SELECT status FROM plots WHERE plot_no=?", (reg_plot,))
        r = cur.fetchone()
        if r and r["status"] == "EMPTY":
            mark_plot_in_use(db, reg_plot, plate, operator=operator, notes="registered assigned")
            flash(f"Regular vehicle assigned to its registered plot {reg_plot}", "success")
            return redirect(url_for("main"))
        else:
            cur = db.execute("SELECT plot_no FROM plots WHERE type='regular' AND status='EMPTY' ORDER BY plot_no ASC")
            alt = cur.fetchone()
            if alt:
                mark_plot_in_use(db, alt["plot_no"], plate, operator=operator, notes="registered fallback")
                flash(f"Registered plot busy. Assigned alternate regular plot {alt['plot_no']}", "warning")
                return redirect(url_for("main"))
            else:
                flash("No regular slots available. Try visitor slot (if allowed).", "danger")
                return redirect(url_for("main"))
    else:
        v = find_free_visitor_slot(db)
        if v:
            mark_plot_in_use(db, v, plate, operator=operator, notes="visitor")
            flash(f"Visitor vehicle assigned visitor slot {v}", "success")
            return redirect(url_for("main"))
        else:
            flash("No visitor slots available", "danger")
            return redirect(url_for("main"))


def handle_departure_plate(db, plate, operator=None):
    plot_no = get_plot_by_plate(db, plate)
    if not plot_no:
        flash(f"No currently parked vehicle with plate {plate} found.", "warning")
        return redirect(url_for("main"))
    mark_plot_empty(db, plot_no, operator=operator, notes="automatic departure")
    flash(f"Departure processed. Plot {plot_no} is now EMPTY.", "success")
    return redirect(url_for("main"))


####################
# ANPR helpers     #
####################
def preprocess_plate_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 400:
        gray = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_plate_from_image(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return ""
    h, w = gray.shape
    edged = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidate_texts = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / (ch + 1e-6)
        area = cw * ch
        if area < 500 or cw < 50:
            continue
        if 2 < aspect < 8 and ch > 10:
            crop = img_bgr[y:y + ch, x:x + cw]
            proc = preprocess_plate_for_ocr(crop)
            pil = Image.fromarray(proc)
            text = pytesseract.image_to_string(pil, config="--psm 7")
            text = sanitize_plate_text(text)
            if text:
                candidate_texts.append(text)

    if not candidate_texts:
        proc = preprocess_plate_for_ocr(img_bgr)
        pil = Image.fromarray(proc)
        text = pytesseract.image_to_string(pil, config="--psm 6")
        text = sanitize_plate_text(text)
        if text:
            candidate_texts.append(text)

    if candidate_texts:
        candidate_texts.sort(key=lambda s: len(s), reverse=True)
        return candidate_texts[0]
    return ""


def sanitize_plate_text(raw):
    if not raw:
        return ""
    s = raw.upper()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
    s = "".join(ch for ch in s if ch in allowed)
    s = s.replace(" ", "")
    s = s.replace("--", "-")
    s = s.strip("-")
    return s if len(s) >= 3 else ""


####################
# Routes - main    #
####################
@app.route("/")
@login_required
def main():
    db = get_db()
    cur = db.execute("SELECT * FROM plots ORDER BY plot_no ASC")
    plots = cur.fetchall()
    cur2 = db.execute("SELECT plate, assigned_plot, owner_name FROM registrations")
    regs = cur2.fetchall()
    return render_template("main.html", plots=plots, regs=regs, visitor_count=VISITOR_SLOTS, total=TOTAL_SLOTS)


@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    db = get_db()
    if request.method == "POST":
        plate = request.form.get("plate", "").upper().strip()
        owner = request.form.get("owner", "").strip()
        assigned_plot = request.form.get("assigned_plot", "")
        if not plate:
            flash("Plate number required", "warning")
            return redirect(url_for("register"))
        if assigned_plot:
            assigned_plot = int(assigned_plot)
            cur = db.execute("SELECT * FROM plots WHERE plot_no=? AND type='regular'", (assigned_plot,))
            if not cur.fetchone():
                flash("Invalid plot selected", "warning")
                return redirect(url_for("register"))
        try:
            db.execute("INSERT INTO registrations (plate, owner_name, assigned_plot) VALUES (?, ?, ?)",
                       (plate, owner or None, assigned_plot or None))
            db.commit()
            flash("Registered successfully", "success")
        except sqlite3.IntegrityError:
            flash("Plate already registered", "danger")
        return redirect(url_for("register"))
    cur = db.execute("SELECT plot_no FROM plots WHERE type='regular' AND status='EMPTY' ORDER BY plot_no ASC")
    regular_plots = [r["plot_no"] for r in cur.fetchall()]
    return render_template("register.html", regular_plots=regular_plots)


####################
# Arrival/Departure (pages – still support manual) #
####################
@app.route("/arrival", methods=["GET", "POST"])
@login_required
def arrival():
    if request.method == "POST":
        mode = request.form.get("mode", "camera")
        db = get_db()
        operator = session.get("username")
        if mode == "camera":
            data_url = request.form.get("image_data")
            if not data_url:
                flash("No image captured", "warning")
                return redirect(url_for("arrival"))
            header, encoded = data_url.split(",", 1)
            img_bytes = BytesIO()
            img_bytes.write(base64.b64decode(encoded))
            img_bytes.seek(0)
            file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # YOLO gate
            if not vehicle_present(img):
                flash("No vehicle detected. Try again or use manual entry.", "warning")
                return redirect(url_for("arrival"))

            plate = ocr_plate_from_image(img)
            if not plate:
                flash("OCR did not detect a plate. Please enter manually.", "warning")
                return render_template("arrival.html", detected_plate="", fallback=True)
            return handle_arrival_plate(db, plate, operator=operator)
        else:
            plate = request.form.get("manual_plate", "").upper().strip()
            if not plate:
                flash("Enter plate number", "warning")
                return redirect(url_for("arrival"))
            db = get_db()
            return handle_arrival_plate(db, plate, operator=operator)
    return render_template("arrival.html", detected_plate="", fallback=False)


@app.route("/departure", methods=["GET", "POST"])
@login_required
def departure():
    if request.method == "POST":
        mode = request.form.get("mode", "camera")
        db = get_db()
        operator = session.get("username")
        if mode == "camera":
            data_url = request.form.get("image_data")
            if not data_url:
                flash("No image captured", "warning")
                return redirect(url_for("departure"))
            header, encoded = data_url.split(",", 1)
            img_bytes = BytesIO()
            img_bytes.write(base64.b64decode(encoded))
            img_bytes.seek(0)
            file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if not vehicle_present(img):
                flash("No vehicle detected. Try again or use manual departure.", "warning")
                return redirect(url_for("departure"))

            plate = ocr_plate_from_image(img)
            if not plate:
                flash("OCR did not detect a plate. Please enter manually.", "warning")
                return render_template("departure.html", detected_plate="", fallback=True)
            return handle_departure_plate(db, plate, operator=operator)
        else:
            plate = request.form.get("manual_plate", "").upper().strip()
            if not plate:
                flash("Enter plate number", "warning")
                return redirect(url_for("departure"))
            db = get_db()
            return handle_departure_plate(db, plate, operator=operator)
    return render_template("departure.html", detected_plate="", fallback=False)


####################
# Arrival/Departure API endpoints (auto-capture loop) #
####################
@app.route("/arrival_api", methods=["POST"])
@login_required
def arrival_api():
    data_url = request.form.get("image_data")
    if not data_url:
        return jsonify({"status": "error", "message": "No image provided"}), 400
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = BytesIO(base64.b64decode(encoded))
        img_bytes.seek(0)
        file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"status": "error", "message": "Invalid image data", "detail": str(e)}), 400

    if not vehicle_present(img):
        return jsonify({"status": "no_vehicle", "message": "No vehicle detected"}), 200

    plate = ocr_plate_from_image(img)
    db = get_db()
    operator = session.get("username")
    if not plate:
        return jsonify({"status": "no_plate", "message": "Vehicle found but OCR failed"}), 200

    existing = get_plot_by_plate(db, plate)
    if existing:
        return jsonify({"status": "already_in", "message": f"Plate {plate} already in plot {existing}", "plate": plate, "plot_no": existing}), 200

    reg_plot = find_registered_plot_for_plate(db, plate)
    if reg_plot:
        cur = db.execute("SELECT status FROM plots WHERE plot_no=?", (reg_plot,))
        r = cur.fetchone()
        if r and r["status"] == "EMPTY":
            mark_plot_in_use(db, reg_plot, plate, operator=operator, notes="registered assigned (auto)")
            return jsonify({"status": "assigned", "message": f"Assigned registered plot {reg_plot}", "plate": plate, "plot_no": reg_plot}), 200
        else:
            cur = db.execute("SELECT plot_no FROM plots WHERE type='regular' AND status='EMPTY' ORDER BY plot_no ASC")
            alt = cur.fetchone()
            if alt:
                mark_plot_in_use(db, alt["plot_no"], plate, operator=operator, notes="registered fallback (auto)")
                return jsonify({"status": "assigned", "message": f"Assigned alternate regular plot {alt['plot_no']}", "plate": plate, "plot_no": alt['plot_no']}), 200
            else:
                return jsonify({"status": "full", "message": "No regular slots available"}), 200
    else:
        v = find_free_visitor_slot(db)
        if v:
            mark_plot_in_use(db, v, plate, operator=operator, notes="visitor (auto)")
            return jsonify({"status": "assigned", "message": f"Assigned visitor slot {v}", "plate": plate, "plot_no": v}), 200
        else:
            return jsonify({"status": "full", "message": "No visitor slots available"}), 200


@app.route("/departure_api", methods=["POST"])
@login_required
def departure_api():
    data_url = request.form.get("image_data")
    if not data_url:
        return jsonify({"status": "error", "message": "No image provided"}), 400
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = BytesIO(base64.b64decode(encoded))
        img_bytes.seek(0)
        file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"status": "error", "message": "Invalid image data", "detail": str(e)}), 400

    if not vehicle_present(img):
        return jsonify({"status": "no_vehicle", "message": "No vehicle detected"}), 200

    plate = ocr_plate_from_image(img)
    db = get_db()
    operator = session.get("username")
    if not plate:
        return jsonify({"status": "no_plate", "message": "Vehicle found but OCR failed"}), 200

    plot_no = get_plot_by_plate(db, plate)
    if not plot_no:
        return jsonify({"status": "not_found", "message": f"No parked vehicle with plate {plate} found"}), 200

    mark_plot_empty(db, plot_no, operator=operator, notes="automatic departure (auto)")
    return jsonify({"status": "departed", "message": f"Plot {plot_no} freed", "plate": plate, "plot_no": plot_no}), 200


####################
# Manual page, history, export #
####################
@app.route("/manual", methods=["GET", "POST"])
@login_required
def manual():
    db = get_db()
    operator = session.get("username")
    if request.method == "POST":
        action = request.form.get("action")
        if action == "enter_plate":
            plate = request.form.get("plate", "").upper().strip()
            chosen = request.form.get("chosen_plot")
            if not plate:
                flash("Plate required", "warning")
                return redirect(url_for("manual"))
            if not chosen:
                flash("Choose a plot", "warning")
                return redirect(url_for("manual"))
            chosen = int(chosen)
            cur = db.execute("SELECT status FROM plots WHERE plot_no=?", (chosen,))
            r = cur.fetchone()
            if r and r["status"] == "EMPTY":
                mark_plot_in_use(db, chosen, plate, operator=operator, notes="manual assign")
                flash(f"Manual arrival: Plate {plate} assigned to plot {chosen}", "success")
            else:
                flash("Selected plot not empty", "danger")
            return redirect(url_for("manual"))
        elif action == "free_plot":
            chosen = request.form.get("chosen_plot_free")
            if not chosen:
                flash("Choose a plot to free", "warning")
                return redirect(url_for("manual"))
            chosen = int(chosen)
            mark_plot_empty(db, chosen, operator=operator, notes="manual free")
            flash(f"Plot {chosen} set to EMPTY", "success")
            return redirect(url_for("manual"))
    cur = db.execute("SELECT * FROM plots ORDER BY plot_no")
    plots = cur.fetchall()
    return render_template("manual.html", plots=plots)


@app.route("/history")
@login_required
def history():
    db = get_db()
    # Only the query is limited to 500 rows; DB keeps all
    cur = db.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 500")
    rows = cur.fetchall()
    return render_template("history.html", rows=rows)


@app.route("/export")
@login_required
def export_csv():
    db = get_db()
    cur = db.execute("SELECT timestamp, action, plate, plot_no, operator, notes FROM history ORDER BY timestamp ASC")
    rows = cur.fetchall()

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["timestamp", "action", "plate", "plot_no", "operator", "notes"])
    for r in rows:
        cw.writerow([r["timestamp"], r["action"], r["plate"], r["plot_no"], r["operator"], r["notes"]])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=parking_history.csv"
    output.headers["Content-type"] = "text/csv"
    return output


####################
# Boot             #
####################
if __name__ == "__main__":
    with app.app_context():
        _ = get_db()
    app.run(debug=True)
