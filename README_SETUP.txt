1) Create a python venv:
   python -m venv venv
   Windows: venv\Scripts\activate
   macOS/Linux: source venv/bin/activate

2) Install requirements:
   pip install -r requirements.txt

3) Install Tesseract OCR as explained in tesseract_instructions.txt.

4) Open app.py and set TESSERACT_CMD if necessary (Windows).

5) Run:
   python app.py

6) Open browser at http://127.0.0.1:5000

Default security login:
   username: admin
   password: admin123

Note: The first run will create an SQLite DB with parking plots:
 - By default: 200 total plots (first 30 are visitor slots, others regular) as in your sketch.
You can change counts in app.py (variables TOTAL_SLOTS, VISITOR_SLOTS).
