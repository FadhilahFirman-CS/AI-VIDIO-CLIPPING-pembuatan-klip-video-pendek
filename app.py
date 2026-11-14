import os
import re
import sys
import subprocess
import threading
import uuid
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "web_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
jobs = {}


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Ukuran video terlalu besar. Maksimal 2 GB."), 413


@app.errorhandler(Exception)
def handle_server_error(error):
    if request.path.startswith("/api/"):
        return jsonify(error=f"Server error: {error}"), 500
    if isinstance(error, HTTPException):
        return error
    raise error


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "video"


def run_job(job_id, source, options):
    job = jobs[job_id]
    try:
        output_base = OUTPUT_DIR / job_id / "hasil"
        output_base.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "cliper.py"), str(source), "-o", str(output_base),
               "--clips", str(options.get("clips", 4)),
               "--target", str(options.get("target", 25)), "--titles"]
        if options.get("vertical"):
            cmd.append("--vertical")
        if options.get("whisper"):
            cmd.append("--whisper")
        job["status"] = "processing"
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
            raise RuntimeError("Program tidak ditemukan: " + ", ".join(missing))
        try:
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Python executable tidak dapat dijalankan: {sys.executable}") from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-1800:] or result.stdout[-1800:])
        files = sorted(output_base.parent.glob("hasil_*.mp4"))
        job.update({"status": "done", "outputs": [f.name for f in files],
                    "log": result.stdout[-3000:]})
    except Exception as exc:
        job.update({"status": "error", "error": str(exc)})


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/clip")
def create_clip():
    source_type = request.form.get("source_type", "file")
    if source_type == "file":
        uploaded = request.files.get("video")
        if not uploaded or not uploaded.filename:
            return jsonify(error="Pilih file video terlebih dahulu."), 400
        filename = f"{uuid.uuid4().hex}_{safe_name(uploaded.filename)}"
        source = UPLOAD_DIR / filename
        uploaded.save(source)
    else:
        url = request.form.get("url", "").strip()
        if not re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", url):
            return jsonify(error="Masukkan URL YouTube yang valid."), 400
        source = UPLOAD_DIR / f"{uuid.uuid4().hex}_youtube.%(ext)s"
        download = ["yt-dlp", "--no-playlist", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                    "-o", str(source), url]
        yt_dlp = shutil.which("yt-dlp")
        if yt_dlp:
            download[0] = yt_dlp
        else:
            # Fallback untuk Windows: package pip sering terpasang tetapi
            # folder Scripts belum masuk PATH.
            download = [sys.executable, "-m", "yt_dlp", *download[1:]]
        result = subprocess.run(download, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify(error="Gagal mengunduh video YouTube. Pastikan yt-dlp terpasang.",
                           detail=result.stderr[-800:]), 422
        matches = list(UPLOAD_DIR.glob(source.name.replace("%(ext)s", "*")))
        if not matches:
            return jsonify(error="Video YouTube tidak ditemukan setelah diunduh."), 422
        source = matches[0]

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "outputs": [], "error": None}
    options = {"clips": int(request.form.get("clips", 4)),
               "target": float(request.form.get("target", 25)),
               "vertical": request.form.get("format", "landscape") == "vertical",
               "whisper": request.form.get("whisper") == "on"}
    threading.Thread(target=run_job, args=(job_id, source, options), daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job tidak ditemukan."), 404
    result = dict(job)
    result["urls"] = [f"/outputs/{job_id}/{name}" for name in job.get("outputs", [])]
    return jsonify(result)


@app.get("/outputs/<job_id>/<path:filename>")
def output_file(job_id, filename):
    return send_from_directory(OUTPUT_DIR / job_id, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
