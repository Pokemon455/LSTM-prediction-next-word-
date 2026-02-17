from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_FILE = BASE_DIR / "anime_posts.json"
ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

UPLOAD_DIR.mkdir(exist_ok=True)
if not DATA_FILE.exists():
    DATA_FILE.write_text("[]", encoding="utf-8")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_posts() -> list[dict]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def save_posts(posts: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(posts, indent=2), encoding="utf-8")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "anime"


@app.route("/")
def index():
    posts = sorted(load_posts(), key=lambda x: x["created_at"], reverse=True)
    return render_template("index.html", posts=posts)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        file = request.files.get("video")

        if not title:
            flash("Title is required.", "error")
            return redirect(url_for("upload"))

        if not file or file.filename == "":
            flash("Please choose a video file.", "error")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            flash("Unsupported file type. Allowed: mp4, mkv, avi, mov, webm", "error")
            return redirect(url_for("upload"))

        cleaned_name = secure_filename(file.filename)
        file_id = f"{slugify(title)}-{uuid4().hex[:8]}"
        file_ext = cleaned_name.rsplit(".", 1)[1].lower()
        stored_name = f"{file_id}.{file_ext}"
        destination = UPLOAD_DIR / stored_name
        file.save(destination)

        posts = load_posts()
        posts.append(
            {
                "id": file_id,
                "title": title,
                "description": description,
                "filename": stored_name,
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        save_posts(posts)

        flash("Anime uploaded successfully!", "success")
        return redirect(url_for("index"))

    return render_template("upload.html", allowed_extensions=", ".join(sorted(ALLOWED_EXTENSIONS)))


@app.route("/video/<path:filename>")
def video(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
