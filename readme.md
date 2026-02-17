# Anime Upload Hub (RareToons-style Demo)

This repository now includes a lightweight Flask web app where users can upload anime video files and browse uploaded entries in a card-based layout.

## Features
- Upload anime videos with title and description
- Basic file type validation (`mp4`, `mkv`, `avi`, `mov`, `webm`)
- Homepage listing with embedded HTML5 video player
- JSON-based metadata persistence (`anime_posts.json`)

## Project Files
- `app.py` - Flask server and routes
- `templates/index.html` - Home listing page
- `templates/upload.html` - Upload form page
- `static/styles.css` - Styling
- `anime_posts.json` - Saved upload metadata
- `uploads/` - Stored video files

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser.

## Notes
- This is a demo app and uses a local JSON file instead of a database.
- For production, use a proper DB, authentication, and object storage.
