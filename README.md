# ProTools Dub MVP

Automated multi-track audio post-production web app for Pro Tools projects.

## Features

- Upload Pro Tools .ptx files or ZIP files containing 2-5 WAV stems
- Automatic noise reduction using Librosa
- Loudness normalization to -24 LUFS using PyLoudnorm
- Custom processing via JSON configuration
- Mix processed stems into a single WAV output
- Simple web interface for upload and download

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

### JSON Configuration Format

Optional JSON config to customize processing per stem:

```json
{
  "vocal": {
    "compressor_threshold": -20,
    "gain_db": 2
  },
  "sfx": {
    "gain_db": 3
  }
}
```

### ZIP File Structure

Your ZIP should contain 2-5 WAV files:
- voice.wav
- sfx.wav
- music.wav
- etc.

## Deployment

Deployed on Heroku. See Procfile and runtime.txt for deployment configuration.

## Technology Stack

- Flask - Web framework
- aaf2 - Pro Tools .ptx file parsing
- PyDub - Audio manipulation
- Librosa - Audio analysis and noise reduction
- PyLoudnorm - LUFS normalization
- SoundFile - Audio I/O

## License

MIT
