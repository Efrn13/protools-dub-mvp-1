import os
import json
import zipfile
import logging
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from pydub import AudioSegment
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed_audio'
app.config['ALLOWED_EXTENSIONS'] = {'ptx', 'zip', 'wav', 'json'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_noise_reduction(audio_data, sr):
    logger.info("Applying noise reduction using spectral gating...")
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    noise_threshold = np.percentile(magnitude, 10)
    mask = magnitude > noise_threshold
    magnitude_cleaned = magnitude * mask
    
    stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
    audio_cleaned = librosa.istft(stft_cleaned)
    
    return audio_cleaned

def normalize_to_lufs(audio_path, target_lufs=-24.0):
    logger.info(f"Normalizing audio to {target_lufs} LUFS...")
    data, rate = sf.read(audio_path)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    
    normalized_data = pyln.normalize.loudness(data, loudness, target_lufs)
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_output.name, normalized_data, rate)
    
    return temp_output.name

def process_stem(stem_path, stem_name, config=None):
    logger.info(f"Processing stem: {stem_name}")
    
    data, sr = librosa.load(stem_path, sr=None, mono=False)
    
    if len(data.shape) == 1:
        logger.info("Converting mono to stereo...")
        data = np.stack([data, data])
    
    logger.info("Applying noise reduction to each channel...")
    processed_channels = []
    for channel in data:
        processed_channel = apply_noise_reduction(channel, sr)
        processed_channels.append(processed_channel)
    
    processed_data = np.array(processed_channels)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, processed_data.T, sr)
    
    normalized_path = normalize_to_lufs(temp_file.name)
    os.unlink(temp_file.name)
    
    if config and stem_name in config:
        stem_config = config[stem_name]
        if 'gain_db' in stem_config:
            logger.info(f"Applying gain: {stem_config['gain_db']} dB")
            audio = AudioSegment.from_wav(normalized_path)
            audio = audio + stem_config['gain_db']
            audio.export(normalized_path, format='wav')
    
    return normalized_path

def mix_stems(stem_paths):
    logger.info("Mixing processed stems...")
    
    combined = None
    for stem_path in stem_paths:
        audio = AudioSegment.from_wav(stem_path)
        if combined is None:
            combined = audio
        else:
            combined = combined.overlay(audio)
    
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'final_mix.wav')
    combined.export(output_path, format='wav')
    
    return output_path

def process_ptx_file(ptx_path, config=None):
    logger.info("Attempting to parse .ptx file with pyaaf2...")
    try:
        import pyaaf2
        with pyaaf2.open(ptx_path, 'r') as f:
            logger.info("Successfully opened .ptx file")
            logger.warning(".ptx parsing is complex - falling back to WAV processing")
            return None
    except Exception as e:
        logger.warning(f"Failed to parse .ptx: {e}")
        return None

def process_zip_file(zip_path, config=None):
    logger.info("Extracting ZIP file...")
    extract_dir = tempfile.mkdtemp()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    wav_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(wav_files)} WAV files in ZIP")
    
    if len(wav_files) < 2 or len(wav_files) > 5:
        raise ValueError(f"Expected 2-5 WAV stems, found {len(wav_files)}")
    
    processed_stems = []
    for wav_file in wav_files:
        stem_name = os.path.splitext(os.path.basename(wav_file))[0]
        processed_path = process_stem(wav_file, stem_name, config)
        processed_stems.append(processed_path)
    
    final_mix = mix_stems(processed_stems)
    
    for stem in processed_stems:
        os.unlink(stem)
    shutil.rmtree(extract_dir)
    
    return final_mix

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        config = None
        if 'config' in request.files:
            config_file = request.files['config']
            if config_file and config_file.filename.endswith('.json'):
                config = json.load(config_file)
                logger.info(f"Loaded config: {config}")
        
        logger.info(f"Processing file: {filename}")
        
        if filename.lower().endswith('.ptx'):
            result = process_ptx_file(filepath, config)
            if result is None:
                return jsonify({'error': '.ptx parsing not fully implemented. Please upload a ZIP with WAV stems instead.'}), 400
        elif filename.lower().endswith('.zip'):
            result = process_zip_file(filepath, config)
        else:
            return jsonify({'error': 'Please upload a .ptx or .zip file'}), 400
        
        os.unlink(filepath)
        
        return jsonify({
            'success': True,
            'download_url': '/download/final_mix.wav'
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
