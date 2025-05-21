# Photo Downscaler

A Python tool for downscaling photos and videos while preserving directory structure. It's designed to efficiently process large media collections, maintaining organization and reducing storage requirements.

## Features

- Downscales both images and videos to specified dimensions
- Preserves directory structure
- Maintains EXIF data for images
- Configurable quality settings for both images and videos
- Efficient parallel processing with configurable worker threads
- Tracks processed files to avoid reprocessing unchanged files
- Configurable path exclusions
- Smart video processing using FFmpeg
- Periodic metadata saving to prevent data loss

## Requirements

- Python 3.6 or higher
- FFmpeg (for video processing)
- Pillow (PIL Fork) for image processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/photo-downscaler.git
cd photo-downscaler
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

## Configuration

Create a `config.json` file in the project directory with the following structure:

```json
{
    "input_dirs": [
        "/path/to/input/dir1",
        "/path/to/input/dir2"
    ],
    "output_dir": "/path/to/output/dir",
    "width": 1920,
    "height": 1080,
    "quality": 75,
    "workers": 4,
    "video_preset": "medium",
    "video_crf": 23,
    "exclude_paths": ["Bulk", "Temp"]
}
```

### Configuration Options

- `input_dirs`: List of directories containing original media files
- `output_dir`: Directory to save downscaled media files
- `width`: Maximum width for downscaled media (default: 1920)
- `height`: Maximum height for downscaled media (default: 1080)
- `quality`: JPEG quality for images (1-100, default: 75)
- `workers`: Number of worker threads (default: 4)
- `video_preset`: FFmpeg preset for video processing (default: "medium")
- `video_crf`: Constant Rate Factor for video quality (0-51, lower is better, default: 23)
- `exclude_paths`: List of path components to exclude from processing

## Usage

Run the script with:

```bash
python photo_downscaler.py
```

Or specify a custom config file:

```bash
python photo_downscaler.py --config /path/to/config.json
```

## License

MIT License - see LICENSE file for details 
3. It keeps track of processed files in a JSON file (`processed_files.json`) in the output directory 