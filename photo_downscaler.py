#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import json
import shutil
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global folder renaming map
# Format: "original/path" -> "renamed path"
# Example: "Wallpapers/Nature" -> "Wallpapers - Nature"
FOLDER_RENAME_MAP = {
    "Wallpapers/Nature": "Wallpapers - Nature",
    "Wallpapers/Family": "Wallpapers - Family",
}

class MediaDownscaler:
    def __init__(self, input_dirs, output_dir, max_size=(1920, 1080), quality=75, workers=4, rename_map=None,
                 video_preset="medium", video_crf=23, exclude_paths=None):
        """
        Initialize the media downscaler.
        
        Args:
            input_dirs: List of directories containing the original media files
            output_dir: Directory to save downscaled media files
            max_size: Maximum dimensions (width, height) for the downscaled images
            quality: JPEG quality (1-100)
            workers: Number of worker threads
            rename_map: Dictionary mapping original folder paths to renamed paths
            video_preset: FFmpeg preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            video_crf: Constant Rate Factor for video quality (0-51, lower is better quality, 23 is default)
            exclude_paths: List of path components to exclude from processing
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.max_size = max_size
        self.quality = quality
        self.workers = workers
        self.rename_map = rename_map or FOLDER_RENAME_MAP
        self.metadata_file = self.output_dir / "photo_downscaler.metadata.json"
        self.processed_files = self._load_processed_files()
        self.video_preset = video_preset
        self.video_crf = video_crf
        self.last_flush_time = time.time()  # Initialize flush time
        self.exclude_paths = set(exclude_paths or [])  # Convert to set for faster lookups
        
        # Check if FFmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available on the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info("FFmpeg is available: video processing enabled")
                return True
            else:
                logger.error("FATAL ERROR: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"FATAL ERROR: FFmpeg not found or cannot be executed: {e}")
            sys.exit(1)
    
    def _load_processed_files(self):
        """Load the list of already processed files from metadata file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading processed files metadata: {e}")
                return {}
        return {}
    
    def _save_processed_files(self):
        """Save the list of processed files to metadata file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving processed files metadata: {e}")
    
    def _get_file_hash(self, file_path):
        """Calculate a hash of the file for tracking changes."""
        try:
            # For large files like videos, only hash the first 10MB to save time
            file_size = os.path.getsize(file_path)
            hash_size = min(10 * 1024 * 1024, file_size)  # 10MB or file size, whichever is smaller
            
            with open(file_path, 'rb') as f:
                file_content = f.read(hash_size)
                # Add modification time and file size to the hash for better uniqueness
                mod_time = str(os.path.getmtime(file_path)).encode()
                size_bytes = str(file_size).encode()
                return hashlib.md5(file_content + mod_time + size_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def _is_image_file(self, file_path):
        """Check if a file is an image based on its extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return file_path.suffix.lower() in image_extensions
    
    def _is_video_file(self, file_path):
        """Check if a file is a video based on its extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.mpg', '.mpeg', '.mkv', '.flv'}
        return file_path.suffix.lower() in video_extensions
    
    def _should_skip_path(self, path):
        """Check if a path should be skipped based on exclusion list."""
        # Check if any part of the path matches an excluded component
        return any(excluded in path.parts for excluded in self.exclude_paths)
    
    def _get_renamed_path(self, rel_path):
        """
        Get renamed path based on the folder rename map.
        
        Args:
            rel_path: Relative path from input directory
            
        Returns:
            Path object with renamed directories if matching rules found
        """
        # Convert to string for matching
        path_str = str(rel_path).replace('\\', '/')
        
        # Check if any patterns in rename map match the start of the path
        for original, renamed in self.rename_map.items():
            # Normalize path separators to forward slashes for matching
            original_norm = original.replace('\\', '/')
            
            # If the path starts with the pattern followed by a slash or if it equals the pattern exactly
            if path_str.startswith(original_norm + '/') or path_str == original_norm:
                # Replace the matched pattern with its renamed version
                new_path = path_str.replace(original_norm, renamed, 1)
                return Path(new_path)
        
        # No matches found, return the original path
        return rel_path
    
    def _get_output_path(self, file_path):
        """Generate the output path for a file using renaming rules while preserving directory structure."""
        # Get relative path from input directory
        rel_path = None
        for input_dir in self.input_dirs:
            try:
                rel_path = file_path.relative_to(input_dir)
                break
            except ValueError:
                continue
        
        if rel_path is None:
            logger.error(f"File {file_path} is not in any input directory")
            return None, None, None
        
        # Check if the path matches any renaming rules
        renamed_path = self._get_renamed_path(rel_path)
        
        # Create output path preserving directory structure
        output_path = self.output_dir / renamed_path
        
        # Create output directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path, str(rel_path), str(renamed_path)
    
    def _check_and_flush_metadata(self):
        """Check if 30 seconds have passed since last flush and save if needed."""
        current_time = time.time()
        if current_time - self.last_flush_time >= 30:
            self._save_processed_files()
            self.last_flush_time = current_time
    
    def _process_image(self, image_path):
        """Process a single image file."""
        if self._should_skip_path(image_path):
            return True
        
        # Get output path and path strings
        output_path, rel_path_str, new_rel_path_str = self._get_output_path(image_path)
        
        # Get file hash to check if it's been processed
        file_hash = self._get_file_hash(image_path)
        if file_hash is None:
            return False
        
        # Check if file has already been processed and hasn't changed
        if rel_path_str in self.processed_files and self.processed_files[rel_path_str] == file_hash:
            return True
        
        try:
            # Open and process image
            with Image.open(image_path) as img:
                # Preserve image format
                img_format = img.format
                
                # Get original EXIF data and handle orientation
                exif_bytes = img.info.get('exif', b'')
                
                # Convert to RGB if needed (for PNG with transparency)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                    img = background
                
                # Calculate new dimensions while maintaining aspect ratio
                img.thumbnail(self.max_size, Image.LANCZOS)
                
                # Save the resized image, preserving format and quality
                save_kwargs = {
                    'format': img_format,
                    'quality': self.quality,
                    'optimize': True
                }
                
                # Add EXIF data if we have it
                if exif_bytes:
                    save_kwargs['exif'] = exif_bytes
                
                img.save(output_path, **save_kwargs)
                
                # Update processed files dictionary with original path as key
                self.processed_files[rel_path_str] = file_hash
                
                logger.info(f"Processed image: {rel_path_str} -> {new_rel_path_str}")
                
                # Check if we need to flush metadata
                self._check_and_flush_metadata()
                
                return True
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return False
    
    def _process_video(self, video_path):
        """Process a single video file using FFmpeg."""
        if self._should_skip_path(video_path):
            return True
        
        # Get output path and path strings
        output_path, rel_path_str, new_rel_path_str = self._get_output_path(video_path)
        
        # Get file hash to check if it's been processed
        file_hash = self._get_file_hash(video_path)
        if file_hash is None:
            return False
        
        # Check if file has already been processed and hasn't changed
        if rel_path_str in self.processed_files and self.processed_files[rel_path_str] == file_hash:
            return True
        
        try:
            # Ensure output is mp4 for best compatibility
            output_path = output_path.with_suffix('.mp4')
            
            # Build FFmpeg command for smartphone-friendly video
            # -vf scale parameters ensure we don't upscale small videos
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-c:v", "libx264", "-preset", self.video_preset, "-crf", str(self.video_crf),
                "-c:a", "aac", "-b:a", "128k",
                "-vf", f"scale='min({self.max_size[0]},iw)':'min({self.max_size[1]},ih)':force_original_aspect_ratio=decrease",
                "-movflags", "+faststart",  # Optimize for web streaming
                "-metadata", "title=",  # Clear metadata to save space
                str(output_path)
            ]
            
            # Run FFmpeg process
            logger.info(f"Processing video: {video_path}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Update processed files dictionary with original path as key
                self.processed_files[rel_path_str] = file_hash
                
                logger.info(f"Processed video: {rel_path_str} -> {output_path.name}")
                
                # Check if we need to flush metadata
                self._check_and_flush_metadata()
                
                return True
            else:
                logger.error(f"FFmpeg error processing {video_path}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return False
    
    def _delete_removed_files(self, current_files):
        """Delete files from output directory that no longer exist in the input directory."""
        current_relative_paths = {str(p.relative_to(self.input_dirs[0])) for p in current_files}
        files_to_remove = []
        
        # Find files that no longer exist in the input directory
        for rel_path in list(self.processed_files.keys()):
            if rel_path not in current_relative_paths:
                # Get the output path using the same path resolution logic as processing
                output_path = self.output_dir / self._get_renamed_path(Path(rel_path))
                files_to_remove.append((rel_path, output_path))
        
        # Delete files and their entries in processed_files
        for rel_path, output_path in files_to_remove:
            try:
                if output_path.exists():
                    if output_path.is_file():
                        output_path.unlink()
                    else:
                        shutil.rmtree(output_path)
                    logger.info(f"Deleted: {output_path} (no longer in source)")
                
                # Remove from processed_files
                del self.processed_files[rel_path]
            except Exception as e:
                logger.error(f"Error deleting {output_path}: {e}")
        
        # Clean up empty directories
        for dirpath, dirnames, filenames in os.walk(self.output_dir, topdown=False):
            if dirpath != str(self.output_dir) and not dirnames and not filenames:
                try:
                    os.rmdir(dirpath)
                    logger.info(f"Removed empty directory: {dirpath}")
                except Exception as e:
                    logger.error(f"Error removing directory {dirpath}: {e}")

    def process_directory(self):
        """Process all media files in the input directories."""
        all_files = []
        all_image_files = []
        all_video_files = []
        
        # Check all input directories exist
        for input_dir in self.input_dirs:
            if not input_dir.exists():
                logger.error(f"Input directory does not exist: {input_dir}")
                return False
            
            # Find all image and video files in this directory
            image_files = [p for p in input_dir.glob('**/*') 
                         if p.is_file() and self._is_image_file(p) and not self._should_skip_path(p)]
            
            video_files = [p for p in input_dir.glob('**/*') 
                         if p.is_file() and self._is_video_file(p) and not self._should_skip_path(p)]
            
            all_image_files.extend(image_files)
            all_video_files.extend(video_files)
        
        all_files = all_image_files + all_video_files
        
        logger.info(f"Found {len(all_image_files)} image files and {len(all_video_files)} video files to process")
        
        # Process files with thread pool
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Process images
            if all_image_files:
                img_results = list(executor.map(self._process_image, all_image_files))
                results.extend(img_results)
                
            # Process videos (serially if only 1 worker to avoid overloading)
            if all_video_files:
                if self.workers == 1 or len(all_video_files) == 1:
                    # Process videos serially if only one worker
                    video_results = [self._process_video(video) for video in all_video_files]
                else:
                    # Process videos in parallel with fewer workers to avoid overloading
                    video_workers = max(1, min(self.workers // 2, len(all_video_files)))
                    with ThreadPoolExecutor(max_workers=video_workers) as video_executor:
                        video_results = list(video_executor.map(self._process_video, all_video_files))
                results.extend(video_results)
        
        # Delete files that no longer exist in any source directory
        self._delete_removed_files(all_files)
        
        # Save processed files metadata one final time
        self._save_processed_files()
        
        # Summary
        successful = sum(1 for r in results if r)
        logger.info(f"Processing complete. Successfully processed {successful} of {len(all_files)} files.")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Downscale photos and videos while preserving directory structure")
    parser.add_argument("--config",
                        help="Path to JSON configuration file (defaults to config.json in script directory)")
    
    args = parser.parse_args()
    
    # Use default config.json in script directory if no config specified
    config_path = args.config if args.config else Path(__file__).parent / "config.json"
    
    # Load configuration from JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        return 1
    
    # Validate required configuration fields
    required_fields = ['input_dirs', 'output_dir']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        logger.error(f"Missing required configuration fields: {', '.join(missing_fields)}")
        return 1
    
    logger.info(f"Starting media downscaler")
    logger.info(f"Input directories: {', '.join(config['input_dirs'])}")
    logger.info(f"Output directory: {config['output_dir']}")
    logger.info(f"Max dimensions: {config.get('width', 1920)}x{config.get('height', 1080)}, Image quality: {config.get('quality', 75)}%")
    logger.info(f"Video settings: preset={config.get('video_preset', 'medium')}, crf={config.get('video_crf', 23)}")
    if config.get('exclude_paths'):
        logger.info(f"Excluding paths containing: {', '.join(config['exclude_paths'])}")
    
    # Create the downscaler - this will check for FFmpeg and exit if not found
    downscaler = MediaDownscaler(
        input_dirs=config['input_dirs'],
        output_dir=config['output_dir'],
        max_size=(config.get('width', 1920), config.get('height', 1080)),
        quality=config.get('quality', 75),
        workers=config.get('workers', 4),
        rename_map=config.get('rename_map'),
        video_preset=config.get('video_preset', 'medium'),
        video_crf=config.get('video_crf', 23),
        exclude_paths=config.get('exclude_paths')
    )
    
    success = downscaler.process_directory()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 