#!/usr/bin/env python3
"""
AI Photo Enhancement Tool - Natural Portrait Enhancement
Focus on faces with minimal artifacts and natural-looking results
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import urllib.request

def check_and_install_packages():
    """Check and install required packages"""
    required = {
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'torch': 'torch',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing required packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages"] + missing)
            print("Installation complete!\n")
        except:
            print("Auto-install failed. Please run:")
            print(f"pip install {' '.join(missing)} --break-system-packages")
            sys.exit(1)

check_and_install_packages()

from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import torch
import torch.nn as nn


class RRDBNet(nn.Module):
    """Real-ESRGAN Generator Network"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        self.body = nn.ModuleList([
            self._make_rrdb_block(num_features) for _ in range(num_blocks)
        ])
        
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
    
    def _make_rrdb_block(self, num_features):
        return nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat
        
        for block in self.body:
            body_feat = block(body_feat) + body_feat * 0.2
        
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        out = self.upsampler(feat)
        out = self.conv_last(out)
        
        return out


class NaturalPhotoEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.model_path = None
    
    def download_model(self, url, save_path):
        """Download model weights"""
        if os.path.exists(save_path):
            print(f"Model found at {save_path}")
            return save_path
        
        print(f"Downloading AI model (one-time, ~250MB)...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"✓ Model downloaded!")
            return save_path
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    def load_model(self, model_type='realesrgan'):
        """Load AI enhancement model"""
        if self.model is not None:
            return self.model
        
        print("\n[1/3] Loading AI model...")
        
        models_dir = os.path.join(os.path.expanduser('~'), '.cache', 'photo_enhancer')
        
        if model_type == 'realesrgan':
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            model_path = os.path.join(models_dir, 'RealESRGAN_x4plus.pth')
        else:
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
            model_path = os.path.join(models_dir, 'RealESRGAN_anime.pth')
        
        self.model_path = self.download_model(model_url, model_path)
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = RRDBNet(in_channels=3, out_channels=3, num_features=64, num_blocks=23, scale=4)
                
                loadnet = torch.load(self.model_path, map_location=self.device, weights_only=True)
                if 'params_ema' in loadnet:
                    loadnet = loadnet['params_ema']
                elif 'params' in loadnet:
                    loadnet = loadnet['params']
                
                self.model.load_state_dict(loadnet, strict=True)
                self.model.eval()
                self.model = self.model.to(self.device)
                
                print("✓ AI model loaded!")
                return self.model
            except Exception as e:
                print(f"✗ Model loading failed: {e}")
                self.model = None
        
        return None
    
    def enhance_with_ai(self, img, scale=4):
        """AI-based enhancement - no extra processing"""
        print("[2/3] Running AI enhancement...")
        
        img_tensor = torch.from_numpy(np.transpose(img / 255.0, (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
        
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        print("✓ AI enhancement complete")
        return output
    
    def fallback_enhance(self, img, scale):
        """Simple high-quality upscaling without over-processing"""
        print("[2/3] Using high-quality upscaling...")
        
        h, w = img.shape[:2]
        enhanced = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        
        # Very minimal noise reduction
        enhanced = cv2.bilateralFilter(enhanced, d=3, sigmaColor=10, sigmaSpace=10)
        
        print("✓ Upscaling complete")
        return enhanced
    
    def detect_faces(self, img):
        """Detect faces and return their locations"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def enhance_face_region(self, face_roi):
        """Enhance a single face region naturally"""
        # Smooth skin while preserving features
        # Use bilateral filter - smooths flat areas, keeps edges
        smoothed = cv2.bilateralFilter(face_roi, d=7, sigmaColor=50, sigmaSpace=50)
        
        # Very subtle contrast enhancement on face only
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced_face = cv2.merge([l, a, b])
        enhanced_face = cv2.cvtColor(enhanced_face, cv2.COLOR_LAB2BGR)
        
        return enhanced_face
    
    def create_smooth_mask(self, shape, center, radius):
        """Create a smooth circular mask for blending"""
        mask = np.zeros(shape[:2], dtype=np.float32)
        y, x = np.ogrid[:shape[0], :shape[1]]
        
        center_y, center_x = center
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Smooth falloff
        mask = np.clip(1.0 - (dist_from_center / radius), 0, 1)
        mask = mask ** 2  # Smoother falloff
        
        # Heavy blur for seamless blend
        mask = cv2.GaussianBlur(mask, (51, 51), 25)
        
        return np.stack([mask] * 3, axis=2)
    
    def enhance_faces_only(self, img):
        """Enhance faces while keeping background natural"""
        print("[2/3] Detecting and enhancing faces...")
        
        faces = self.detect_faces(img)
        
        if len(faces) == 0:
            print("  No faces detected")
            return img
        
        print(f"  Found {len(faces)} face(s)")
        
        # Work on a copy
        result = img.copy()
        
        for idx, (x, y, w, h) in enumerate(faces):
            print(f"  Enhancing face {idx + 1}/{len(faces)}...")
            
            # Expand region slightly for context
            padding = int(w * 0.4)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            # Extract face region
            face_roi = img[y1:y2, x1:x2].copy()
            
            # Enhance face region
            enhanced_face = self.enhance_face_region(face_roi)
            
            # Create smooth mask for blending
            center = ((y2 - y1) // 2, (x2 - x1) // 2)
            radius = max(w, h) * 0.6
            mask = self.create_smooth_mask(enhanced_face.shape, center, radius)
            
            # Blend enhanced face with original
            result[y1:y2, x1:x2] = (
                enhanced_face * mask + 
                result[y1:y2, x1:x2] * (1 - mask)
            ).astype(np.uint8)
        
        print("✓ Face enhancement complete")
        return result
    
    def minimal_post_process(self, img):
        """Minimal post-processing - just slight color adjustment"""
        print("[3/3] Final touches...")
        
        # Very subtle saturation boost
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.03, 0, 255)  # Just 3% boost
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        print("✓ Complete!")
        return img
    
    def process_image(self, input_path, output_path, mode='auto', scale=4, sharpen=False):
        """Process image naturally with focus on faces"""
        print(f"\nProcessing: {input_path}")
        print(f"Mode: {mode} | Scale: {scale}x | Sharpen: {sharpen}")
        
        # Read image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        print(f"Original size: {img.shape[1]}x{img.shape[0]}")
        
        # Detect faces first
        faces = self.detect_faces(img)
        has_faces = len(faces) > 0
        
        if has_faces:
            print(f"Portrait mode: {len(faces)} face(s) detected")
        
        # Enhancement strategy based on content
        if mode == 'face' or (mode == 'auto' and has_faces):
            # Face-focused enhancement
            # First upscale the whole image
            if scale > 1:
                model = self.load_model('realesrgan')
                if model is not None:
                    try:
                        enhanced = self.enhance_with_ai(img, scale)
                    except:
                        enhanced = self.fallback_enhance(img, scale)
                else:
                    enhanced = self.fallback_enhance(img, scale)
            else:
                enhanced = img.copy()
            
            # Then enhance faces specifically
            enhanced = self.enhance_faces_only(enhanced)
            
        else:
            # General enhancement
            if scale > 1:
                model = self.load_model('realesrgan' if mode != 'anime' else 'anime')
                if model is not None:
                    try:
                        enhanced = self.enhance_with_ai(img, scale)
                    except:
                        enhanced = self.fallback_enhance(img, scale)
                else:
                    enhanced = self.fallback_enhance(img, scale)
            else:
                enhanced = img.copy()
        
        # Optional light sharpening (disabled by default)
        if sharpen:
            kernel = np.array([[0, -0.2, 0], [-0.2, 1.8, -0.2], [0, -0.2, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Minimal color adjustment
        enhanced = self.minimal_post_process(enhanced)
        
        print(f"\nFinal size: {enhanced.shape[1]}x{enhanced.shape[0]}")
        print(f"Saving to: {output_path}")
        
        # Save with high quality
        cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95,
                                             cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"✓ Saved!")
        
        return enhanced


def process_folder(input_folder, output_folder, mode, scale, sharpen):
    """Process all images in a folder"""
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all image files
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' not found!")
        return
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in '{input_folder}'")
        return
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Output folder: {output_folder}\n")
    
    # Process each image
    enhancer = NaturalPhotoEnhancer()
    success_count = 0
    failed_files = []
    
    for idx, img_file in enumerate(image_files, 1):
        print("="*70)
        print(f"Processing [{idx}/{len(image_files)}]: {img_file.name}")
        print("="*70)
        
        try:
            output_file = output_path / img_file.name
            enhancer.process_image(str(img_file), str(output_file), mode, scale, sharpen)
            success_count += 1
            print(f"✓ Saved: {output_file}\n")
        except Exception as e:
            print(f"✗ Failed: {e}\n")
            failed_files.append(img_file.name)
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {success_count}/{len(image_files)}")
    print(f"Output folder: {output_folder}")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for filename in failed_files:
            print(f"  - {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Natural AI Photo Enhancement - Focus on Faces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhancement Modes:
  auto   - Automatically detect and enhance faces (recommended)
  face   - Force face enhancement mode
  photo  - General photo enhancement
  anime  - Anime/illustration enhancement

Single Image Examples:
  # Auto-detect and enhance (recommended for portraits)
  python photo_enhancer.py photo.jpg

  # Force face enhancement mode
  python photo_enhancer.py portrait.jpg -m face

  # Add subtle sharpening (optional)
  python photo_enhancer.py photo.jpg --sharpen

  # 2x upscaling (faster)
  python photo_enhancer.py photo.jpg -s 2

Batch Processing Examples:
  # Enhance all images in a folder
  python photo_enhancer.py /path/to/photos -b /path/to/enhanced

  # Batch with face mode
  python photo_enhancer.py ~/Pictures/portraits -b ~/Pictures/enhanced -m face

  # Batch with 2x scaling (faster)
  python photo_enhancer.py ./input_folder -b ./output_folder -s 2

  # Quick batch enhancement
  python photo_enhancer.py ./photos -b ./enhanced_photos

Tips:
  - Use -b flag for batch processing entire folders
  - Auto mode detects faces and enhances them naturally
  - Background stays natural with minimal processing
  - Progress shown for each image in batch mode
        """
    )
    
    parser.add_argument('input', help='Input image path or folder path (for batch)')
    parser.add_argument('-b', '--batch', dest='output_folder', 
                       help='Output folder for batch processing (enables batch mode)')
    parser.add_argument('-o', '--output', help='Output image path (single file mode)')
    parser.add_argument('-m', '--mode', 
                       choices=['auto', 'face', 'photo', 'anime'], 
                       default='auto',
                       help='Enhancement mode (default: auto)')
    parser.add_argument('-s', '--scale', type=int, default=4,
                       choices=[1, 2, 4],
                       help='Upscaling factor: 1=no upscale, 2, or 4 (default: 4)')
    parser.add_argument('--sharpen', action='store_true',
                       help='Add subtle sharpening (use carefully)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' not found!")
        sys.exit(1)
    
    print("="*70)
    print("NATURAL AI PHOTO ENHANCEMENT - PORTRAIT FOCUSED")
    print("="*70)
    
    # Check if batch mode
    if args.output_folder:
        # Batch processing mode
        process_folder(args.input, args.output_folder, args.mode, args.scale, args.sharpen)
    else:
        # Single file mode
        if os.path.isdir(args.input):
            print("\nError: Input is a folder but batch mode not enabled!")
            print("Use -b flag to specify output folder for batch processing:")
            print(f"  python photo_enhancer.py {args.input} -b ./output_folder")
            sys.exit(1)
        
        if args.output is None:
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}")
        
        enhancer = NaturalPhotoEnhancer()
        
        try:
            enhancer.process_image(args.input, args.output, args.mode, args.scale, args.sharpen)
            
            print("\n" + "="*70)
            print("✓ SUCCESS!")
            print("="*70)
            print(f"\nOriginal: {args.input}")
            print(f"Enhanced: {args.output}")
            
        except Exception as e:
            print("\n" + "="*70)
            print(f"✗ Error: {e}")
            print("="*70)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
