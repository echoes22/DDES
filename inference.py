#!/usr/bin/env python3
"""
Inference script for running emotion prediction on image folders.

Usage:
    python -m public_emotion_pred.inference \
        --checkpoint path/to/model.pt \
        --image-dir path/to/images \
        --model convnext_8 \
        --output-dir ./results
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

from .models import get_model
from .inference_utils import (
    soft_argmax_2d, weighted_average_to_va, va_to_emotion_distribution,
    EMOTION_COORDS, EMOTION_NAMES
)
from .kde import sample_kde_batch


# Image extensions to search for
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def get_image_transform():
    """Get standard image transform for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path, transform):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_image_files(image_dir):
    """Get all image files from directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(sorted(image_dir.glob(f'*{ext}')))
        image_files.extend(sorted(image_dir.glob(f'*{ext.upper()}')))
    
    return list(set(image_files))  # Remove duplicates


def run_inference(
    image_dir,
    model,
    device,
    model_type='convnext_8',
    batch_size=32,
    transform=None
):
    """
    Run inference on a folder of images.
    
    Args:
        image_dir: Path to directory containing images
        model: Model instance
        device: torch.device
        model_type: Type of model (convnext_unet | convnext_8 | convnext_2)
        batch_size: Batch size for inference
        transform: Image transform to apply
    
    Returns:
        results: Dict of {image_name: predictions}
    """
    if transform is None:
        transform = get_image_transform()
    
    model.eval()
    image_files = get_image_files(image_dir)
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    results = {}
    
    with torch.no_grad():
        for batch_idx in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[batch_idx:batch_idx + batch_size]
            batch_images = []
            valid_files = []
            
            for img_file in batch_files:
                img_tensor = load_image(str(img_file), transform)
                if img_tensor is not None:
                    batch_images.append(img_tensor)
                    valid_files.append(img_file)
            
            if not batch_images:
                continue
            
            batch_images = torch.stack(batch_images).to(device)
            outputs = model(batch_images)
            
            for i, img_file in enumerate(valid_files):
                img_name = img_file.name
                
                if model_type == 'convnext_unet':
                    # outputs: (batch, 1, 28, 28) — already log-softmax from ConvNeXtDecoder
                    heatmap = outputs[i:i+1]
                    hm_softmax = heatmap.exp()

                    # Convert to emotion distribution via KDE (sample_kde_batch returns log-probs)
                    emotion_dist = sample_kde_batch(hm_softmax.squeeze(1))
                    emotion_probs = emotion_dist.exp().squeeze().cpu().numpy()
                    
                    # Convert to VA
                    va = soft_argmax_2d(hm_softmax.squeeze(1)).squeeze().cpu().numpy()
                    
                    results[img_name] = {
                        'model_type': 'convnext_unet',
                        'va': [float(va[0]), float(va[1])],
                        'emotion_distribution': {
                            name: float(prob) for name, prob in zip(EMOTION_NAMES, emotion_probs)
                        },
                        'top_emotion': EMOTION_NAMES[np.argmax(emotion_probs)]
                    }
                
                elif model_type == 'convnext_8':
                    # outputs: (batch, 8) — already log-softmax from CustomConvNeXt
                    log_probs = outputs[i]
                    probs = log_probs.exp().cpu().numpy()

                    # Convert to VA
                    va = weighted_average_to_va(log_probs.exp().unsqueeze(0)).squeeze().cpu().numpy()
                    
                    results[img_name] = {
                        'model_type': 'convnext_8',
                        'va': [float(va[0]), float(va[1])],
                        'emotion_distribution': {
                            name: float(prob) for name, prob in zip(EMOTION_NAMES, probs)
                        },
                        'top_emotion': EMOTION_NAMES[np.argmax(probs)]
                    }
                
                elif model_type == 'convnext_2':
                    # outputs: (batch, 2) - VA coordinates
                    va = outputs[i].cpu().numpy()

                    # Convert to emotion distribution (va_to_emotion_distribution returns probabilities)
                    emotion_probs = va_to_emotion_distribution(outputs[i:i+1]).squeeze().cpu().numpy()
                    
                    results[img_name] = {
                        'model_type': 'convnext_2',
                        'va': [float(va[0]), float(va[1])],
                        'emotion_distribution': {
                            name: float(p) for name, p in zip(EMOTION_NAMES, emotion_probs)
                        },
                        'top_emotion': EMOTION_NAMES[int(np.argmax(emotion_probs))]
                    }
    
    return results


def save_results(results, output_dir, output_format='json'):
    """Save inference results."""
    os.makedirs(output_dir, exist_ok=True)
    
    if output_format == 'json':
        output_file = os.path.join(output_dir, 'predictions.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    elif output_format == 'csv':
        import csv
        output_file = os.path.join(output_dir, 'predictions.csv')
        with open(output_file, 'w', newline='') as f:
            if results:
                first_result = next(iter(results.values()))
                fieldnames = ['image_name', 'top_emotion', 'valence', 'arousal'] + \
                            [f'{name}_prob' for name in EMOTION_NAMES]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for img_name, pred in results.items():
                    row = {
                        'image_name': img_name,
                        'top_emotion': pred['top_emotion'],
                        'valence': pred['va'][0],
                        'arousal': pred['va'][1]
                    }
                    for name in EMOTION_NAMES:
                        row[f'{name}_prob'] = pred['emotion_distribution'][name]
                    writer.writerow(row)
        
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on image folder')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--model', type=str, default='convnext_8',
                        choices=['convnext_unet', 'convnext_8', 'convnext_2'],
                        help='Model type')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--format', type=str, default='json',
                        choices=['json', 'csv', 'both'],
                        help='Output format')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = get_model(args.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Run inference
    print(f"Running inference on {args.image_dir}")
    results = run_inference(
        args.image_dir,
        model,
        device,
        model_type=args.model,
        batch_size=args.batch_size
    )
    
    # Save results
    if args.format == 'both':
        save_results(results, args.output_dir, 'json')
        save_results(results, args.output_dir, 'csv')
    else:
        save_results(results, args.output_dir, args.format)
    
    print(f"Inference complete! Processed {len(results)} images")
    
    # Print sample results
    print("\nSample results:")
    for img_name, pred in list(results.items())[:3]:
        print(f"\n{img_name}:")
        print(f"  Top emotion: {pred['top_emotion']}")
        print(f"  VA: [{pred['va'][0]:.3f}, {pred['va'][1]:.3f}]")


if __name__ == '__main__':
    main()
