from dataset.dataset import SFCHD, collate_fn
from modules.DETR import DETR, build_position_encoding
from modules.backbone import make_backbone
from modules.transformer import build_transformer
from PIL import Image, ImageDraw

import argparse
import torch
import os
import json
import numpy as np

def argument_parser():
    parser = argparse.ArgumentParser(description="Test DETR on SFCHD dataset")
    parser.add_argument('--dataset', type=str, default='SFCHD', help='Dataset name')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the SFCHD dataset')
    parser.add_argument('--image-folder', type=str, required=True, help='Folder containing images')
    parser.add_argument('--N', type=int, default=100, help='Number of queries for DETR')
    parser.add_argument('--num-classes', type=int, default=91, help='Number of classes for DETR')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for testing (e.g., cuda or cpu)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./test_outputs', help='Directory to save outputs')
    ## Transformer parameters
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for transformer')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads in transformer')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of feedforward network in transformer')
    parser.add_argument('--enc_layers', type=int, default=6, help='Number of encoder layers in transformer')
    parser.add_argument('--dec_layers', type=int, default=6, help='Number of decoder layers in transformer')
    parser.add_argument('--pre_norm', action='store_true', help='Use pre-norm in transformer layers')
    ## Position embedding parameters
    parser.add_argument('--position-embedding', type=str, default='sine', choices=['sine', 'learned'], help='Type of position embedding')
    ## Backbone parameters
    parser.add_argument('--backbone-name', type=str, default='resnet50', help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--batch-norm-freeze', action='store_true', help='Freeze batch normalization layers in the backbone')
    parser.add_argument('--resolution-increase', action='store_true', help='Increase resolution in the backbone')
    return parser

def plot_results(pil_img, boxes, labels, scores, class_names=None, score_thresh=0.1, gt_boxes=None):
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    print(scores)
    # Draw predicted boxes (cx, cy, w, h format)
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x_c, y_c, bw, bh = box
        x_c *= w
        y_c *= h
        bw *= w
        bh *= h
        x0 = x_c - bw / 2
        y0 = y_c - bh / 2
        x1 = x_c + bw / 2
        y1 = y_c + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        label_text = str(label)
        if class_names and label < len(class_names):
            label_text = class_names[label]
        draw.text((x0, y0), f"{label_text}:{score:.2f}", fill="red")

    # Draw ground truth boxes (in green, cx, cy, w, h format)
    if gt_boxes is not None:
        for gt_box in gt_boxes:
            x_c, y_c, bw, bh = gt_box
            x_c *= w
            y_c *= h
            bw *= w
            bh *= h
            x0 = x_c - bw / 2
            y0 = y_c - bh / 2
            x1 = x_c + bw / 2
            y1 = y_c + bh / 2
            draw.rectangle([x0, y0, x1, y1], outline="green", width=2)

    return pil_img

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the dataset
    if args.dataset == 'SFCHD':
        dataset = SFCHD(root=args.data_path, image_folder=args.image_folder, train=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print('init model')
    # Initialize the DETR model
    backbone = make_backbone(args)
    pos_encoder = build_position_encoding(args)
    transformer = build_transformer(args)
    detr_model = DETR(
        backbone=backbone,
        pos_encoder=pos_encoder,
        transformer=transformer,
        num_queries=args.N,
        num_classes=args.num_classes
    )

    device = args.device if torch.cuda.is_available() else 'cpu'
    detr_model.to(device)
    detr_model.eval()

    print('start loading checkpoint')
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    detr_model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    results = []
    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            outputs = detr_model(images)
            # Move outputs to cpu and convert to list for JSON serialization
            output_dict = {k: v.cpu().tolist() for k, v in outputs.items()}
            # Optionally, add image index or filename if available
            result = {
                "image_idx": idx,
                "outputs": output_dict
            }
            results.append(result)
            print(f"Processed image {idx+1}/{len(test_loader)}")

            # --- Visualization ---
            # Assume batch size 1 for visualization
            img_tensor = images[0].cpu()
            # Undo normalization if needed (adjust as per your dataset transforms)
            img_np = img_tensor.permute(1,2,0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            # Get predicted boxes and scores
            pred_logits = torch.tensor(output_dict['pred_logits'])[0]  # [num_queries, num_classes]
            pred_boxes = torch.tensor(output_dict['pred_boxes'])[0]    # [num_queries, 4]
            prob = pred_logits.softmax(-1)
            scores, labels = prob.max(-1)
            # Filter out predictions for "no object" (last class)
            keep = labels != (args.num_classes - 1)
            boxes = pred_boxes[keep].numpy()
            labels = labels[keep].numpy()
            scores = scores[keep].numpy()
            # Draw and save
            gt_boxes = None
            if targets and "boxes" in targets[0]:
                gt_boxes = targets[0]["boxes"].cpu().numpy()
            vis_img = plot_results(pil_img, boxes, labels, scores, gt_boxes=gt_boxes)
            vis_path = os.path.join(args.output_dir, f"vis_{idx}.png")
            vis_img.save(vis_path)
            print(f"Saved visualization to {vis_path}")

    # Save all outputs to a JSON file
    output_path = os.path.join(args.output_dir, "outputs.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved outputs to {output_path}")

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
