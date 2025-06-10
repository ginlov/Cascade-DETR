from dataset.dataset import SFCHD, collate_fn
from modules.DETR import DETR, build_position_encoding
from modules.backbone import make_backbone
from modules.transformer import build_transformer
from modules.loss import SetCriterion
from modules.matcher import HungarianMatcher

import argparse
import torch
import logging
import os

def argument_parser():
    parser = argparse.ArgumentParser(description="Train DETR on SFCHD dataset")
    parser.add_argument('--dataset', type=str, default='SFCHD', help='Dataset name')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the SFCHD dataset')
    parser.add_argument('--image-folder', type=str, required=True, help='Folder containing images')
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
    ## DETR parameters
    parser.add_argument('--N', type=int, default=100, help='Number of queries for DETR')
    parser.add_argument('--num-classes', type=int, default=91, help='Number of classes for DETR')
    ## Training parameters
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save model checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default='', help='Path to resume training from a checkpoint')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
    parser.add_argument('--save-freq', type=int, default=1, help='Frequency of saving checkpoints')
    ## HugarianMatcher parameters
    parser.add_argument('--cost-class', type=float, default=1.0, help='Weight for classification cost in HungarianMatcher')
    parser.add_argument('--cost-bbox', type=float, default=5.0, help='Weight for bounding box cost in HungarianMatcher')
    parser.add_argument('--cost-giou', type=float, default=2.0, help='Weight for GIoU cost in HungarianMatcher')
    ## SetCriterion parameters
    parser.add_argument('--eos-coef', type=float, default=0.1, help='Coefficient for the no-object class in SetCriterion')
    parser.add_argument('--losses', nargs='+', default=['labels', 'boxes', 'giou'], help='List of losses to compute')
    parser.add_argument('--labels-loss-coef', type=float, default=1.0, help='Coefficient for labels loss')
    parser.add_argument('--bbox-loss-coef', type=float, default=5.0, help='Coefficient for bounding box loss')
    parser.add_argument('--giou-loss-coef', type=float, default=2.0, help='Coefficient for GIoU loss')
    parser.add_argument('--log-interval', type=int, default=10, help='Interval for logging training progress')
    return parser

def main(args):
    # Set up logging to file
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")
    logger = logging.getLogger("detr_train")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logger.addHandler(logging.StreamHandler())  # Also log to stdout

    # Initialize the dataset
    if args.dataset == 'SFCHD':
        dataset = SFCHD(root=args.data_path, image_folder=args.image_folder, train=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Initialize the DETR model
    backbone = make_backbone(args)
    pos_encoder = build_position_encoding(args)
    transformer = build_transformer(args)
    detr_model = DETR(
        backbone=backbone,  # Replace with actual backbone
        pos_encoder=pos_encoder,  # Replace with actual position encoder
        transformer=transformer,  # Replace with actual transformer
        num_queries=args.N,
        num_classes=args.num_classes
    )

    # Set the device
    device = args.device if torch.cuda.is_available() else 'cpu'
    detr_model.to(device)
    print(f"Using device: {device}")
    # Define the optimizer
    # Set different learning rates for backbone and the rest
    param_dicts = [
        {"params": [p for n, p in detr_model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in detr_model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Define the loss function
    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=HungarianMatcher(cost_class=args.cost_class, cost_bbox=args.cost_bbox, cost_giou=args.cost_giou),  # Replace with actual matcher
        weight_dict={'loss_ce': args.labels_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef},  # Example weights
        eos_coef=0.1,
        losses=['labels', 'boxes', 'cardinality']
    )
    criterion.to(device)

    # Training loop
    for epoch in range(args.num_epochs):
        detr_model.train()
        criterion.train()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in tgt_item.items()} for tgt_item in targets]
            optimizer.zero_grad()
            outputs = detr_model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses.backward()
            optimizer.step()
            print(detr_model.query_embed.weight[0])
            print(detr_model.query_embed.weight[1])
            print(detr_model.query_embed.weight.mean(dim=0))

            if i % args.log_interval == 0:
                msg = f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {losses.item():.4f}"
                logger.info(msg)

        # Save the model checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = f"{args.output_dir}/detr_epoch_{epoch+1}.pth"
            torch.save(detr_model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    
    main(args)