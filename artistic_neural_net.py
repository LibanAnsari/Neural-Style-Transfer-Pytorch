from pathlib import Path
import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore")

# Define globals
device = None
model = None

# Heavy imports will be stored here or imported when needed
def init_globals():
    global device, model, torch, nn, ContentLoss, StyleLoss, tv_loss, SummaryWriter, utils, transforms, tqdm
    import torch
    from torch import nn
    from models.model import vgg_model
    from losses.content import ContentLoss
    from losses.style import StyleLoss
    from losses.tv_loss import tv_loss
    from torch.utils.tensorboard import SummaryWriter
    from utils import utils, transforms
    from tqdm.auto import tqdm
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f"[INFO] Using device: {device}")

    model = vgg_model.to(device)
    model.eval()
    print(f"[INFO] Model Initialized successfully.")


def compute_losses(model, image, args, content_loss, style_losses):
    features = model(image.unsqueeze(0))

    content_loss(features[args.content_layer])
    content_val = content_loss.loss.item()

    style_val = 0.0
    for layer in args.style_layers:
        style_losses[layer](features[layer])
        style_val += style_losses[layer].loss.item()

    total_val = args.alpha * content_val + args.beta * style_val

    return content_val, style_val, total_val


def train(args, content_loss, style_losses, generated_image, content_image, style_image):
    log_dir = f"runs/{Path(args.content_path).stem}/{Path(args.style_path).stem}/{args.output_name}"
    writer = SummaryWriter(log_dir=log_dir)
    print("[INFO] Tensorboard Writer created successfully.")
    
    # Log content and style images only once 
    with torch.no_grad():
        content_img = utils.denormalize(content_image.detach().clone()).clamp(0, 1)
        style_img = utils.denormalize(style_image.detach().clone()).clamp(0, 1)

        writer.add_image("Input/Content Image", content_img.squeeze().cpu(), 0)
        writer.add_image("Input/Style Image", style_img.squeeze().cpu(), 0)
    
    # Set up the optimizer (LFGBS is 100x better than Adam here)
    optimizer = torch.optim.LBFGS([generated_image])

    print(f"[INFO] Training for {args.epochs} iterations")
    for step in tqdm(range(args.epochs)):

        def closure():
            # forward pass
            features = model(generated_image.unsqueeze(0))
            
            # content loss
            content_loss(features[args.content_layer])
            
            # style loss
            style_score = 0.0
            for layer in args.style_layers:
                style_losses[layer](features[layer])
                style_score += style_losses[layer].loss
            
            # Total loss
            total_loss = args.alpha * content_loss.loss + args.beta * style_score
            total_loss += 1e-6 * tv_loss(generated_image.unsqueeze(0))

            optimizer.zero_grad()
            total_loss.backward()
            
            return total_loss

        optimizer.step(closure) # 20 internal evaulations so, 20 x 25 ~= 500 steps

        with torch.no_grad():
            generated_image.clamp_(-3, 3)  # normalized space clamp
        
        with torch.no_grad():
            content_val, style_val, total_val = compute_losses(
                model, generated_image, args, content_loss, style_losses
            )

        if step % 2 == 0:
            writer.add_scalar("Loss/Content", content_val, step)
            writer.add_scalar("Loss/Style", style_val, step)
            writer.add_scalar("Loss/Total", total_val, step)
            
            img = utils.denormalize(generated_image.detach().clone())
            img = img.clamp(0, 1)
            writer.add_image(
                tag="generated_image",
                img_tensor=img.squeeze().cpu(),
                global_step=step
            )
        
        print(f"\n[INFO] Epoch: {step + 1} | Content Loss: {content_val:.4f} | Style Score: {style_val:.4f} | Total Loss: {total_val:.4f}")
        
    writer.close()
        
    return generated_image


def generate_image(args):
    # Prepare Images in correct format and device
    content_transforms, style_transforms = transforms.get_transforms(args.img_size)

    content_image = utils.img_path_to_pil(args.content_path)
    content_image = content_transforms(content_image).to(device)
    
    style_image = utils.img_path_to_pil(args.style_path)
    style_image = style_transforms(style_image).to(device)

    # Intialize target values for loss funtion (only computed once)
    with torch.no_grad():
        content_targets = model(content_image.unsqueeze(0))
        style_targets = model(style_image.unsqueeze(0))

    content_loss = ContentLoss(content_targets[args.content_layer])
    style_losses = {
        layer: StyleLoss(style_targets[layer], args.wl)
        for layer in args.style_layers
    }

    # Create the Generated Image (Image to optimize)
    generated_image = content_image.clone()
    noise = torch.rand_like(generated_image) * 0.8
    generated_image = generated_image + noise
    
    generated_image = nn.Parameter(generated_image.to(device).detach(), requires_grad=True) # Gradients to track

    generated_image = train(
        args, content_loss, style_losses, generated_image, content_image, style_image
    )

    return generated_image


def main():
    parser = argparse.ArgumentParser(description="Artistic Neural Style Transfer CLI🎨",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument(
        "--content-path",
        type=str,
        required=True,
        help="Path to the content image"
    )
    parser.add_argument(
        "--style-path",
        type=str,
        required=True,
        help="Path to the style image"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs",
        help="Directory to save generated image"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="generated_art",
        help="Output file name for the generated image"
    )

    # Image settings
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Resize images to this size"
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of optimization steps"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Content loss weight"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e3,
        help="Style loss weight"
    )

    # Layers
    parser.add_argument(
        "--content-layer",
        type=str,
        default="conv4_1",
        help="Content loss layer, single layer only"
    )
    parser.add_argument(
        "--style-layers",
        nargs="+",
        default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        help="Style loss layers, space separated"
    )

    # Style weight per layer
    parser.add_argument(
        "--wl",
        type=float,
        default=None,
        help="Weight per style layer (1 / num_layers)"
    )

    args = parser.parse_args()

    # Validate paths before loading libraries
    if not os.path.exists(args.content_path):
        print(f"[ERROR] Content image not found: {args.content_path}")
        sys.exit(1)
        
    if not os.path.exists(args.style_path):
        print(f"[ERROR] Style image not found: {args.style_path}")
        sys.exit(1)

    # Initialize heavy libraries and globals only after parsing args
    init_globals()

    # Set default wl if not provided
    if args.wl is None:
        args.wl = 1.0 / len(args.style_layers)

    os.makedirs(args.save_path, exist_ok=True)

    # Run generation
    image = generate_image(args)

    # Save image
    utils.save_image(
        image,
        args.save_path,
        Path(args.content_path),
        Path(args.style_path),
        args.output_name
    )
    
if __name__ == "__main__":
    main()