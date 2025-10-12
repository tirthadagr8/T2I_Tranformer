import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, T5EncoderModel
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import Transformer2DModel
from transformers import get_cosine_schedule_with_warmup
import lightning.pytorch as pl
from tqdm import tqdm

text_encoder_path = "tirthadagr8/T2I_Transformer"
text_tokenizer_path = "tirthadagr8/T2I_Transformer"
vae_model = "tirthadagr8/T2I_Transformer"
scheduler_path = "tirthadagr8/T2I_Transformer"
transformer_path = "tirthadagr8/T2I_Transformer" # Path to your saved transformer
model_max_length = 128
img_size = (64, 64)

import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# feature_extractor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# Load captions
df = pd.read_csv("/kaggle/working/generated_image_ds/generated_image_dataset/results.csv", 
                 sep="|", names=["image_name", "comment_number", "comment"])
df = df.dropna(subset=['comment'])                 # Drop NaNs
df = df[df['comment'].str.strip() != ""]           # Drop empty strings
df = df.reset_index(drop=True)                    # Reset index

# Clean up whitespace
df["comment"] = df["comment"].str.strip()

# Group captions by image
caption_dict = defaultdict(list)
for _, row in df.iterrows():
    if _==0:
        continue
    caption_dict[row["image_name"]].append(row["comment"])

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms as T

class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_dict, transform=None):
        self.image_dir = image_dir
        self.synthetic_dir = "/kaggle/working/generated_image_ds/generated_image_dataset"
        self.caption_dict = caption_dict
        self.image_names = list(caption_dict.keys())
        self.transform = transform or self.default_transforms()

    def default_transforms(self):
        return T.Compose([
            T.Resize(img_size),  # Adjust based on your model
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        if "image_" in image_name:
            image_path = os.path.join(self.synthetic_dir, image_name)
        else:
            image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Skip bad images

        # Randomly pick one caption
        caption = random.choice(self.caption_dict[image_name])
        
        return {
            "image": image,
            "text": caption,
            "image_name": image_name
        }

def collate_fn(batch, tokenizer, max_length=77):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    # print('MaxLength:',max_length)
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    
    return {
        "image": images,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path,subfolder='tokenizer')
tokenizer.model_max_length=model_max_length
# Define dataset
image_dir = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
dataset = FlickrDataset(image_dir=image_dir, caption_dict=caption_dict)
dataloader = DataLoader(
    dataset,
    batch_size=8,
    # shuffle=True,
    num_workers=4,
    collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer),
    drop_last=True
)

# ------------------------ PYTORCH LIGHTNING MODEL ------------------------

class FlowDiffusionModel(pl.LightningModule):
    def __init__(self, lr=1e-4, use_xformers=True):
        super().__init__()
        self.save_hyperparameters()
        
        # --- 1. Initialize models ---
        self.vae = AutoencoderKL.from_pretrained(vae_model, subfolder='vae')
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, subfolder='text_encoder')
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, subfolder='tokenizer')
        self.tokenizer.model_max_length = model_max_length
        
        # Although hdm uses a FlowMatch scheduler, the core training logic is what matters.
        # We can keep DDIMScheduler for inference later. The training logic itself is independent.
        self.scheduler = DDIMScheduler.from_pretrained(scheduler_path, subfolder='scheduler')
        
        self.transformer = Transformer2DModel.from_pretrained(transformer_path, subfolder='transformer')

        # --- 2. Freeze components that are not being trained ---
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # --- 3. Optimizations ---
        self.transformer.enable_gradient_checkpointing()
        if use_xformers:
            try:
                import xformers
                self.transformer.enable_xformers_memory_efficient_attention()
                print("xFormers memory efficient attention enabled.")
            except ImportError:
                print("xFormers not found; proceeding without it.")

        # --- 4. For Classifier-Free Guidance ---
        self.empty_tokens = self.tokenizer(
            [""], padding='max_length', truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        self.ema_loss = 0.0
        self.ema_decay = 0.995

    def training_step(self, batch, batch_idx):
        pixel_values = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # --- Encode images to latents ---
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # --- Get text embeddings (with support for CFG) ---
        with torch.no_grad():
            # 15% chance of unconditional training for CFG
            if random.random() < 0.15:
                uncond_input_ids = self.empty_tokens['input_ids'].expand(latents.shape[0], -1).to(self.device)
                uncond_attn_mask = self.empty_tokens['attention_mask'].expand(latents.shape[0], -1).to(self.device)
                text_embeddings = self.text_encoder(uncond_input_ids).last_hidden_state
                encoder_attention_mask = uncond_attn_mask
            else:
                text_embeddings = self.text_encoder(input_ids).last_hidden_state
                encoder_attention_mask = attention_mask
        
        # --- HDM Flow Matching Objective ---
        noise = torch.randn_like(latents)
        
        # Sample continuous time `t` from a sigmoid-transformed normal distribution
        # This is more stable than uniform sampling for flow matching
        t = torch.sigmoid(torch.randn(latents.shape[0], device=self.device))
        t = t.view(-1, 1, 1, 1) # Reshape for broadcasting

        # Create the noisy latent by interpolating between the clean latent and noise
        noisy_latents = t * noise + (1 - t) * latents
        
        # The target in flow matching is the "velocity", which is the difference vector
        target = noise - latents
        
        # --- Predict the velocity ---
        # The transformer needs a timestep, but for continuous time, we pass `t`
        # Note: Transformer2DModel expects a 1D tensor for timestep, so we reshape and scale
        model_pred = self.transformer(
            noisy_latents,
            timestep=(t.squeeze() * 1000).long(), # Scale to a pseudo-discrete step
            encoder_hidden_states=text_embeddings,
            encoder_attention_mask=encoder_attention_mask
        ).sample

        # --- Calculate Loss ---
        loss = F.mse_loss(model_pred, target)

        # --- Logging ---
        self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * loss.item()
        self.log_dict({
            'train_loss': loss.item(),
            'ema_loss': self.ema_loss,
            'lr': self.lr_schedulers().get_last_lr()[0]
        }, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        
        # The hdm code uses a complex scheduler, but a cosine schedule with warmup is a strong baseline
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.05) # 5% warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

from lightning.pytorch.strategies import DDPStrategy

if __name__ == '__main__':
    # --- IMPORTANT: Restart your Kaggle/Jupyter Kernel before running this! ---
    
    # --- Initialize Dataset and DataLoader ---
    tokenizer_for_collate = AutoTokenizer.from_pretrained(text_tokenizer_path, subfolder='tokenizer')
    tokenizer_for_collate.model_max_length = model_max_length

    image_dir = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
    dataset = FlickrDataset(image_dir=image_dir, caption_dict=caption_dict)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer_for_collate),
        drop_last=True,
        pin_memory=True
    )

    # --- Initialize Model ---
    model = FlowDiffusionModel(lr=1e-4)

    # --- Define the DDP strategy for multi-GPU environments ---
    # This ensures a clean start for each process, avoiding the CUDA error.
    strategy = "auto"
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(start_method='spawn')

    # --- Initialize Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy, # Use the spawn strategy if we have multiple GPUs
        precision="16-mixed",
        max_epochs=10,
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch:02d}-{step}-{ema_loss:.4f}",
                save_top_k=3,
                monitor="ema_loss",
                mode="min",
                every_n_train_steps=500
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )
    
    # --- Start Training ---
    # To resume, find the last .ckpt file in the 'checkpoints' directory and use:
    # trainer.fit(model, dataloader, ckpt_path="checkpoints/your_checkpoint.ckpt")
    trainer.fit(model, dataloader)
