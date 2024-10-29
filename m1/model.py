import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel
from einops import rearrange
from typing import Optional, Dict
from dataclasses import dataclass
from loguru import logger
import math
import torchaudio

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    audio_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    
@dataclass
class ModelConfig:
    dim: int = 512
    depth: int = 12
    heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    
class MultiQueryAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n d -> b 1 n d')
        v = rearrange(v, 'b n d -> b 1 n d')

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, None, :].expand(-1, h, -1, -1)
            dots.masked_fill_(~mask, float('-inf'))

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiQueryAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        time_emb = self.time_mlp(time_emb)
        x = self.norm1(x)
        x = x + self.attn(x)
        x = x + time_emb.unsqueeze(1)
        x = self.norm2(x)
        x = x + self.ffn(x)
        return x

class MusicDiffusionTransformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        audio_config: AudioConfig
    ) -> None:
        super().__init__()
        logger.info("Initializing MusicDiffusionTransformer")
        
        self.model_config = model_config
        self.audio_config = audio_config
        
        # Load pretrained T5 encoder
        logger.info("Loading pretrained T5 encoder")
        self.condition_encoder = T5EncoderModel.from_pretrained('t5-small')
        for param in self.condition_encoder.parameters():
            param.requires_grad = False
            
        # Audio preprocessing
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_config.sample_rate,
            n_fft=audio_config.n_fft,
            win_length=audio_config.win_length,
            hop_length=audio_config.hop_length,
            n_mels=audio_config.n_mels
        )
            
        # Audio encoding
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, model_config.dim // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(model_config.dim // 2, model_config.dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
        # Time embedding
        self.time_pos_emb = nn.Sequential(
            nn.Linear(model_config.dim, model_config.dim * 4),
            nn.GELU(),
            nn.Linear(model_config.dim * 4, model_config.dim),
        )
        
        # Transformer blocks
        logger.info(f"Creating {model_config.depth} transformer blocks")
        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                dim=model_config.dim,
                heads=model_config.heads,
                dim_head=model_config.dim_head,
                mlp_dim=model_config.mlp_dim,
                dropout=model_config.dropout
            ) for _ in range(model_config.depth)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(model_config.dim)
        self.to_audio = nn.Sequential(
            nn.Linear(model_config.dim, model_config.dim * 2),
            nn.GELU(),
            nn.Linear(model_config.dim * 2, 1)
        )

    def get_time_embeddings(
        self,
        timesteps: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        half_dim = self.model_config.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.time_pos_emb(embeddings)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition_text: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logger.debug(f"Forward pass with input shape: {x.shape}")
        
        # Encode conditioning text
        condition_embeddings = self.condition_encoder(
            condition_text['input_ids'],
            attention_mask=condition_text['attention_mask']
        ).last_hidden_state
        
        # Process audio input
        x = self.audio_encoder(x.unsqueeze(1))
        x = rearrange(x, 'b c t -> b t c')
        audio_length = x.shape[1]
        
        # Get time embeddings
        time_emb = self.get_time_embeddings(timesteps, x.shape[0])
        
        # Combine audio and condition embeddings
        x = torch.cat((x, condition_embeddings), dim=1)
        
        # Apply transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            logger.debug(f"Processing transformer block {idx}")
            x = block(x, time_emb)
            
        x = self.final_norm(x)
        x = self.to_audio(x)
        
        return x[:, :audio_length].squeeze(-1)

class DiffusionScheduler:
    def __init__(
        self,
        num_inference_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        logger.info(f"Initializing diffusion scheduler with {num_inference_steps} steps")
        self.betas = torch.linspace(beta_start, beta_end, num_inference_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[timesteps])
        
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        noisy_samples = sqrt_alpha_prod.unsqueeze(1) * original_samples + \
                       sqrt_one_minus_alpha_prod.unsqueeze(1) * noise
                       
        return noisy_samples

def train_step(
    model: MusicDiffusionTransformer,
    scheduler: DiffusionScheduler,
    optimizer: torch.optim.Optimizer,
    audio_batch: torch.Tensor,
    text_conditions: Dict[str, torch.Tensor],
    device: torch.device
) -> float:
    model.train()
    batch_size = audio_batch.shape[0]
    
    logger.debug(f"Training step with batch size: {batch_size}")
    
    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.num_inference_steps, (batch_size,), device=device)
    
    # Add noise to the audio
    noise = torch.randn_like(audio_batch)
    noisy_audio = scheduler.add_noise(audio_batch, noise, timesteps)
    
    # Predict the noise
    noise_pred = model(noisy_audio, timesteps, text_conditions)
    
    # Calculate loss
    loss = F.mse_loss(noise_pred, noise)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def generate_audio(
    model: MusicDiffusionTransformer,
    scheduler: DiffusionScheduler,
    condition_text: Dict[str, torch.Tensor],
    device: torch.device,
    audio_length: int = 1024
) -> torch.Tensor:
    logger.info("Generating audio...")
    model.eval()
    
    # Start from random noise
    audio = torch.randn(1, audio_length, device=device)
    
    # Gradually denoise
    for t in reversed(range(scheduler.num_inference_steps)):
        logger.debug(f"Denoising step {t}")
        timesteps = torch.tensor([t], device=device)
        
        # Predict and remove noise
        noise_pred = model(audio, timesteps, condition_text)
        alpha_t = scheduler.alphas[t]
        alpha_t_cumprod = scheduler.alphas_cumprod[t]
        beta_t = scheduler.betas[t]
        
        if t > 0:
            noise = torch.randn_like(audio)
        else:
            noise = torch.zeros_like(audio)
            
        audio = (1 / torch.sqrt(alpha_t)) * (
            audio - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * noise_pred
        ) + torch.sqrt(beta_t) * noise
        
    logger.info("Audio generation complete")
    return audio
