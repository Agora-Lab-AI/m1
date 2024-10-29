import torch
from m1.model import ModelConfig, AudioConfig, MusicDiffusionTransformer, DiffusionScheduler, train_step, generate_audio
from loguru import logger

# Example usage
def main():
    logger.info("Setting up model configurations")
    
    # Configure logging
    logger.add("music_diffusion.log", rotation="500 MB")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize configurations
    model_config = ModelConfig(
        dim=512,
        depth=12,
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1
    )
    
    audio_config = AudioConfig(
        sample_rate=16000,
        n_mels=80,
        audio_length=1024,
        hop_length=256,
        win_length=1024,
        n_fft=1024
    )
    
    # Initialize model and scheduler
    model = MusicDiffusionTransformer(model_config, audio_config).to(device)
    scheduler = DiffusionScheduler(num_inference_steps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Example forward pass
    logger.info("Preparing example forward pass")
    batch_size = 4
    example_audio = torch.randn(batch_size, audio_config.audio_length).to(device)
    example_text = {
        'input_ids': torch.randint(0, 1000, (batch_size, 50)).to(device),
        'attention_mask': torch.ones(batch_size, 50).bool().to(device)
    }
    
    # Training step
    logger.info("Executing training step")
    loss = train_step(
        model,
        scheduler,
        optimizer,
        example_audio,
        example_text,
        device
    )
    logger.info(f"Training loss: {loss:.4f}")
    generation_text = {
        'input_ids': torch.randint(0, 1000, (1, 50)).to(device),
        'attention_mask': torch.ones(1, 50).bool().to(device)
    }
    
    # Generation example
    logger.info("Generating example audio")
    generated_audio = generate_audio(
        model,
        scheduler,
        generation_text,
        device,
        audio_config.audio_length
    )
    logger.info(f"Generated audio shape: {generated_audio.shape}")

if __name__ == "__main__":
    main()