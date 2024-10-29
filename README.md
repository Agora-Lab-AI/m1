[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# M1: Music Generation via Diffusion Transformers 🎵🔬

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


M1 is a research project exploring large-scale music generation using diffusion transformers. This repository contains the implementation of our proposed architecture combining recent advances in diffusion models, transformer architectures, and music processing.

## 🔬 Research Overview

We propose a novel approach to music generation that combines:
- Diffusion-based generative modeling
- Multi-query attention mechanisms
- Hierarchical audio encoding
- Text-conditional generation
- Scalable training methodology

### Key Hypotheses

1. Diffusion transformers can capture long-range musical structure better than traditional autoregressive models
2. Multi-query attention mechanisms can improve training efficiency without sacrificing quality
3. Hierarchical audio encoding preserves both local and global musical features
4. Text conditioning enables semantic control over generation

## 🏗️ Architecture

```
                                              ┌─────────────────┐
                                              │  Time Encoding  │
                                              └────────┬────────┘
                                                      │
┌──────────────┐                              ┌──────▼─────────┐
│ Audio Input  ├──► mel spectrogram ──────────►               │
└──────────────┘                              │   Diffusion    │
                                              │  Transformer   │ ──► Generated Audio
┌──────────────┐    ┌─────────────┐          │     Block      │
│ Text Input   ├──► │ T5 Encoder  ├──────────►               │
└──────────────┘    └─────────────┘          └───────────────┘
```

### Implementation Details

```python
# Key architectural dimensions
MODEL_CONFIG = {
    'dim': 512,          # Base dimension
    'depth': 12,         # Number of transformer layers
    'heads': 8,          # Attention heads
    'dim_head': 64,      # Dimension per head
    'mlp_dim': 2048,     # FFN dimension
    'dropout': 0.1       # Dropout rate
}

# Audio processing parameters
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_mels': 80,
    'n_fft': 1024,
    'hop_length': 256
}
```

## 📊 Proposed Experiments

### Phase 1: Architecture Validation
- [ ] Baseline model training on synthetic data
- [ ] Ablation studies on attention mechanisms
- [ ] Time embedding comparison study
- [ ] Audio encoding architecture experiments

### Phase 2: Dataset Construction
We plan to build a research dataset from multiple sources:

1. **Initial Development Dataset**
   - 10k Creative Commons music samples
   - Focused on single-instrument recordings
   - Clear genre categorization

2. **Scaled Dataset** (Future Work)
   - Spotify API integration
   - SoundCloud API integration
   - Public domain music archives

### Phase 3: Training & Evaluation
Planned training configurations:
```yaml
initial_training:
  batch_size: 32
  gradient_accumulation: 4
  learning_rate: 1e-4
  warmup_steps: 1000
  max_steps: 100000
  
evaluation_metrics:
  - spectral_convergence
  - magnitude_error
  - musical_consistency
  - genre_accuracy
```

## 🛠️ Development Setup

```bash
# Clone repository
git clone https://github.com/Agora-Lab-AI/m1.git
cd m1-music

# Create environment
conda create -n m1 python=3.10
conda activate m1

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```


## Example

```python
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

```


## 📝 Project Structure

```
m1/
├── configs/               # Training configurations
├── m1/
│   ├── models/           # Model architectures
│   ├── diffusion/        # Diffusion scheduling
│   ├── data/             # Data loading/processing
│   └── training/         # Training loops
├── notebooks/            # Research notebooks
├── scripts/              # Training scripts
└── tests/                # Unit tests
```

## 🧪 Current Status

This is an active research project in early stages. Current focus:
- [ ] Implementing and testing base architecture
- [ ] Setting up data processing pipeline
- [ ] Designing initial experiments
- [ ] Building evaluation framework

## 📚 References

Key papers informing this work:
- "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, 2021)
- "Structured Denoising Diffusion Models" (Sohl-Dickstein et al., 2015)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)

## 🤝 Contributing

We welcome research collaborations! Areas where we're looking for contributions:
- Novel architectural improvements
- Efficient training methodologies
- Evaluation metrics
- Dataset curation tools

## 📬 Contact

For research collaboration inquiries:
- Submit an issue
- Start a discussion
- Email: research@m1music.ai

## ⚖️ License

This research code is released under the MIT License. 

## 🔍 Citation

If you use this code in your research, please cite:
```bibtex
@misc{m1music2024,
  title={M1: Experimental Music Generation via Diffusion Transformers},
  author={M1 Research Team},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Agora-Lab-AI/m1}}
}
```

## 🚧 Disclaimer

This is experimental research code:
- Architecture and training procedures may change significantly
- Not yet optimized for production use
- Results and capabilities are being actively researched
- Breaking changes should be expected

We're sharing this code to foster collaboration and advance the field of AI music generation research.
