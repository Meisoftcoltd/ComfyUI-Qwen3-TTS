# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-01-27

### Added
- **New nodes: Qwen3-TTS Save Prompt & Load Prompt** for persistent voice embeddings
  - Save voice clone prompts to `models/Qwen3-TTS/prompts/` as safetensors files
  - Load saved prompts directly without recomputing from audio
  - Enables building reusable voice libraries

## [1.5.0] - 2026-01-27

### Added
- `max_new_tokens` parameter for Voice Clone node to control generation length and prevent hangs
- `ref_audio_max_seconds` parameter to auto-trim long reference audio (default: 30s)
- Troubleshooting section in README for common issues
- **New node: Qwen3-TTS Audio Compare** for evaluating fine-tuned models (speaker similarity, mel spectrogram distance, speaking rate)
- Resume training from checkpoints with `resume_training` parameter
- Per-epoch checkpointing with automatic cleanup of intermediate checkpoints
- 8-bit AdamW optimizer support via bitsandbytes for reduced VRAM usage
- Gradient checkpointing option (~30-40% VRAM savings during training)
- Learning rate warmup scheduling with state persistence
- SHA256-based dataset caching to skip reprocessing unchanged datasets
- Configurable `batch_size` parameter in Data Prep node for VRAM control
- Support for 0.6B model fine-tuning (text_projection fix)
- UI progress status updates with loss tracking during training
- Optimizer and scheduler state saving for true training resume

### Changed
- Reduced default `max_new_tokens` from 8192 to 2048 to prevent generation hangs
- Reduced default learning rate from 2e-5 to 2e-6 for training stability
- Updated documentation with generation hang mitigation tips and new fine-tuning features

### Fixed
- Mitigated infinite generation loop issue when using long reference audio or generating long outputs
- Fixed dtype parameter handling in fine-tuning
- Fixed speaker embedding CPU conversion issue

### Contributors
- @rekuenkdr - Training enhancements, VRAM optimizations, Audio Compare node

## [1.4.0] - 2026-01-24

### Added
- Exposed finetuning seed parameter for reproducible training
- Resource cleanup for mixed workflows

### Changed
- Improved memory management when switching between inference and training

## [1.3.0] - 2026-01-24

### Changed
- Updated model download management with improved caching and error handling
- Enhanced README documentation

## [1.2.0] - 2026-01-24

### Added
- RNG seed support for reproducible audio generation in inference nodes

## [1.1.0] - 2026-01-23

### Added
- Fine-tuning support with dataset handling (`dataset.py`)
- Simple finetuning example workflow
- MPS (Apple Silicon) device support

### Changed
- Use ComfyUI `model_management` for device detection
- Updated transformers dependency warning
- Updated requirements with safetensors

### Fixed
- Project name in pyproject.toml to match repository naming convention

## [1.0.0] - 2026-01-22

### Added
- Initial release
- Qwen3-TTS model loader node
- Text-to-speech generation node
- Custom voice workflow configuration
- Flash attention support with automatic fallback
