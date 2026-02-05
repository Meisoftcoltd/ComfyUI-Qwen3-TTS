# ComfyUI-Qwen3-TTS

Custom nodes for [Qwen2.5-Audio / Qwen3-TTS](https://huggingface.co/Qwen/Qwen2.5-Audio-Instruct), a powerful multi-modal audio model capable of Text-to-Speech (TTS), Voice Cloning, and Voice Design.

## Features

*   **🎙️ Text-to-Speech (TTS):** Generate high-quality speech from text in multiple languages.
*   **👥 Voice Cloning:** Clone voices from a short reference audio clip (3-10s recommended).
*   **🎨 Voice Design:** Design custom voices by describing attributes like gender, age, pitch, speed, and emotion.
*   **🎓 Fine-Tuning & LoRA:** Complete pipeline to fine-tune the model or train lightweight LoRA adapters on your own voice dataset.
*   **📁 Modular Dataset Pipeline:** Automate dataset creation: Load raw audio -> Transcribe with Whisper -> Auto-Label emotions with Qwen2-Audio -> Export JSONL.
*   **⚙️ Advanced Config:** Fixes for "Unsupported speakers" in fine-tuned models and detailed prompt control.

## Installation

1.  **Install ComfyUI** (if you haven't already).
2.  Clone this repository into your `ComfyUI/custom_nodes/` folder:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/your-repo/ComfyUI-Qwen3-TTS.git
    cd ComfyUI-Qwen3-TTS
    ```
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Training features require `peft`, `bitsandbytes`, and `accelerate`. Dataset creation requires `openai-whisper` and `pydub`.*

## Nodes Overview

### 🎙️ Inference
*   **Qwen3Loader:** Loads the base model (e.g., `Qwen/Qwen3-TTS-12Hz-1.7B-Base`).
*   **Qwen3LoadFineTuned:** Loads a fine-tuned model (full checkpoint) and injects the custom speaker configuration required for inference.
*   **Qwen3ApplyLoRA:** Loads a LoRA adapter (`.safetensors` directory) and applies it to a base model.
*   **Qwen3VoiceDesign:** Generates speech based on text and a set of optional design parameters (Gender, Pitch, Emotion, etc.).
*   **Qwen3VoiceClone:** Generates speech by cloning a reference audio.

### 📁 Dataset Creation (Modular Pipeline)
1.  **Qwen3LoadDatasetAudio:** Scans a folder for `.wav` files.
2.  **Qwen3TranscribeWhisper:** Transcribes audio using Whisper, trims silence, and slices long files. (Requires `openai-whisper`).
3.  **Qwen3AutoLabelEmotions:** Uses `Qwen2-Audio-Instruct` to listen to the audio and generate descriptive labels (emotion, gender, tone) automatically.
4.  **Qwen3ExportJSONL:** Exports the final processed data to a `.jsonl` file ready for training.

### 🎓 Training
*   **Qwen3DataPrep:** Pre-processes the JSONL file into tokenized tensors (`input_ids`, `labels`) for efficient training.
*   **Qwen3TrainLoRA:** Trains a LoRA adapter on the pre-processed data. Supports `rank`, `alpha`, `epochs`, etc.
*   **Qwen3FineTune:** (Legacy) Full fine-tuning logic.

### 🛠️ Utils
*   **Qwen3SaveAudio:** Saves generated audio batches to a specific subfolder in the output directory.
*   **Qwen3LoadAudioFromPath:** Loads audio from an absolute path (useful for testing).

## Usage Tips
*   **Voice Design:** Use the individual fields (Gender, Pitch, etc.) to craft a specific voice. You don't need to fill them all.
*   **LoRA Training:** Always run **DataPrep** first to generate the `_codes.jsonl` file. This speeds up training significantly by pre-calculating tokens.
