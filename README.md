# [WIP] ComfyUI Qwen3-TTS

A ComfyUI custom node suite for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), supporting 1.7B and 0.6B models, Custom Voice, Voice Design, and Voice Cloning.

## Features

- **Auto-Download Models**: Automatically downloads models from HuggingFace (or ModelScope) if not present.
- **Full Qwen3-TTS Support**:
  - **Custom Voice**: Use 9 preset high-quality voices (Vivian, Ryan, etc.).
  - **Voice Design**: Create new voices using natural language descriptions.
  - **Voice Cloning**: Clone voices from a short reference audio clip.
- **Cross-Lingual Support**: Generate speech in Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.
- **Flexible Attention**: robust support for `flash_attention_2` with automatic fallback to `sdpa` (standard PyTorch 2.0 attention) if dependencies are missing.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes` folder:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/your-username/ComfyUI-Qwen3-TTS.git
    ```
2.  Install dependencies:
    ```bash
    cd ComfyUI-Qwen3-TTS
    pip install -r requirements.txt
    ```
    *Note: For GPU acceleration, ensure you have a CUDA-compatible PyTorch installed.*

## Usage

### 1. Load Model
Use the **Qwen3-TTS Loader** node.
- **repo_id**: Select the model you want to use.
  - `CustomVoice` models: For using preset speakers.
  - `VoiceDesign` models: For designing voices with text prompts.
  - `Base` models: For voice cloning.
- **attention**: Leave at `auto` for best performance (tries Flash Attention 2, falls back to SDPA).

### 2. Generate Audio

Connect the loaded model to one of the generator nodes:

#### **Custom Voice** (Requires `CustomVoice` Model)
- **speaker**: Choose one of the 9 presets (e.g., Vivian, Ryan).
- **text**: The text to speak.
- **language**: Target language (or Auto).
- **instruct**: (Optional) Add emotional instructions like "Happy" or "Whispering".

#### **Voice Design** (Requires `VoiceDesign` Model)
- **instruct**: Describe the voice you want, e.g., *"A deep, resonant male voice, narrator style, calm and professional."*
- **text**: The text to speak.

#### **Voice Clone** (Requires `Base` Model)
- **ref_audio**: Upload a reference audio file (1-10 seconds ideal).
- **ref_text**: The transcription of the reference audio (improves quality).
- **text**: The text for the cloned voice to speak.

### 3. Advanced: Prompt Caching
Use the **Qwen3-TTS Prompt Maker** node to pre-calculate the voice features from a reference audio. Connect the output `Qwen3_Prompt` to the **Voice Clone** node. This is faster if you are generating many sentences with the same cloned voice.

## Credits

Based on the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) library by QwenLM.
