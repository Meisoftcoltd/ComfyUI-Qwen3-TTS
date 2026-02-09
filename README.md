# ComfyUI-Qwen3-TTS

Custom nodes for [Qwen2.5-Audio / Qwen3-TTS](https://huggingface.co/Qwen/Qwen2.5-Audio-Instruct), a powerful multi-modal audio model capable of Text-to-Speech (TTS), Voice Cloning, and Voice Design.

## Features

*   **🎙️ Text-to-Speech (TTS):** Generate high-quality speech from text in multiple languages (English, Chinese, Spanish, etc.).
*   **👥 Voice Cloning:** Clone voices from a short reference audio clip (3-10s recommended).
*   **🎨 Voice Design:** Design custom voices by describing attributes like gender, age, pitch, speed, and emotion.
*   **🎓 Fine-Tuning:** Complete pipeline to fine-tune the model on your own voice dataset. Fine-tuning provides superior stability and tone matching compared to zero-shot cloning.
*   **📁 Modular Dataset Pipeline:** Automate dataset creation: Load raw audio -> Transcribe with Whisper -> Auto-Label emotions with Qwen2-Audio -> Export JSONL. Or use the all-in-one **Dataset Maker**.
*   **⚙️ Advanced Config:** Fixes for "Unsupported speakers" in fine-tuned models and detailed prompt control.
*   **📊 Audio Analysis:** Tools to compare generated audio against reference audio (Speaker Similarity & Mel Distance).
*   **⏳ Progress Reporting:** Real-time progress bars in the ComfyUI node title and HUD for long-running operations (transcription, labeling, training).

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
    *Note: Training features require `bitsandbytes` and `accelerate`. Dataset creation requires `openai-whisper` and `pydub` (and ffmpeg installed on your system).*

---

## 📚 Nodes Detailed Description

### 🎙️ Inference Nodes

#### **Qwen3Loader**
*   **Function:** Loads the base model (e.g., `Qwen/Qwen3-TTS-12Hz-1.7B-Base`) or specialized variants like `CustomVoice` or `VoiceDesign`.
*   **Inputs:** `repo_id`, `precision` (bf16 recommended), `attention` (sdpa/flash_attn).
*   **Outputs:** `QWEN3_MODEL` object.
*   **Details:** Handles downloading from HuggingFace/ModelScope and caching. If a checkpoint path is provided, it attempts to load it as a base (useful for debugging).

#### **Qwen3LoadFineTuned**
*   **Function:** Loads a fine-tuned model checkpoint for inference.
*   **Inputs:** `base_model` (required for architecture/tokenizer), `speaker_name`, `version`.
*   **Outputs:** `QWEN3_MODEL` object ready for generation.
*   **Details:** Crucial node for using your trained voices. It performs "Deep Injection" of the custom speaker configuration (`spk_id`) into the base model structure, preventing "Unsupported speaker" errors that occur if you just load weights.

#### **Qwen3CustomVoice**
*   **Function:** Generates speech using a specific trained speaker ID.
*   **Inputs:** `model`, `text`, `language`, `speaker` (dropdown of detected fine-tuned speakers).
*   **Outputs:** Audio waveform.
*   **Details:** Used for fine-tuned models. Allows selecting the specific `speaker_name` you trained.

#### **Qwen3VoiceDesign**
*   **Function:** Generates speech based on text and a set of descriptive attributes.
*   **Inputs:** `gender`, `pitch`, `speed`, `emotion`, `tone`, `age`, etc.
*   **Outputs:** Audio waveform.
*   **Details:** Uses the `VoiceDesign` variant of the model. You don't need to fill all fields; empty fields are ignored. Great for creating unique characters without reference audio.

#### **Qwen3VoiceClone**
*   **Function:** Zero-shot voice cloning from a reference audio.
*   **Inputs:** `ref_audio` (3-10s clip), `ref_text` (transcription of the audio), `text` (what you want it to say).
*   **Outputs:** Audio waveform.
*   **Details:** Uses the `Base` or `CustomVoice` variants. Requires the reference text for accurate prompt alignment.

### 📁 Dataset & Utilities

#### **Qwen3AudioToDataset (Dataset Maker)**
*   **Function:** All-in-one node to create a dataset from a folder of audio files.
*   **Inputs:** `audio_folder`, `model_size` (Whisper), `output_folder_name`, `min/max_duration`, `silence_threshold`.
*   **Outputs:** Path to the generated `dataset.jsonl`.
*   **Details:** Automatically loads, transcribes, slices, and formats the dataset for training.

#### **Qwen3TranscribeSingle**
*   **Function:** Transcribe a single audio clip using Whisper.
*   **Inputs:** `audio` (AUDIO input), `model_size`.
*   **Outputs:** `text` (STRING).
*   **Details:** Useful for preparing `ref_text` for Voice Cloning.

#### **Qwen3AudioCompare**
*   **Function:** Compares two audio clips (Reference vs Generated) to evaluate quality.
*   **Inputs:** `reference_audio`, `generated_audio`, `speaker_encoder_model` (Base model path).
*   **Outputs:** Text report with Speaker Similarity (Cosine) and Mel Spectrogram Distance (MSE).

#### **Utilities**
*   **Qwen3LoadAudioFromPath / Folder:** Load audio from absolute paths.
*   **Qwen3VideoToAudio:** Batch convert a folder of video files (mp4, mkv, etc.) to .wav audio files. Optimized for large datasets to prevent OOM errors.
*   **Qwen3SavePrompt / LoadPrompt:** Save generated voice prompts to .safetensors to reuse a cloned voice without re-computing.

### 📁 Modular Dataset Pipeline (Step-by-Step)

1.  **Qwen3LoadDatasetAudio:**
    *   Scans a local folder for `.wav` files. Returns a list of files.
2.  **Qwen3TranscribeWhisper:**
    *   Uses OpenAI Whisper to transcribe audio.
    *   Automatically slices long audio into chunks (e.g., < 15s) and trims silence.
    *   Outputs `DATASET_ITEMS` (audio path + text).
3.  **Qwen3AutoLabelEmotions:**
    *   Uses `Qwen2-Audio-Instruct` to "listen" to each clip.
    *   Generates tags like "Male voice, angry, shouting, fast speed".
    *   Enhances dataset quality by allowing the model to learn emotional conditioning.
4.  **Qwen3ExportJSONL:**
    *   Saves the processed items into a `dataset.jsonl` file.
    *   Format: `{"audio": "path/to/wav", "text": "transcription", "instruction": "tags"}`.

### 🎓 Training Nodes

#### **Qwen3DataPrep**
*   **Function:** Pre-tokenizes the audio and text data.
*   **Inputs:** `jsonl_path` (from Step 4).
*   **Outputs:** Path to `_codes.jsonl`.
*   **Details:** Converts audio to discrete codes using the `speech_tokenizer` and text to tokens. This step is heavy but ensures the training loop is fast and doesn't run OOM during tokenization. Handles OOM by falling back to sequential processing if batch processing fails.

#### **Qwen3FineTune**
*   **Function:** Performs full fine-tuning of the model.
*   **Inputs:** `train_jsonl` (the `_codes.jsonl` file), `init_model`, `epochs`, `batch_size`, `lr`, `target_loss`, `save_loss_threshold`.
*   **Outputs:** Path to the saved checkpoint.
*   **Details:**
    *   **Epochs:** Minimum 50 recommended for convergence on small datasets.
    *   **Target Loss:** Auto-stop mechanism. If loss drops below this value (e.g., 2.0), training stops and saves the model.
    *   **Save Loss Threshold:** Saves an intermediate checkpoint when loss drops below this value, without stopping training.
    *   **Learning Rate:** Defaults to `2e-6`. Higher values (e.g., `1e-5`) might cause noise/instability.
    *   **Mixed Precision:** Supports `bf16` (Ampere GPUs) and `fp32`.
    *   **Saving:** Saves `pytorch_model.bin` and `config.json` correctly mapped for immediate loading with `Qwen3LoadFineTuned`.

---

## 🧪 Workflow Examples

### 1. Dataset Creation Workflow
1.  **Load Audio:** Connect `Qwen3LoadDatasetAudio` pointing to your raw wavs folder.
2.  **Transcribe:** Connect to `Qwen3TranscribeWhisper`. Set `max_duration` to 15.0s.
3.  **Label:** Connect to `Qwen3AutoLabelEmotions`. This adds style tags.
4.  **Export:** Connect to `Qwen3ExportJSONL`.
5.  **Run:** This generates `dataset.jsonl`.

### 2. Training (Fine-Tuning) Workflow
1.  **Prep Data:** Connect the `dataset.jsonl` (from above) to `Qwen3DataPrep`.
    *   *Tip: Run this once. It creates `dataset_codes.jsonl`.*
2.  **Train:** Connect `Qwen3DataPrep` output to `Qwen3FineTune`.
    *   **Base Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
    *   **Speaker Name:** e.g., "Batman".
    *   **Epochs:** 100.
    *   **Batch Size:** 2 or 4 (depending on VRAM).
    *   **LR:** 2e-6.
3.  **Run:** Monitor the console. It will save checkpoints to `models/tts/finetuned_model/Batman/epoch_100`.

### 3. Inference with Fine-Tuned Voice
1.  **Load:** Use `Qwen3LoadFineTuned`.
    *   **Speaker Name:** Select "Batman".
    *   **Version:** Select "epoch_100".
2.  **Generate:** Connect to `Qwen3CustomVoice`.
    *   **Text:** "I am vengeance."
    *   **Speaker:** "Batman" (should appear in list).
3.  **Save:** Connect to `Qwen3SaveAudio`.

### 4. Voice Design Inference (Zero-Shot)
1.  **Load:** Use `Qwen3Loader` with `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
2.  **Generate:** Connect to `Qwen3VoiceDesign`.
    *   **Gender:** "Male"
    *   **Tone:** "Deep, raspy, intimidating"
    *   **Text:** "This city is mine."
3.  **Save:** Connect to `Qwen3SaveAudio`.
