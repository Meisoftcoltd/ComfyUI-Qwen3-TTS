import os
import json
import shutil
import torch
import contextlib
import io
import logging
import hashlib
import math
from datetime import datetime, timezone
import soundfile as sf
import numpy as np
import folder_paths
import comfy.model_management as mm
from server import PromptServer
import platform
from tqdm import tqdm

def fix_wsl_path(path):
    """Helper to convert Windows paths to WSL paths if running on Linux."""
    if not path or not isinstance(path, str):
        return path

    # Clean up quotes and whitespace first
    path = path.strip().strip('"').strip("'")

    if platform.system() == "Linux":
        # Check for Windows style path like "Z:\" or "C:\" or "Z:/"
        # We look for [Letter]:
        if len(path) >= 2 and path[1] == ':' and path[0].isalpha():
            drive = path[0].lower()
            rest = path[2:].replace('\\', '/')
            if not rest.startswith('/'):
                 rest = '/' + rest
            new_path = f"/mnt/{drive}{rest}"
            print(f"[Qwen3-TTS DEBUG] Detected Windows path in WSL: '{path}' -> converting to '{new_path}'")
            return new_path
    return path

from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from .dataset import TTSDataset
try:
    import whisper
    from pydub import AudioSegment, silence
    HAS_WHISPER_PYDUB = True
except ImportError:
    HAS_WHISPER_PYDUB = False
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
from accelerate import Accelerator
from torch.optim import AdamW
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, get_linear_schedule_with_warmup, AutoProcessor, Qwen2AudioForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers.utils import cached_file
from safetensors.torch import save_file, load_file

# Register Qwen3-TTS models folder with ComfyUI
# We support both 'models/tts' (new default) and 'models/Qwen3-TTS' (legacy)
TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "tts")
OLD_QWEN3_TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS")

os.makedirs(TTS_MODELS_DIR, exist_ok=True)

# Primary directory is now models/tts
QWEN3_TTS_MODELS_DIR = TTS_MODELS_DIR

folder_paths.add_model_folder_path("tts", TTS_MODELS_DIR)
folder_paths.add_model_folder_path("Qwen3-TTS", OLD_QWEN3_TTS_MODELS_DIR)

# Register Qwen3-TTS prompts folder for voice embeddings
# Now stored in models/tts/prompts
QWEN3_TTS_PROMPTS_DIR = os.path.join(TTS_MODELS_DIR, "prompts")
os.makedirs(QWEN3_TTS_PROMPTS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS-Prompts", QWEN3_TTS_PROMPTS_DIR)

# Model repo mappings
QWEN3_TTS_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "Qwen3-TTS-12Hz-0.6B-Base",
}

# Tokenizer repo mapping
QWEN3_TTS_TOKENIZERS = {
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": "Qwen3-TTS-Tokenizer-12Hz",
}

def get_finetuned_speakers() -> list:
    """Scan configured output directories for fine-tuned speakers."""
    speakers = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee"
    ]

    # Paths to scan
    scan_paths = [
        os.path.join(folder_paths.models_dir, "tts", "finetuned_model"), # Default output
        os.path.join(folder_paths.models_dir, "Qwen3-TTS", "finetuned_model"), # Legacy
        os.path.abspath("models/tts/finetuned_model"), # Relative fallback
    ]

    found_speakers = []

    for base_path in scan_paths:
        if os.path.exists(base_path):
            # 1. New Structure: Check immediate subfolders as speaker names
            # Structure: finetuned_model/{speaker_name}/epoch_N
            for item in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, item)):
                    # Add directory name as speaker (if not already known)
                    if item not in speakers and item not in found_speakers:
                        found_speakers.append(item)

            # 2. Deep Scan: Check config.json for spk_id (Legacy & Robustness)
            for root, dirs, files in os.walk(base_path):
                if "config.json" in files:
                    try:
                        with open(os.path.join(root, "config.json"), 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                            if "talker_config" in cfg and "spk_id" in cfg["talker_config"]:
                                spk_ids = cfg["talker_config"]["spk_id"]
                                for name in spk_ids.keys():
                                    if name not in speakers and name not in found_speakers:
                                        found_speakers.append(name)
                    except:
                        pass

    return sorted(found_speakers) + speakers

def get_local_model_path(repo_id: str) -> str:
    """Get the local path for a model/tokenizer in ComfyUI's models folder."""
    folder_name = QWEN3_TTS_MODELS.get(repo_id) or QWEN3_TTS_TOKENIZERS.get(repo_id) or repo_id.replace("/", "_")

    # Check new location first
    new_path = os.path.join(TTS_MODELS_DIR, folder_name)
    if os.path.exists(new_path):
        return new_path

    # Check old location
    old_path = os.path.join(OLD_QWEN3_TTS_MODELS_DIR, folder_name)
    if os.path.exists(old_path):
        return old_path

    # Default to new location for download
    return new_path

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"

def count_jsonl_lines(file_path: str) -> int:
    """Count lines in a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def load_cache_metadata(meta_path: str) -> dict | None:
    """Load cache metadata, return None if invalid."""
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata.get('version') != 1:
            return None
        return metadata
    except (json.JSONDecodeError, IOError):
        return None

def save_cache_metadata(meta_path: str, metadata: dict) -> None:
    """Save cache metadata to file."""
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def migrate_cached_model(repo_id: str, target_path: str) -> bool:
    """Check for model in HuggingFace/ModelScope cache and migrate to ComfyUI folder."""
    if os.path.exists(target_path) and os.listdir(target_path):
        return True  # Already exists in target
    
    # Check HuggingFace cache
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hf_model_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    if os.path.exists(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                source = os.path.join(snapshots_dir, snapshots[0])
                print(f"Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True
    
    # Check ModelScope cache
    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True
    
    return False

def download_model_to_comfyui(repo_id: str, source: str) -> str:
    """Download a model directly to ComfyUI's models folder."""
    target_path = get_local_model_path(repo_id)
    
    # First check if we can migrate from cache
    if migrate_cached_model(repo_id, target_path):
        print(f"Model available at: {target_path}")
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    
    if source == "ModelScope":
        from modelscope import snapshot_download
        print(f"Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} from HuggingFace to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    
    return target_path

def get_available_models() -> list:
    """Get list of available models (downloaded + all options + local folders)."""
    available = []

    # 1. Known Repo IDs
    for repo_id, folder_name in QWEN3_TTS_MODELS.items():
        # Always add the raw repo_id to ensure validation passes for existing workflows
        available.append(repo_id)

        local_path = get_local_model_path(repo_id)
        if os.path.exists(local_path) and os.listdir(local_path):
            available.append(f"✓ {repo_id}")

    # 2. Other Local Folders (not matching known repos)
    # Scan both new and old directories
    for search_dir in [TTS_MODELS_DIR, OLD_QWEN3_TTS_MODELS_DIR]:
        if os.path.exists(search_dir):
            # 2a. Top-level folders
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if os.path.isdir(item_path):
                    # Check if this folder is already covered by the mapping
                    is_known = False
                    for repo_id, folder_name in QWEN3_TTS_MODELS.items():
                        if item == folder_name:
                            is_known = True
                            break
                    if not is_known and item != "prompts":
                        name = f"Local: {item}"
                        if name not in available:
                            available.append(name)

    return sorted(list(set(available)))

# Helper to convert audio to ComfyUI format
def convert_audio(wav, sr):
    # wav is (channels, samples) or just (samples)
    # ComfyUI audio format: {"waveform": tensor(1, channels, samples), "sample_rate": int}
    # But usually audio nodes expect (batch, samples, channels) or (batch, channels, samples)?
    # Standard LoadAudio in ComfyUI returns:
    # "audio": {"waveform": audio_tensor, "sample_rate": sample_rate}
    # audio_tensor is [batch, channels, samples] (usually batch=1)
    
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) # (1, samples) (channels=1)
    
    # Qwen outputs numpy float32 usually.
    # Check if stereo/mono. Qwen3-TTS is mono usually?
    # Ensure shape is [1, channels, samples] for ComfyUI

    # Robust shape detection:
    # Audio usually has Samples >> Channels (e.g. 24000 samples vs 1 or 2 channels).
    # If dim0 > dim1, it's likely (Samples, Channels).
    # We verify this by checking if dim1 is reasonably small (< 2048) to be channels.
    if wav.shape[0] > wav.shape[1] and wav.shape[1] < 2048:
        # Detected (Samples, Channels) format -> Transpose to (Channels, Samples)
        wav = wav.transpose(0, 1)
        
    # If it's just (samples,), we made it (1, samples). 
    # ComfyUI often expects [Batch, Channels, Samples]. 
    # Let's wrap in batch.
    wav = wav.unsqueeze(0) # (1, channels, samples)
    
    return {"waveform": wav, "sample_rate": sr}

def load_audio_input(audio_input):
    # audio_input is {"waveform": tensor, "sample_rate": int}
    # waveform can be various formats: [B, C, T], [C, T], [T]
    # We need (samples,) or (channels, samples) numpy for Qwen
    # Qwen accepts numpy array.

    if audio_input is None:
        return None

    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]

    print(f"load_audio_input received: waveform.shape={waveform.shape}, waveform.dim={waveform.dim()}")

    # Handle different waveform formats
    if waveform.dim() == 0:
        # 0-dimensional tensor - this causes "Len() of unsized object"
        raise ValueError(f"Waveform is 0-dimensional (scalar tensor). This causes 'Len() of unsized object' error.")

    elif waveform.dim() == 1:
        # Already [samples], just return as-is
        wav = waveform
        print(f"Format [samples], returning as-is")

    elif waveform.dim() == 2:
        # Could be [C, T] or [T, C]
        if waveform.shape[0] < waveform.shape[1]:
            # Likely [C, T] - channels first
            if waveform.shape[0] > 1:
                wav = torch.mean(waveform, dim=0)  # Mix to mono
            else:
                wav = waveform[0]  # Take single channel
            print(f"Format [C, T], mixed to mono: {wav.shape}")
        else:
            # Likely [T, C] - samples first (shouldn't happen normally)
            if waveform.shape[1] > 1:
                wav = torch.mean(waveform, dim=1)  # Mix to mono
            else:
                wav = waveform[:, 0]
            wav = wav.unsqueeze(0)  # Add channel dimension back
            print(f"Format [T, C], converted: {wav.shape}")

    elif waveform.dim() == 3:
        # Expected [B, C, T] - take first batch
        wav = waveform[0]  # (channels, samples)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0)  # Mix to mono
        else:
            wav = wav.squeeze(0)  # (samples,)
        print(f"Format [B, C, T], took first batch and mixed: {wav.shape}")

    else:
        raise ValueError(f"Unexpected waveform dimension: {waveform.dim()} (shape: {waveform.shape})")

    # Ensure we have a 1D tensor
    if wav.dim() > 1:
        wav = wav.squeeze()
        print(f"Squeezed to 1D: {wav.shape}")

    return (wav.numpy(), sr)

class Qwen3Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (get_available_models(), {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to local model or checkpoint. If checkpoint (no speech_tokenizer/), base model loads from repo_id first."}),
            }
        }

    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-TTS/Loader"

    def load_model(self, repo_id, source, precision, attention, local_model_path=""):
        # Handle "✓ " prefix or "Local: " prefix
        model_name = repo_id
        is_local_selection = False

        if model_name.startswith("✓ "):
            model_name = model_name[2:]
        elif model_name.startswith("Local: "):
            model_name = model_name[7:]
            is_local_selection = True

        # Clean model name for output (remove path chars if any, though repo_id usually fine)
        clean_model_name = model_name.replace("/", "_").replace("\\", "_")

        device = mm.get_torch_device()

        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        checkpoint_path = None
        local_path_stripped = local_model_path.strip() if local_model_path else ""

        # Robust Path Logic:
        # 1. Determine the primary target path (either repo download or local path)
        target_path = None
        if local_path_stripped:
            target_path = local_path_stripped
        elif is_local_selection:
            # Check both new and old dirs for local model
            p1 = os.path.join(QWEN3_TTS_MODELS_DIR, model_name)
            p2 = os.path.join(OLD_QWEN3_TTS_MODELS_DIR, model_name)
            if os.path.exists(p1): target_path = p1
            elif os.path.exists(p2): target_path = p2
            else: target_path = p1 # Default fallback
        else:
             # Repo ID logic
             local_path = get_local_model_path(model_name)
             if os.path.exists(local_path) and os.listdir(local_path):
                 target_path = local_path
             else:
                 print(f"Model not found locally. Downloading {model_name}...")
                 target_path = download_model_to_comfyui(model_name, source)

        # 2. Check if target_path is a full model (has speech_tokenizer) or just a checkpoint
        speech_tokenizer_path = os.path.join(target_path, "speech_tokenizer")

        if os.path.exists(speech_tokenizer_path) and os.path.exists(os.path.join(speech_tokenizer_path, "config.json")):
             # It's a full model
             model_path = target_path
             checkpoint_path = None
             print(f"Loading full model from: {model_path}")
        else:
             # It's likely a fine-tune checkpoint (missing tokenizer), treat as checkpoint
             print(f"Target path '{target_path}' appears to be a checkpoint (missing speech_tokenizer).")
             checkpoint_path = target_path

             # Fallback to a default base model (1.7B Base)
             default_base = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
             local_base = get_local_model_path(default_base)

             if os.path.exists(local_base) and os.path.exists(os.path.join(local_base, "speech_tokenizer")):
                 model_path = local_base
             else:
                 print(f"Downloading default base model {default_base} for checkpoint overlay...")
                 model_path = download_model_to_comfyui(default_base, source)

             print(f"Loading Base Model: {model_path}")
             print(f"Will apply checkpoint: {checkpoint_path}")

        print(f"Loading Qwen3-TTS model on {device} as {dtype}")

        attn_impl = "sdpa"
        if attention != "auto":
            attn_impl = attention
        else:
            try:
                import flash_attn
                import importlib.metadata
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"

        print(f"Using attention implementation: {attn_impl}")

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )

        if checkpoint_path:
            ckpt_weights = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(ckpt_weights):
                state_dict = torch.load(ckpt_weights, map_location="cpu", weights_only=True)
                model.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded checkpoint weights from {ckpt_weights}")
            else:
                raise ValueError(f"Checkpoint weights not found: {ckpt_weights}")

        # FORCE SPEAKER MAPPING FIX - Deep Injection
        try:
            cfg_file = os.path.join(checkpoint_path, "config.json") if checkpoint_path else os.path.join(model_path, "config.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
                
                if "talker_config" in cfg_data and "spk_id" in cfg_data["talker_config"]:
                    new_spk_id = cfg_data["talker_config"]["spk_id"]
                    new_spk_dialect = cfg_data["talker_config"].get("spk_is_dialect", {})
                    
                    # --- FIX: Extract Correct Speaker Name for Output ---
                    if isinstance(new_spk_id, dict) and len(new_spk_id) > 0:
                        potential_name = list(new_spk_id.keys())[0]
                        if potential_name and potential_name.strip():
                            clean_model_name = potential_name.strip()
                            # CRITICAL: This was missing! Actually update the output variable if you want it to work.
                            # Although Qwen3Loader returns (model, clean_model_name), clean_model_name is a local variable.
                            # We just need to ensure clean_model_name holds the correct string.
                            print(f"[Qwen3-TTS] Updated model_name from config: {clean_model_name}")
                    # ----------------------------------------------------

                    # Target List: where spk_id might be hidden
                    configs_to_update = []
                    
                    # 1. Main model wrapper config
                    if hasattr(model, "config"): configs_to_update.append(model.config)
                    # 2. Internal model config
                    if hasattr(model, "model") and hasattr(model.model, "config"): configs_to_update.append(model.model.config)
                    
                    found_any = False
                    for root_cfg in configs_to_update:
                        # Try to find talker_config within these
                        t_cfg = getattr(root_cfg, "talker_config", None)
                        if t_cfg is not None:
                            for attr, val in [("spk_id", new_spk_id), ("spk_is_dialect", new_spk_dialect)]:
                                if not hasattr(t_cfg, attr) or getattr(t_cfg, attr) is None:
                                    setattr(t_cfg, attr, {})
                                cur_val = getattr(t_cfg, attr)
                                if isinstance(cur_val, dict):
                                    cur_val.update(val)
                                    found_any = True
                    
                    # 3. Direct access to the Talker's internal config (Most important)
                    if hasattr(model, "model") and hasattr(model.model, "talker") and hasattr(model.model.talker, "config"):
                        st_cfg = model.model.talker.config
                        for attr, val in [("spk_id", new_spk_id), ("spk_is_dialect", new_spk_dialect)]:
                            if not hasattr(st_cfg, attr) or getattr(st_cfg, attr) is None:
                                setattr(st_cfg, attr, {})
                            cur_val = getattr(st_cfg, attr)
                            if isinstance(cur_val, dict):
                                cur_val.update(val)
                                found_any = True
                    
                    if found_any:
                        print(f"DEBUG: Successfully injected custom speaker mapping: {new_spk_id}", flush=True)
                    else:
                        print("DEBUG: Failed to find an appropriate config object to inject mapping into.", flush=True)

                # Inject tts_model_type if present in checkpoint config
                if "tts_model_type" in cfg_data:
                    new_tts_model_type = cfg_data["tts_model_type"]

                    # Inject into config objects
                    for root_cfg in configs_to_update:
                        if hasattr(root_cfg, "tts_model_type"):
                            setattr(root_cfg, "tts_model_type", new_tts_model_type)

                    # CRITICAL: Also update the direct attribute on the inner model
                    # This is what generate_custom_voice() actually checks
                    if hasattr(model, "model") and hasattr(model.model, "tts_model_type"):
                        model.model.tts_model_type = new_tts_model_type

                    print(f"DEBUG: Injected tts_model_type = {new_tts_model_type}", flush=True)
        except Exception as e:
            print(f"DEBUG: Error during deep speaker injection: {e}", flush=True)
        
        return (model,)


class Qwen3LoadFineTuned:
    @classmethod
    def INPUT_TYPES(s):
        # Scan finetuned_model directory
        scan_paths = [
            os.path.join(folder_paths.models_dir, "tts", "finetuned_model"),
            os.path.join(folder_paths.models_dir, "Qwen3-TTS", "finetuned_model"),
        ]

        speakers = []
        versions = []

        for base_path in scan_paths:
            if os.path.exists(base_path):
                # Scan speakers (directories)
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isdir(item_path):
                        speakers.append(item)
                        # Scan versions (subdirectories)
                        for v in os.listdir(item_path):
                            if os.path.isdir(os.path.join(item_path, v)):
                                versions.append(v)

        speakers = sorted(list(set(speakers)))
        versions = sorted(list(set(versions)))

        if not speakers:
            speakers = ["No fine-tuned speakers found"]
        if not versions:
            versions = ["No versions found"]

        # Base models list
        base_models = [k for k in QWEN3_TTS_MODELS.keys() if "Base" in k]
        # Add local base models if available
        available = get_available_models()
        for m in available:
            if "Base" in m and m not in base_models:
                base_models.append(m)

        return {
            "required": {
                "base_model": (base_models, {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}),
                "speaker_name": (speakers,),
                "version": (versions,),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN3_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_name")
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-TTS/Loader"

    def load_model(self, base_model, speaker_name, version, precision, attention):
        if speaker_name == "No fine-tuned speakers found":
            raise ValueError("No fine-tuned models found. Please train a model first or check models/tts/finetuned_model.")

        # Resolve paths
        scan_paths = [
            os.path.join(folder_paths.models_dir, "tts", "finetuned_model"),
            os.path.join(folder_paths.models_dir, "Qwen3-TTS", "finetuned_model"),
        ]

        checkpoint_path = None
        for base_path in scan_paths:
            candidate = os.path.join(base_path, speaker_name, version)
            if os.path.exists(candidate) and os.path.isdir(candidate):
                checkpoint_path = candidate
                break

        if not checkpoint_path:
             raise FileNotFoundError(f"Checkpoint not found for speaker '{speaker_name}' version '{version}'. Checked paths: {[os.path.join(p, speaker_name, version) for p in scan_paths]}")

        print(f"[Qwen3-TTS] Loading Fine-Tuned Checkpoint: {checkpoint_path}")
        print(f"[Qwen3-TTS] Using Base Model: {base_model}")

        # Resolve Base Model Path
        # CRITICAL: Ensure we rely on a valid BASE model with tokenizer, NOT the checkpoint folder
        clean_base = base_model.replace("✓ ", "").replace("Local: ", "")

        # 1. Is it a known repo ID?
        if clean_base in QWEN3_TTS_MODELS or clean_base in QWEN3_TTS_MODELS.values():
             local_path = get_local_model_path(clean_base)
             if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                 base_model_path = local_path
             else:
                 print(f"[Qwen3-TTS] Downloading missing base model: {clean_base}")
                 base_model_path = download_model_to_comfyui(clean_base, "HuggingFace")
        else:
             # 2. Is it a local folder?
             # Check if it has "speech_tokenizer" to confirm it's a valid base
             potential_paths = [
                 os.path.join(TTS_MODELS_DIR, clean_base),
                 os.path.join(OLD_QWEN3_TTS_MODELS_DIR, clean_base),
                 get_local_model_path(clean_base)
             ]

             base_model_path = None
             for p in potential_paths:
                 if os.path.exists(p) and os.path.exists(os.path.join(p, "speech_tokenizer")):
                     base_model_path = p
                     break

             if not base_model_path:
                 # Fallback: Default to 1.7B Base if the user selected a checkpoint by mistake as base
                 print(f"[Qwen3-TTS] Warning: '{clean_base}' does not appear to be a valid base model (missing speech_tokenizer). Falling back to default Base.")
                 default_repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                 local_default = get_local_model_path(default_repo)
                 if os.path.exists(local_default):
                     base_model_path = local_default
                 else:
                     base_model_path = download_model_to_comfyui(default_repo, "HuggingFace")

        print(f"[Qwen3-TTS] Final Base Model Path: {base_model_path}")

        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model path not found: {base_model_path}")

        device = mm.get_torch_device()
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        attn_impl = "sdpa"
        if attention != "auto":
            attn_impl = attention
        else:
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"

        print(f"Loading Base Qwen3-TTS model on {device} as {dtype}")

        # Load Base Model (Initializes architecture + tokenizer correctly)
        model = Qwen3TTSModel.from_pretrained(
            base_model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )

        # Load Weights from Checkpoint
        ckpt_weights = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(ckpt_weights):
            print(f"Loading fine-tuned weights from {ckpt_weights}...")
            state_dict = torch.load(ckpt_weights, map_location="cpu", weights_only=True)
            # Loose strictness to allow for potential mismatch in unused layers if any,
            # though usually finetune matches base structure.
            keys = model.model.load_state_dict(state_dict, strict=False)
            if keys.missing_keys:
                print(f"DEBUG: Missing keys (expected for PEFT or partial loads, check if critical): {keys.missing_keys[:5]}...")
            if keys.unexpected_keys:
                print(f"DEBUG: Unexpected keys: {keys.unexpected_keys[:5]}...")
        else:
             raise ValueError(f"pytorch_model.bin not found in checkpoint: {checkpoint_path}")

        # INJECTION LOGIC (Speaker Config)
        try:
            cfg_file = os.path.join(checkpoint_path, "config.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)

                configs_to_update = []
                if hasattr(model, "config"): configs_to_update.append(model.config)
                if hasattr(model, "model") and hasattr(model.model, "config"): configs_to_update.append(model.model.config)

                if "talker_config" in cfg_data and "spk_id" in cfg_data["talker_config"]:
                    new_spk_id = cfg_data["talker_config"]["spk_id"]
                    new_spk_dialect = cfg_data["talker_config"].get("spk_is_dialect", {})

                    found_any = False
                    for root_cfg in configs_to_update:
                        t_cfg = getattr(root_cfg, "talker_config", None)
                        if t_cfg is not None:
                            if not hasattr(t_cfg, "spk_id") or t_cfg.spk_id is None:
                                t_cfg.spk_id = {}
                            if isinstance(t_cfg.spk_id, dict):
                                t_cfg.spk_id.update(new_spk_id)

                            if not hasattr(t_cfg, "spk_is_dialect") or t_cfg.spk_is_dialect is None:
                                t_cfg.spk_is_dialect = {}
                            if isinstance(t_cfg.spk_is_dialect, dict):
                                t_cfg.spk_is_dialect.update(new_spk_dialect)
                            found_any = True

                    # Update internal talker config
                    if hasattr(model, "model") and hasattr(model.model, "talker") and hasattr(model.model.talker, "config"):
                        st_cfg = model.model.talker.config
                        if not hasattr(st_cfg, "spk_id") or st_cfg.spk_id is None:
                            st_cfg.spk_id = {}
                        if isinstance(st_cfg.spk_id, dict):
                            st_cfg.spk_id.update(new_spk_id)
                        found_any = True

                    if found_any:
                         print(f"DEBUG: Successfully injected custom speaker mapping from {checkpoint_path}: {list(new_spk_id.keys())}")

                if "tts_model_type" in cfg_data:
                    new_tts_model_type = cfg_data["tts_model_type"]
                    for root_cfg in configs_to_update:
                        if hasattr(root_cfg, "tts_model_type"):
                            setattr(root_cfg, "tts_model_type", new_tts_model_type)

                    if hasattr(model, "model") and hasattr(model.model, "tts_model_type"):
                         model.model.tts_model_type = new_tts_model_type

                    print(f"DEBUG: Injected tts_model_type = {new_tts_model_type}")

        except Exception as e:
            print(f"DEBUG: Error during deep speaker injection: {e}")

        return (model, speaker_name)


class Qwen3CustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "speaker": (get_finetuned_speakers(), {"default": "Vivian"}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                "custom_speaker_name": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Penalty for repetition. Increase (e.g., 1.1-1.2) to prevent infinite loops/stuttering."}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, language, speaker, seed, instruct="", custom_speaker_name="", max_new_tokens=8192, top_p=0.8, temperature=0.7, repetition_penalty=1.1):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS/Inference"

    def generate(self, model, text, language, speaker, seed, instruct="", custom_speaker_name="", max_new_tokens=8192, top_p=0.8, temperature=0.7, repetition_penalty=1.1):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        lang = language if language != "Auto" else None
        inst = instruct if instruct.strip() != "" else None
        
        target_speaker = speaker
        if custom_speaker_name and custom_speaker_name.strip() != "":
            target_speaker = custom_speaker_name.strip()
            print(f"Using custom speaker: {target_speaker}")
        
        # Manual lookup and case-matching to bypass library validation failures
        try:
            configs_to_check = []
            if hasattr(model, "config"): configs_to_check.append(model.config)
            if hasattr(model, "model") and hasattr(model.model, "config"): configs_to_check.append(model.model.config)
            
            for root_cfg in configs_to_check:
                t_cfg = getattr(root_cfg, "talker_config", None)
                if t_cfg:
                    spk_map = getattr(t_cfg, "spk_id", None)
                    if isinstance(spk_map, dict):
                        # Case-insensitive match
                        match = next((s for s in spk_map if s.lower() == target_speaker.lower()), None)
                        if match:
                            print(f"DEBUG: Found case-matched speaker: '{match}' (original: '{target_speaker}')", flush=True)
                            target_speaker = match # Use the name the model expects
                            break
        except Exception as e:
            print(f"DEBUG: Speaker case-matching failed: {e}", flush=True)

        gen_kwargs = {
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }

        try:
            try:
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=target_speaker,
                    instruct=inst,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs
                )
            except TypeError:
                print("Warning: Model generation function does not support extra parameters (top_p, temperature, repetition_penalty). Ignoring them.")
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=target_speaker,
                    instruct=inst,
                    max_new_tokens=max_new_tokens
                )
        except ValueError as e:
            # Catch model type mismatch errors from qwen-tts
            msg = str(e)
            if "does not support generate_custom_voice" in msg:
                raise ValueError("Model Type Error: You are trying to use 'Custom Voice' with an incompatible model. Please load a 'CustomVoice' model (e.g. Qwen3-TTS-12Hz-1.7B-CustomVoice).") from e
            raise e
            
        return (convert_audio(wavs[0], sr),)


class Qwen3VoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "gender": ("STRING", {"default": "Male", "multiline": False}),
                "pitch": ("STRING", {"default": "Deep and resonant with subtle downward inflections suggesting gravity", "multiline": True}),
                "speed": ("STRING", {"default": "Deliberately slow with extended pauses between sentences", "multiline": True}),
                "volume": ("STRING", {"default": "Moderate to soft, creating an intimate atmosphere", "multiline": True}),
                "age": ("STRING", {"default": "Middle-aged to older adult", "multiline": False}),
                "clarity": ("STRING", {"default": "Crystal clear enunciation with careful articulation", "multiline": True}),
                "fluency": ("STRING", {"default": "Smooth and controlled with intentional dramatic pauses", "multiline": True}),
                "accent": ("STRING", {"default": "Standard American English", "multiline": False}),
                "texture": ("STRING", {"default": "Rich and velvety with a slightly smoky quality", "multiline": True}),
                "emotion": ("STRING", {"default": "Contemplative and intriguing", "multiline": True}),
                "tone": ("STRING", {"default": "Mysterious, philosophical, and atmospheric", "multiline": True}),
                "personality": ("STRING", {"default": "Introspective, wise, and captivating", "multiline": True}),
                "custom_instruction": ("STRING", {"default": "", "multiline": True, "tooltip": "Any additional custom instructions to append."}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Penalty for repetition. Increase (e.g., 1.1-1.2) to prevent infinite loops/stuttering."}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, seed, **kwargs):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS/Inference"

    def generate(self, model, text, language, seed, top_p=0.8, temperature=0.7, repetition_penalty=1.1, **kwargs):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        lang = language if language != "Auto" else None
        
        # Construct instruct string from kwargs
        fields = [
            ("gender", kwargs.get("gender")),
            ("pitch", kwargs.get("pitch")),
            ("speed", kwargs.get("speed")),
            ("volume", kwargs.get("volume")),
            ("age", kwargs.get("age")),
            ("clarity", kwargs.get("clarity")),
            ("fluency", kwargs.get("fluency")),
            ("accent", kwargs.get("accent")),
            ("texture", kwargs.get("texture")),
            ("emotion", kwargs.get("emotion")),
            ("tone", kwargs.get("tone")),
            ("personality", kwargs.get("personality")),
        ]

        prompt_lines = []
        for key, value in fields:
            if value and value.strip():
                prompt_lines.append(f"{key}: {value.strip()}")

        custom_instruction = kwargs.get("custom_instruction")
        if custom_instruction and custom_instruction.strip():
            prompt_lines.append(custom_instruction.strip())

        instruct = "\n".join(prompt_lines)
        print(f"[Qwen3-TTS] Generated Voice Design Prompt:\n{instruct}")

        gen_kwargs = {
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }

        try:
            try:
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=instruct,
                    **gen_kwargs
                )
            except TypeError:
                print("Warning: Model generation function does not support extra parameters (top_p, temperature, repetition_penalty). Ignoring them.")
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=instruct
                )
        except ValueError as e:
             msg = str(e)
             if "does not support generate_voice_design" in msg:
                 raise ValueError("Model Type Error: You are trying to use 'Voice Design' with an incompatible model. Please load a 'VoiceDesign' model (e.g. Qwen3-TTS-12Hz-1.7B-VoiceDesign).") from e
             raise e
             
        return (convert_audio(wavs[0], sr),)




class Qwen3PromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "ref_audio_max_seconds": ("FLOAT", {"default": 30.0, "min": -1.0, "max": 120.0, "step": 5.0}),
            }
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS/Inference"

    def create_prompt(self, model, ref_audio, ref_text, ref_audio_max_seconds=30.0):
        audio_tuple = load_audio_input(ref_audio)
        
        # Trim reference audio if too long to prevent generation hangs (-1 = no limit)
        if audio_tuple is not None and ref_audio_max_seconds > 0:
            wav_data, audio_sr = audio_tuple
            max_samples = int(ref_audio_max_seconds * audio_sr)
            if len(wav_data) > max_samples:
                print(f"Trimming reference audio from {len(wav_data)/audio_sr:.1f}s to {ref_audio_max_seconds}s to prevent generation issues")
                wav_data = wav_data[:max_samples]
                audio_tuple = (wav_data, audio_sr)
        
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=audio_tuple,
                ref_text=ref_text
            )
        except ValueError as e:
             msg = str(e)
             # Assumption: create_voice_clone_prompt might also be restricted to Base models? 
             # README doesn't explicitly restrict it but implies it's for cloning.
             if "does not support" in msg:
                 raise ValueError("Model Type Error: This model does not support creating voice clone prompts. Please load a 'Base' model.") from e
             raise e
             
        return (prompt,)


class Qwen3SavePrompt:
    """Save a QWEN3_PROMPT (voice clone embedding) to disk as safetensors."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("QWEN3_PROMPT",),
                "filename": ("STRING", {"default": "voice_embedding"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_prompt"
    CATEGORY = "Qwen3-TTS/Utils"
    OUTPUT_NODE = True

    def save_prompt(self, prompt, filename):
        # prompt is List[VoiceClonePromptItem], we save the first item
        if not prompt or len(prompt) == 0:
            raise ValueError("Empty prompt - nothing to save")
        
        item = prompt[0]
        
        # Build tensors dict for safetensors
        tensors = {
            "ref_spk_embedding": item.ref_spk_embedding.contiguous().cpu(),
        }
        if item.ref_code is not None:
            tensors["ref_code"] = item.ref_code.contiguous().cpu()
        
        # Build metadata dict (safetensors metadata must be strings)
        metadata = {
            "x_vector_only_mode": str(item.x_vector_only_mode),
            "icl_mode": str(item.icl_mode),
        }
        if item.ref_text is not None:
            metadata["ref_text"] = item.ref_text
        
        # Ensure filename has no extension (we add .safetensors)
        if filename.endswith(".safetensors"):
            filename = filename[:-12]
        
        filepath = os.path.join(QWEN3_TTS_PROMPTS_DIR, f"{filename}.safetensors")
        
        save_file(tensors, filepath, metadata=metadata)
        print(f"Saved voice prompt to: {filepath}")
        
        return (filepath,)


class Qwen3LoadPrompt:
    """Load a QWEN3_PROMPT (voice clone embedding) from disk."""
    
    @classmethod
    def INPUT_TYPES(s):
        # Get list of available prompt files
        prompt_files = []
        if os.path.exists(QWEN3_TTS_PROMPTS_DIR):
            for f in os.listdir(QWEN3_TTS_PROMPTS_DIR):
                if f.endswith(".safetensors"):
                    prompt_files.append(f)
        if not prompt_files:
            prompt_files = ["no prompts saved yet"]
        
        return {
            "required": {
                "prompt_file": (sorted(prompt_files),),
            },
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Qwen3-TTS/Utils"

    @classmethod
    def IS_CHANGED(s, prompt_file):
        # Return file modification time to detect changes
        filepath = os.path.join(QWEN3_TTS_PROMPTS_DIR, prompt_file)
        if os.path.exists(filepath):
            return os.path.getmtime(filepath)
        return float("nan")

    def load_prompt(self, prompt_file):
        if prompt_file == "no prompts saved yet":
            raise ValueError("No prompt files available. Save a prompt first using Qwen3-TTS Save Prompt.")
        
        filepath = os.path.join(QWEN3_TTS_PROMPTS_DIR, prompt_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        
        # Load tensors
        tensors = load_file(filepath)
        
        # Load metadata
        from safetensors import safe_open
        with safe_open(filepath, framework="pt") as f:
            metadata = f.metadata() or {}
        
        # Reconstruct VoiceClonePromptItem
        ref_spk_embedding = tensors["ref_spk_embedding"]
        ref_code = tensors.get("ref_code", None)
        x_vector_only_mode = metadata.get("x_vector_only_mode", "False") == "True"
        icl_mode = metadata.get("icl_mode", "False") == "True"
        ref_text = metadata.get("ref_text", None)
        
        item = VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
            ref_text=ref_text,
        )
        
        print(f"Loaded voice prompt from: {filepath}")
        return ([item],)


class Qwen3VoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
                "prompt": ("QWEN3_PROMPT",),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "ref_audio_max_seconds": ("FLOAT", {"default": 30.0, "min": -1.0, "max": 120.0, "step": 5.0}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Penalty for repetition. Increase (e.g., 1.1-1.2) to prevent infinite loops/stuttering."}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, seed, language="Auto", ref_audio=None, ref_text=None, prompt=None, max_new_tokens=2048, ref_audio_max_seconds=30.0, top_p=0.8, temperature=0.7, repetition_penalty=1.1):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS/Inference"

    def generate(self, model, text, seed, language="Auto", ref_audio=None, ref_text=None, prompt=None, max_new_tokens=2048, ref_audio_max_seconds=30.0, top_p=0.8, temperature=0.7, repetition_penalty=1.1):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        lang = language if language != "Auto" else None
        
        wavs = None
        sr = 0

        gen_kwargs = {
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }
        
        try:
            if prompt is not None:
                # Use pre-calculated prompt
                try:
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang,
                        voice_clone_prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs
                    )
                except TypeError:
                    print("Warning: Model generation function does not support extra parameters. Ignoring them.")
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang,
                        voice_clone_prompt=prompt,
                        max_new_tokens=max_new_tokens
                    )
            elif ref_audio is not None and ref_text is not None and ref_text.strip() != "":
                # Use on-the-fly prompt creation
                audio_tuple = load_audio_input(ref_audio)
                
                # Trim reference audio if too long to prevent generation hangs (-1 = no limit)
                if audio_tuple is not None and ref_audio_max_seconds > 0:
                    wav_data, audio_sr = audio_tuple
                    max_samples = int(ref_audio_max_seconds * audio_sr)
                    if len(wav_data) > max_samples:
                        print(f"Trimming reference audio from {len(wav_data)/audio_sr:.1f}s to {ref_audio_max_seconds}s to prevent generation issues")
                        wav_data = wav_data[:max_samples]
                        audio_tuple = (wav_data, audio_sr)
                
                try:
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang,
                        ref_audio=audio_tuple,
                        ref_text=ref_text,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs
                    )
                except TypeError:
                     print("Warning: Model generation function does not support extra parameters. Ignoring them.")
                     wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang,
                        ref_audio=audio_tuple,
                        ref_text=ref_text,
                        max_new_tokens=max_new_tokens
                    )
            else:
                 raise ValueError("For Voice Clone, you must provide either 'prompt' OR ('ref_audio' AND 'ref_text').")
        except ValueError as e:
            msg = str(e)
            if "does not support generate_voice_clone" in msg:
                raise ValueError("Model Type Error: You are trying to use 'Voice Clone' with an incompatible model. Please load a 'Base' model (e.g. Qwen3-TTS-12Hz-1.7B-Base).") from e
            raise e
             
        return (convert_audio(wavs[0], sr),)

class Qwen3AudioToDataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_folder": ("STRING", {"default": "", "multiline": False}),
                "model_size": (["tiny", "base", "small", "medium", "large", "large-v3"], {"default": "medium"}),
            },
            "optional": {
                "output_folder_name": ("STRING", {"default": "dataset_final"}),
                "min_duration": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 10.0, "step": 0.1}),
                "max_duration": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "silence_threshold": ("FLOAT", {"default": -40.0, "min": -100.0, "max": 0.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jsonl_path",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "Qwen3-TTS/Dataset"

    def process(self, audio_folder, model_size, output_folder_name="dataset_final", min_duration=0.8, max_duration=15.0, silence_threshold=-40.0):
        if not HAS_WHISPER_PYDUB:
             raise ImportError("Please install 'openai-whisper' and 'pydub' to use this node.")

        audio_folder = fix_wsl_path(audio_folder)
        print(f"[Qwen3-TTS] AudioToDataset: Processing {audio_folder}")

        if not os.path.exists(audio_folder):
            raise ValueError(f"Audio folder not found: {audio_folder}")

        # Determine output folder
        # If output_folder_name is relative, put it inside audio_folder parent or similar?
        # User script put it in "dataset_final" inside CURRENT_DIR.
        # Let's put it as a subfolder of audio_folder by default if not absolute.
        if os.path.isabs(output_folder_name):
            output_folder = output_folder_name
        else:
            output_folder = os.path.join(audio_folder, output_folder_name)

        os.makedirs(output_folder, exist_ok=True)
        print(f"[Qwen3-TTS] Output folder: {output_folder}")

        # Load Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Qwen3-TTS] Loading Whisper '{model_size}' on {device}...")
        try:
             model = whisper.load_model(model_size, device=device)
        except Exception as e:
             raise RuntimeError(f"Failed to load Whisper model: {e}")

        # Find files
        files = [f for f in os.listdir(audio_folder)
                 if f.lower().endswith(".wav") and os.path.isfile(os.path.join(audio_folder, f))]
        files.sort()

        if not files:
             raise ValueError(f"No .wav files found in {audio_folder}")

        print(f"[Qwen3-TTS] Found {len(files)} files")

        total_clips = 0

        def trim_silence_wrapper(audio_segment, thresh, chunk=10):
            try:
                start_trim = silence.detect_leading_silence(audio_segment, silence_threshold=thresh, chunk_size=chunk)
                end_trim = silence.detect_leading_silence(audio_segment.reverse(), silence_threshold=thresh, chunk_size=chunk)
                duration = len(audio_segment)
                return audio_segment[start_trim:duration-end_trim]
            except:
                return audio_segment

        for filename in files:
            filepath = os.path.join(audio_folder, filename)
            base_name = os.path.splitext(filename)[0]

            print(f"Processing: {filename}")

            try:
                audio_full = AudioSegment.from_wav(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # Transcribe
            result = model.transcribe(filepath, language="es", verbose=False)

            file_count = 0
            for segment in result['segments']:
                start_ms = segment['start'] * 1000
                end_ms = segment['end'] * 1000
                text = segment['text'].strip()

                if not text:
                    continue

                chunk = audio_full[start_ms:end_ms]
                chunk_trimmed = trim_silence_wrapper(chunk, silence_threshold)

                duration_sec = len(chunk_trimmed) / 1000.0

                if duration_sec > max_duration:
                     continue
                if duration_sec < min_duration:
                     continue

                chunk_name = f"{base_name}_{file_count:04d}"
                wav_path = os.path.join(output_folder, f"{chunk_name}.wav")
                txt_path = os.path.join(output_folder, f"{chunk_name}.txt")

                # Export: Mono, 22050Hz (Standard TTS)
                # Qwen actually uses 24kHz or similar but 22050 is fine for general TTS datasets usually?
                # Qwen3-TTS usually expects 24kHz in other nodes, let's stick to 24000 to match Qwen3 reqs better?
                # But user script had 22050. Let's respect user script (22050) or upgrade?
                # User script: chunk_trimmed.set_frame_rate(22050).set_channels(1)
                # Qwen nodes.py: convert_audio handles resampling if needed or librosa loads at 24k.
                # Let's stick to user script 22050 to stay safe, or 24000?
                # Let's use 24000 to match Qwen3 native preference if possible, but 22050 is safe standard.
                # I'll stick to 22050 as requested in script.

                chunk_trimmed.set_frame_rate(22050).set_channels(1).export(wav_path, format="wav")

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                file_count += 1
                total_clips += 1

        print(f"[Qwen3-TTS] Process finished. Generated {total_clips} clips in {output_folder}")

        # Generate dataset.jsonl for training
        jsonl_path = os.path.join(output_folder, "dataset.jsonl")
        print(f"[Qwen3-TTS] Generating training index: {jsonl_path}")

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for filename in os.listdir(output_folder):
                if filename.endswith(".wav"):
                    wav_path = os.path.abspath(os.path.join(output_folder, filename))
                    txt_path = os.path.splitext(wav_path)[0] + ".txt"

                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as tf:
                            text_content = tf.read().strip()

                        if text_content:
                            entry = {
                                "audio": wav_path,
                                "text": text_content
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Qwen3-TTS] Dataset JSONL created successfully.")

        # Clean up VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (jsonl_path,)


class Qwen3LoadDatasetAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("DATASET_AUDIO_LIST",)
    RETURN_NAMES = ("audio_list",)
    FUNCTION = "load_audio"
    CATEGORY = "Qwen3-TTS/Dataset"

    def load_audio(self, folder_path):
        folder_path = folder_path.strip().strip('"')
        folder_path = fix_wsl_path(folder_path)

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(".wav") and os.path.isfile(os.path.join(folder_path, f))]
        files.sort()

        if not files:
             raise ValueError(f"No .wav files found in {folder_path}")

        print(f"[Qwen3-TTS] Found {len(files)} .wav files in {folder_path}")
        return ({"folder_path": folder_path, "files": files},)


class Qwen3TranscribeWhisper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_list": ("DATASET_AUDIO_LIST",),
                "whisper_model": (["tiny", "base", "small", "medium", "large", "large-v3"], {"default": "medium"}),
            },
            "optional": {
                "output_dataset_folder": ("STRING", {"default": "dataset_final", "multiline": False}),
                "min_duration": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 10.0, "step": 0.1}),
                "max_duration": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "silence_threshold": ("FLOAT", {"default": -40.0, "min": -100.0, "max": 0.0, "step": 1.0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("DATASET_ITEMS",)
    RETURN_NAMES = ("dataset_items",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "Qwen3-TTS/Dataset"

    def process(self, audio_list, whisper_model, output_dataset_folder="dataset_final", min_duration=0.8, max_duration=60.0, silence_threshold=-40.0, unique_id=None):
        if not HAS_WHISPER_PYDUB:
             raise ImportError("Please install 'openai-whisper' and 'pydub' to use this node.")

        folder_path = audio_list["folder_path"]
        files = audio_list["files"]
        
        # Define output directory for processed files
        processed_dir = os.path.join(folder_path, output_dataset_folder)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Check if processed files exist
        processed_wavs = [f for f in os.listdir(processed_dir) if f.lower().endswith('.wav')]

        if not processed_wavs:
            print(f"[Qwen3-TTS] No processed files found in {processed_dir}. Starting Whisper processing...")

            # Load Whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Qwen3-TTS] Loading Whisper '{whisper_model}' on {device}...")
            try:
                model = whisper.load_model(whisper_model, device=device)
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}")

            total_files = len(files)
            print(f"[Qwen3-TTS] Processing {total_files} files...")

            def trim_silence(audio_segment, silence_threshold=-40.0, chunk_size=10):
                try:
                    start_trim = silence.detect_leading_silence(audio_segment, silence_threshold=silence_threshold, chunk_size=chunk_size)
                    end_trim = silence.detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_threshold, chunk_size=chunk_size)
                    duration = len(audio_segment)
                    return audio_segment[start_trim:duration-end_trim]
                except:
                    return audio_segment

            for idx, filename in enumerate(tqdm(files, desc="Processing Audio", unit="file")):
                # Update progress
                # if unique_id:
                #    PromptServer.instance.send_progress(idx, total_files, unique_id)

                filepath = os.path.join(folder_path, filename)
                base_name = os.path.splitext(filename)[0]

                print(f"--- Processing [{idx+1}/{total_files}]: {filename} ---")

                try:
                    audio_full = AudioSegment.from_wav(filepath)
                except Exception as e:
                    print(f"   [Error reading audio] {e}")
                    continue

                # Transcribe
                result = model.transcribe(filepath, language="es", verbose=False)

                file_count = 0
                for segment in result['segments']:
                    start_ms = segment['start'] * 1000
                    end_ms = segment['end'] * 1000
                    text = segment['text'].strip()

                    chunk = audio_full[start_ms:end_ms]

                    # Trim Silence
                    try:
                        chunk_trimmed = trim_silence(chunk, silence_threshold=silence_threshold)
                    except:
                        chunk_trimmed = chunk

                    duration_sec = len(chunk_trimmed) / 1000.0

                    if duration_sec > max_duration:
                        continue
                    if duration_sec < min_duration:
                        continue

                    # Save
                    chunk_name = f"{base_name}_{file_count:04d}"
                    wav_path = os.path.join(processed_dir, f"{chunk_name}.wav")
                    txt_path = os.path.join(processed_dir, f"{chunk_name}.txt")

                    # Standard TTS Format: Mono, 22050Hz
                    chunk_trimmed.set_frame_rate(22050).set_channels(1).export(wav_path, format="wav")

                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    file_count += 1

            # Clean up VRAM
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Refresh list
            processed_wavs = [f for f in os.listdir(processed_dir) if f.lower().endswith('.wav')]
        else:
            print(f"[Qwen3-TTS] Found {len(processed_wavs)} existing processed files. Skipping Whisper.")

        # Build list of items for next step
        items = []
        for wav_file in processed_wavs:
            base_name = os.path.splitext(wav_file)[0]
            txt_path = os.path.join(processed_dir, f"{base_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    items.append({
                        "audio_path": os.path.join(processed_dir, wav_file),
                        "text": text
                    })

        return (items,)


class Qwen3AutoLabelEmotions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset_items": ("DATASET_ITEMS",),
                "model": (["Qwen/Qwen2-Audio-7B-Instruct"], {"default": "Qwen/Qwen2-Audio-7B-Instruct"}),
                "gender_override": (["Auto", "Female", "Male"], {"default": "Auto", "tooltip": "Force a specific gender tag implies changing 'Male voice' to 'Female voice' (or vice versa) in the description."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("DATASET_ITEMS",)
    RETURN_NAMES = ("labeled_items",)
    OUTPUT_NODE = True
    FUNCTION = "label_emotions"
    CATEGORY = "Qwen3-TTS/Dataset"

    def label_emotions(self, dataset_items, model, gender_override, unique_id=None):
        if not HAS_LIBROSA:
             raise ImportError("Please install 'librosa' to use this node.")

        print(f"[Qwen3-TTS] Auto-Labeling Emotions ({len(dataset_items)} items). Gender mode: {gender_override}")

        # Keyword mapping (Defaults mostly to Male, will be fixed by override)
        keyword_map = {
            "angry": "Male voice, very angry, shouting, aggressive tone, high pitch.",
            "enfadado": "Male voice, very angry, shouting, aggressive tone, high pitch.",
            "sad": "Male voice, very sad, crying, low pitch, melancholic tone.",
            "triste": "Male voice, very sad, crying, low pitch, melancholic tone.",
            "happy": "Male voice, happy, laughing, bright tone, excited.",
            "feliz": "Male voice, happy, laughing, bright tone, excited.",
            "whisper": "Male voice, whispering, quiet, soft tone.",
            "susurro": "Male voice, whispering, quiet, soft tone.",
            "scared": "Male voice, scared, trembling, high pitch, fearful.",
            "miedo": "Male voice, scared, trembling, high pitch, fearful.",
            "surprised": "Male voice, surprised, shocked, high pitch.",
            "sorpresa": "Male voice, surprised, shocked, high pitch.",
        }

        items_to_infer = []
        total = len(dataset_items)

        # Helper to apply gender override
        def apply_gender_fix(text, mode):
            if mode == "Auto": return text

            if mode == "Female":
                # Fix common mislabeling
                text = text.replace("Male voice", "Female voice")
                text = text.replace("male voice", "female voice")
                # Ensure presence
                if "Female voice" not in text and "female voice" not in text:
                    text = "Female voice, " + text

            elif mode == "Male":
                text = text.replace("Female voice", "Male voice")
                text = text.replace("female voice", "male voice")
                if "Male voice" not in text and "male voice" not in text:
                    text = "Male voice, " + text

            return text

        # --- PHASE 1: Pre-Scan (Cache & Keywords) ---
        print("[Qwen3-TTS] Scanning for cached labels and keywords...")
        for idx, item in enumerate(dataset_items):
            file_path = item["audio_path"]
            filename = os.path.basename(file_path)
            label_path = os.path.splitext(file_path)[0] + ".emotion.txt"

            instruction = None

            # 1. Try Cache
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        instruction = f.read().strip()
                except Exception: pass

            # 2. Try Keywords (if no cache)
            if not instruction:
                path_lower = file_path.lower()
                for kw, kw_instruction in keyword_map.items():
                    if kw in path_lower:
                        instruction = kw_instruction
                        print(f"[{idx+1}/{total}] Keyword match: {filename}")
                        break

            # If we found an instruction (Cache or Keyword), apply gender fix immediately
            if instruction:
                instruction = apply_gender_fix(instruction, gender_override)
                item["instruction"] = instruction

                # Update cache with the fix to avoid re-doing it later
                try:
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write(instruction)
                except: pass
                continue

            # 3. Needs Inference
            items_to_infer.append(item)

        # --- PHASE 2: Inference ---
        if items_to_infer:
            print(f"[Qwen3-TTS] {len(items_to_infer)} items require inference. Loading model...")

            try:
                # Try offline first
                try:
                    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
                    model_obj = Qwen2AudioForConditionalGeneration.from_pretrained(model, device_map="auto", trust_remote_code=True, local_files_only=True)
                except Exception:
                    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
                    model_obj = Qwen2AudioForConditionalGeneration.from_pretrained(model, device_map="auto", trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen2-Audio model: {e}")

            system_prompt = (
                "You are a dataset tagger. Listen to the audio and output ONLY a short description. "
                "Format: [Gender], [Emotion], [Tone], [Speed], [Pitch]. "
            )

            for idx, item in enumerate(tqdm(items_to_infer, desc="Labeling Emotions", unit="item")):
                file_path = item["audio_path"]
                label_path = os.path.splitext(file_path)[0] + ".emotion.txt"

                try:
                    audio, sr = librosa.load(file_path, sr=processor.feature_extractor.sampling_rate)
                    conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": file_path}, {"type": "text", "text": system_prompt}]}]
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

                    inputs = processor(text=text, audios=audio, sampling_rate=sr, return_tensors="pt", padding=True).to(model_obj.device)

                    with torch.no_grad():
                        generated_ids = model_obj.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.5, top_p=0.9)

                    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
                    raw_instruction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    # APPLY GENDER FIX
                    final_instruction = apply_gender_fix(raw_instruction, gender_override)

                    item["instruction"] = final_instruction

                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write(final_instruction)

                except Exception as e:
                    print(f"Error labeling {file_path}: {e}")
                    item["instruction"] = apply_gender_fix("Neutral voice, normal speed.", gender_override)

            del model_obj, processor
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return (dataset_items,)


class Qwen3ExportJSONL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset_items": ("DATASET_ITEMS",),
                "output_filename": ("STRING", {"default": "dataset.jsonl", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jsonl_path",)
    OUTPUT_NODE = True
    FUNCTION = "export"
    CATEGORY = "Qwen3-TTS/Dataset"

    def export(self, dataset_items, output_filename):
        if not dataset_items:
            raise ValueError("No dataset items to export.")

        # Determine output folder from first item
        first_path = dataset_items[0]["audio_path"]
        output_dir = os.path.dirname(first_path)
        jsonl_path = os.path.join(output_dir, output_filename)

        # Auto-detect reference
        # Try to find 'ref.wav' in the same folder
        ref_path = os.path.join(output_dir, "ref.wav")
        if not os.path.exists(ref_path):
            # Use first item
            ref_path = first_path
            print(f"Using auto-selected reference: {ref_path}")

        full_ref_path = os.path.abspath(ref_path)

        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in dataset_items:
                wav_path = os.path.abspath(item["audio_path"])

                # Check reference
                if wav_path == full_ref_path:
                    pass

                entry = {
                    "audio": wav_path,
                    "text": item["text"],
                    "ref_audio": full_ref_path
                }

                # Add instruction if present
                if "instruction" in item:
                    entry["instruction"] = item["instruction"]

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"Exported {count} items to {jsonl_path}")
        return (jsonl_path,)

class Qwen3DataPrep:
    @classmethod
    def INPUT_TYPES(s):
        # We need the TEXT tokenizer repo too, which is part of the main model usually
        # But Qwen3-TTS usually uses Qwen2.5-7B or similar as text base.
        # The Qwen3TTSTokenizer is mainly for audio codes? No, it wraps both usually.
        # Let's check imports. Qwen3TTSTokenizer in qwen-tts library handles audio.
        # We need the standard HF AutoTokenizer for text.
        return {
            "required": {
                "jsonl_path": ("STRING", {"default": "", "multiline": False}),
                "audio_tokenizer_repo": (list(QWEN3_TTS_TOKENIZERS.keys()), {"default": "Qwen/Qwen3-TTS-Tokenizer-12Hz"}),
                "text_tokenizer_repo": ("STRING", {"default": "Qwen/Qwen2.5-7B-Instruct", "tooltip": "Repo for the text tokenizer (e.g. Qwen2.5-7B-Instruct)"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 32, "tooltip": "Number of audio files to process at once. Lower values use less VRAM."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_jsonl_path",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "Qwen3-TTS/Dataset"

    def process(self, jsonl_path, audio_tokenizer_repo, text_tokenizer_repo, source, batch_size, unique_id=None):
        jsonl_path = fix_wsl_path(jsonl_path)
        print(f"[Qwen3-TTS DEBUG] DataPrep process started. Input: {jsonl_path}")

        # Helper to send progress text to UI
        def send_status(text):
            if unique_id:
                PromptServer.instance.send_progress_text(text, unique_id)

        device = mm.get_torch_device()
        print(f"[Qwen3-TTS DEBUG] Using device: {device}")

        output_path = jsonl_path.replace(".jsonl", "_codes.jsonl")
        meta_path = jsonl_path.replace(".jsonl", "_codes.meta.json")

        input_hash = compute_file_hash(jsonl_path)
        input_line_count = count_jsonl_lines(jsonl_path)

        # Check cache validity
        if os.path.exists(output_path):
            metadata = load_cache_metadata(meta_path)
            if metadata:
                if (metadata.get('input_hash') == input_hash and
                    metadata.get('audio_tokenizer') == audio_tokenizer_repo and
                    metadata.get('output_line_count') == input_line_count):
                    # Verify output file integrity
                    if count_jsonl_lines(output_path) == metadata.get('output_line_count'):
                        print(f"[Qwen3DataPrep] Cache hit - using existing processed data")
                        send_status("Using cached data (no reprocessing needed)")
                        return (output_path,)

        # 1. Load Audio Tokenizer (Qwen3 specific)
        local_path = get_local_model_path(audio_tokenizer_repo)
        if os.path.exists(local_path) and os.listdir(local_path):
            audio_tok_path = local_path
        else:
            print(f"Audio Tokenizer not found locally. Downloading {audio_tokenizer_repo}...")
            audio_tok_path = download_model_to_comfyui(audio_tokenizer_repo, source)

        send_status("Loading audio tokenizer...")
        audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            audio_tok_path,
            device_map=device,
        )

        # 2. Load Text Tokenizer (Standard HF)
        # Note: We don't have a dedicated folder mapping for arbitrary text tokenizers,
        # so we rely on HF cache or download to standard cache for now, or use download_model_to_comfyui generic logic
        # Ideally, we check if it's already in the models folder.
        text_tok_local = get_local_model_path(text_tokenizer_repo)
        if os.path.exists(text_tok_local) and os.listdir(text_tok_local):
             text_tok_path = text_tok_local
        else:
             print(f"Text Tokenizer not found locally. Downloading {text_tokenizer_repo}...")
             # This uses the same logic (snapshot download), which is fine for tokenizers
             text_tok_path = download_model_to_comfyui(text_tokenizer_repo, source)

        send_status("Loading text tokenizer...")
        text_tokenizer = AutoTokenizer.from_pretrained(text_tok_path, trust_remote_code=True)

        inputs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                inputs.append(json.loads(line.strip()))

        total_items = len(inputs)
        total_batches = (total_items + batch_size - 1) // batch_size
        print(f"Processing {total_items} items in {total_batches} batches...")

        with open(output_path, 'w', encoding='utf-8') as out_file:
            def process_item_and_write(item, audio_code_tensor):
                """Helper to process a single item with its audio code and write to file."""
                audio_tokens = audio_code_tensor.cpu().tolist() # List of ints

                # Prepare Prompt
                instruction = item.get("instruction", "Speak clearly.")
                text_content = item.get("text", "")

                # ChatML format roughly (adapt as needed for Qwen instruct)
                # For Qwen2.5 Audio/TTS, the prompt format usually involves <|audio_start|> tokens etc.
                # but for basic causal LM training we treat it as text continuation.
                # Simplified Prompt:
                prompt_text = f"Instruction: {instruction}\nText: {text_content}\nGenerate Audio:"

                # Tokenize Text
                text_ids = text_tokenizer.encode(prompt_text, add_special_tokens=True)

                # Combine: Text + Audio
                input_ids = text_ids + audio_tokens

                # Create Labels: -100 for text, audio_tokens for audio
                labels = ([-100] * len(text_ids)) + audio_tokens

                # Attention Mask
                attention_mask = [1] * len(input_ids)

                # Update item with training tensors
                item['input_ids'] = input_ids
                item['labels'] = labels
                item['attention_mask'] = attention_mask
                # Optional: keep audio_codes for reference
                item['audio_codes'] = audio_tokens

                out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

            for batch_idx, i in enumerate(range(0, total_items, batch_size)):
                batch = inputs[i:i+batch_size]
                audio_paths = [b['audio'] for b in batch]

                status_msg = f"Processing batch {batch_idx + 1}/{total_batches}..."
                print(status_msg)
                send_status(status_msg)

                try:
                    # A. Attempt to Encode Audio (Batch)
                    enc_res = audio_tokenizer.encode(audio_paths)
                    audio_codes_batch = enc_res.audio_codes # List of tensors

                    # B. Process Text and Labels for each item
                    for j, code in enumerate(audio_codes_batch):
                        process_item_and_write(batch[j], code)

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # Catch OOM or "Allocation on device" RuntimeError
                    err_msg = str(e)
                    if "out of memory" in err_msg.lower() or "allocation on device" in err_msg.lower():
                        print(f"[Qwen3DataPrep] ⚠️ OOM detected in batch {batch_idx+1}. Switching to sequential processing for this batch.")
                        send_status(f"⚠️ OOM in batch {batch_idx+1}. Retrying sequentially...")

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Fallback: Process one by one
                        for b_item in batch:
                            try:
                                # Encode single audio
                                # audio_tokenizer.encode expects a list
                                enc_res_single = audio_tokenizer.encode([b_item['audio']])
                                code_single = enc_res_single.audio_codes[0]
                                process_item_and_write(b_item, code_single)

                                # Clear cache after each heavy item in fallback mode
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception as inner_e:
                                print(f"[Qwen3DataPrep] ❌ Failed to process file {b_item['audio']} even individually: {inner_e}")
                                # We skip this file or raise?
                                # If it fails individually, it's likely too big for VRAM even solo.
                                # Let's skip it to allow the rest to finish, but log it error.
                    else:
                        raise e # Re-raise if it's not a memory error

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save cache metadata
        metadata = {
            "version": 1,
            "input_hash": input_hash,
            "audio_tokenizer": audio_tokenizer_repo,
            "output_line_count": len(inputs),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        save_cache_metadata(meta_path, metadata)

        print(f"Processed dataset saved to {output_path}")
        send_status("Data preparation complete!")
        return (output_path,)

class Qwen3FineTune:
    @classmethod
    def INPUT_TYPES(s):
        # Get base models (excluding CustomVoice/VoiceDesign for fine-tuning)
        base_models = [k for k in QWEN3_TTS_MODELS.keys() if "Base" in k]
        return {
            "required": {
                "train_jsonl": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to the preprocessed JSONL file containing training data with audio codes."}),
                "init_model": (base_models, {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "tooltip": "Base model to fine-tune. Must be a 'Base' model variant."}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace", "tooltip": "Download source if model is not found locally."}),
                "output_dir": ("STRING", {"default": "models/tts/finetuned_model", "multiline": False, "tooltip": "Directory to save checkpoints and final model."}),
                "epochs": ("INT", {"default": 3, "min": 1, "max": 1000, "tooltip": "Number of training epochs to run."}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "Number of samples per batch. Lower values use less VRAM."}),
                "lr": ("FLOAT", {"default": 2e-6, "step": 1e-7, "tooltip": "Learning rate. Qwen default (2e-5) is too aggressive for small batches, causing noise output. Defaults to 2e-6 for stability."}),
                "target_loss": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Detener entrenamiento si el Loss baja de este valor (0.0 = desactivado)"}),
                "speaker_name": ("STRING", {"default": "my_speaker", "tooltip": "Name for the custom speaker. Use this name when generating with the fine-tuned model."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility."}),
            },
            "optional": {
                 # Workflow
                 "resume_training": ("BOOLEAN", {"default": False, "tooltip": "Continue training from the latest checkpoint in output_dir."}),
                 "log_every_steps": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Log training progress every N steps."}),
                 "save_every_epochs": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": "Save checkpoint every N epochs. Set to 0 to only save final epoch. Ignored if save_every_steps > 0."}),
                 "save_every_steps": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Save checkpoint every N steps. Set to 0 to use epoch-based saving instead."}),
                 # VRAM Optimizations
                 "mixed_precision": (["bf16", "fp32"], {"default": "bf16", "tooltip": "bf16 recommended. Use fp32 only if GPU doesn't support bf16 (pre-Ampere)."}),
                 "gradient_accumulation": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Accumulate gradients over N steps before updating. Effective batch size = batch_size * gradient_accumulation."}),
                 "gradient_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "Trade compute for VRAM by recomputing activations. Saves ~30-40% VRAM."}),
                 "use_8bit_optimizer": ("BOOLEAN", {"default": True, "tooltip": "Use 8-bit AdamW optimizer. Saves ~50% optimizer VRAM. Requires bitsandbytes."}),
                 # Training Dynamics
                 "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "L2 regularization strength to prevent overfitting."}),
                 "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Gradient clipping threshold to prevent exploding gradients."}),
                 # Learning Rate Schedule
                 "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Number of warmup steps. Set to 0 to disable warmup. Recommended: 5-10% of total steps."}),
                 "warmup_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Warmup as ratio of total steps. Ignored if warmup_steps > 0. E.g., 0.1 = 10% warmup."}),
                 "save_optimizer_state": ("BOOLEAN", {"default": False, "tooltip": "Save optimizer/scheduler state in checkpoints. Enables perfect resume but doubles checkpoint size."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "custom_speaker_name")
    OUTPUT_NODE = True
    FUNCTION = "train"
    CATEGORY = "Qwen3-TTS/Training"

    def train(self, train_jsonl, init_model, source, output_dir, epochs, batch_size, lr, target_loss, speaker_name, seed, mixed_precision="bf16", resume_training=False, log_every_steps=10, save_every_epochs=1, save_every_steps=0, gradient_accumulation=4, gradient_checkpointing=True, use_8bit_optimizer=True, weight_decay=0.01, max_grad_norm=1.0, warmup_steps=0, warmup_ratio=0.0, save_optimizer_state=False, unique_id=None):
        train_jsonl = fix_wsl_path(train_jsonl)
        output_dir = fix_wsl_path(output_dir)
        init_model = fix_wsl_path(init_model)

        print(f"[Qwen3-TTS DEBUG] Train called. JSONL: {train_jsonl}, Output: {output_dir}")
        print(f"[Qwen3-TTS DEBUG] Config: Epochs={epochs}, Batch={batch_size}, LR={lr}, MixedPrecision={mixed_precision}")

        # Helper to send progress text to UI
        def send_status(text):
            if unique_id:
                PromptServer.instance.send_progress_text(text, unique_id)

        # Setup output directory
        # Fix path resolution: if relative, anchor to ComfyUI base path
        if not os.path.isabs(output_dir):
            base_output_dir = os.path.abspath(os.path.join(folder_paths.base_path, output_dir))
        else:
            base_output_dir = os.path.abspath(output_dir)

        # Create the speaker subdirectory
        full_output_dir = os.path.join(base_output_dir, speaker_name)
        os.makedirs(full_output_dir, exist_ok=True)
        print(f"[Qwen3-TTS DEBUG] Full output directory: {full_output_dir}")

        # Check for resume checkpoint
        start_epoch = 0
        resume_from_step = 0  # Track step offset for ckpt_step_N checkpoints
        resume_checkpoint_path = None

        if resume_training:
            # Priority 1: Find checkpoint subfolders (prefer trained-on checkpoints)
            checkpoints = []
            if os.path.exists(full_output_dir):
                for item in os.listdir(full_output_dir):
                    item_path = os.path.join(full_output_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
                        mtime = os.path.getmtime(os.path.join(item_path, "pytorch_model.bin"))
                        checkpoints.append((mtime, item, item_path))

            if checkpoints:
                # Sort by mtime (most recent first)
                checkpoints.sort(key=lambda x: x[0], reverse=True)
                _, item_name, resume_checkpoint_path = checkpoints[0]

                # Extract epoch OR step number from folder name
                if item_name.startswith("epoch_"):
                    try:
                        start_epoch = int(item_name.split("_")[1])
                    except (ValueError, IndexError):
                        pass
                elif item_name.startswith("ckpt_step_"):
                    try:
                        resume_from_step = int(item_name.split("_")[2])
                    except (ValueError, IndexError):
                        pass

                print(f"Resume: Found checkpoint '{item_name}' (most recent)")
            else:
                # Priority 2: Check if output_dir itself is a checkpoint
                direct_weights = os.path.join(full_output_dir, "pytorch_model.bin")
                if os.path.exists(direct_weights):
                    resume_checkpoint_path = full_output_dir
                    dir_name = os.path.basename(full_output_dir)
                    if dir_name.startswith("ckpt_step_"):
                        try:
                            resume_from_step = int(dir_name.split("_")[2])
                        except (ValueError, IndexError):
                            pass
                    print(f"Resume: output_dir is a checkpoint, loading from {resume_checkpoint_path}")

            # Load step_offset from checkpoint's training_config.json if not extracted from folder name
            if resume_checkpoint_path and resume_from_step == 0:
                training_config_path = os.path.join(resume_checkpoint_path, "training_config.json")
                if os.path.exists(training_config_path):
                    with open(training_config_path, 'r') as f:
                        saved_config = json.load(f)
                        resume_from_step = saved_config.get("step_offset", 0)
                        if resume_from_step > 0:
                            print(f"Loaded step_offset={resume_from_step} from checkpoint config")

            if resume_checkpoint_path:
                if resume_from_step > 0:
                    print(f"Will continue from step {resume_from_step}")
                print(f"Will train epochs {start_epoch + 1} to {start_epoch + epochs}")
            else:
                print("Resume enabled but no checkpoints found, starting fresh")

        # Resolve init_model path - check ComfyUI folder first, download if needed
        # NOTE: Always use the original base model, not checkpoint - checkpoint's model.safetensors
        # doesn't include speaker_encoder (it's stripped for inference). We load checkpoint weights separately.
        if init_model in QWEN3_TTS_MODELS:
            local_path = get_local_model_path(init_model)
            if os.path.exists(local_path) and os.listdir(local_path):
                init_model_path = local_path
                print(f"Using model from ComfyUI folder: {init_model_path}")
            else:
                print(f"Base model not found locally. Downloading {init_model}...")
                init_model_path = download_model_to_comfyui(init_model, source)
        else:
            # Assume it's a path
            init_model_path = init_model

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ComfyUI runs in inference_mode by default.
        # We must disable it and enable gradients properly for the entire scope, including model loading.
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                # Clear VRAM before loading to maximize available memory
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                use_cpu = mm.cpu_mode()
                num_gpus = torch.cuda.device_count()
                if num_gpus <= 1 or use_cpu:
                    os.environ.setdefault("RANK", "0")
                    os.environ.setdefault("WORLD_SIZE", "1")
                    os.environ.setdefault("LOCAL_RANK", "0")
                    os.environ.setdefault("MASTER_ADDR", "localhost")
                    os.environ.setdefault("MASTER_PORT", "29500")
                else:
                    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
                        os.environ.pop(key, None)
                    os.environ.setdefault("MASTER_ADDR", "localhost")
                    os.environ.setdefault("MASTER_PORT", "29500")
                    print(f"Multi-GPU training enabled: {num_gpus} GPUs detected")

                # Check GPU bf16 support and fallback to fp32 if needed
                actual_mixed_precision = mixed_precision
                if not use_cpu and torch.cuda.is_available() and mixed_precision == "bf16":
                    device_cap = torch.cuda.get_device_capability()
                    gpu_name = torch.cuda.get_device_name()
                    # bf16 requires compute capability >= 8.0 (Ampere+)
                    if device_cap[0] < 8:
                        print(f"Warning: {gpu_name} (compute {device_cap[0]}.{device_cap[1]}) does not support bf16.")
                        print("Falling back to fp32. Note: FP32 uses ~2x more VRAM than bf16.")
                        actual_mixed_precision = "fp32"

                # Accelerator uses "no" for fp32
                accel_precision = "no" if actual_mixed_precision == "fp32" else actual_mixed_precision
                accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation, mixed_precision=accel_precision, cpu=use_cpu)

                if resume_checkpoint_path:
                    print(f"Loading base model: {init_model_path} (will apply checkpoint weights from {resume_checkpoint_path})")
                else:
                    print(f"Loading base model: {init_model_path}")
                
                attn_impl = "sdpa"
                try:
                     import flash_attn
                     import importlib.metadata
                     importlib.metadata.version("flash_attn")
                     attn_impl = "flash_attention_2"
                except Exception:
                     pass

                print(f"Using attention implementation: {attn_impl}")

                dtype = torch.bfloat16 if actual_mixed_precision == "bf16" else torch.float32

                qwen3tts = Qwen3TTSModel.from_pretrained(
                    init_model_path,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )

                # Load training weights (includes speaker_encoder) if resuming
                if resume_checkpoint_path:
                    ckpt_weights = os.path.join(resume_checkpoint_path, "pytorch_model.bin")
                    if os.path.exists(ckpt_weights):
                        state_dict = torch.load(ckpt_weights, map_location="cpu", weights_only=True)
                        qwen3tts.model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded training weights from {ckpt_weights}")
                    else:
                        print(f"Warning: Training checkpoint not found at {ckpt_weights}, using model.safetensors weights")

                # FORCE GRADIENTS ON
                qwen3tts.model.train()
                for name, param in qwen3tts.model.named_parameters():
                    param.requires_grad = True

                # Enable gradient checkpointing to reduce VRAM usage (~30-40% savings)
                if gradient_checkpointing:
                    if hasattr(qwen3tts.model, 'gradient_checkpointing_enable'):
                        qwen3tts.model.gradient_checkpointing_enable()
                        print("Gradient checkpointing enabled for VRAM optimization")
                    elif hasattr(qwen3tts.model, 'talker') and hasattr(qwen3tts.model.talker, 'gradient_checkpointing_enable'):
                        qwen3tts.model.talker.gradient_checkpointing_enable()
                        print("Gradient checkpointing enabled on talker for VRAM optimization")
                else:
                    print("Gradient checkpointing disabled")

                config = AutoConfig.from_pretrained(init_model_path)
                
                # Load Data
                with open(train_jsonl, 'r', encoding='utf-8') as f:
                    train_lines = [json.loads(line) for line in f]
                    
                dataset = TTSDataset(train_lines, qwen3tts.processor, config)
                generator = torch.Generator()
                generator.manual_seed(seed)

                train_dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=dataset.collate_fn,
                    generator=generator,
                )
                
                # Use 8-bit Adam if available and enabled (saves ~50% optimizer memory)
                if HAS_BNB and use_8bit_optimizer:
                    optimizer = bnb.optim.AdamW8bit(qwen3tts.model.parameters(), lr=lr, weight_decay=weight_decay)
                    print("Using 8-bit AdamW optimizer for VRAM optimization")
                else:
                    optimizer = AdamW(qwen3tts.model.parameters(), lr=lr, weight_decay=weight_decay)
                    if not HAS_BNB:
                        print("Using standard AdamW (install bitsandbytes for lower VRAM usage)")
                    else:
                        print("Using standard AdamW (8-bit optimizer disabled)")

                # Calculate total training steps for THIS run (use ceil to avoid 0 for small datasets)
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation)
                total_training_steps = num_update_steps_per_epoch * epochs

                # Determine warmup steps (explicit steps take priority over ratio)
                actual_warmup_steps = warmup_steps
                if warmup_steps == 0 and warmup_ratio > 0:
                    actual_warmup_steps = int(total_training_steps * warmup_ratio)

                # Create scheduler if warmup is enabled
                scheduler = None
                if actual_warmup_steps > 0:
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=actual_warmup_steps,
                        num_training_steps=total_training_steps
                    )
                    print(f"Using linear warmup scheduler: {actual_warmup_steps} warmup steps out of {total_training_steps} total")

                # Handle resume: restore optimizer and scheduler state if available
                if resume_checkpoint_path:
                    # Load optimizer state (important for momentum/Adam statistics)
                    optimizer_state_path = os.path.join(resume_checkpoint_path, "optimizer.pt")
                    if os.path.exists(optimizer_state_path):
                        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu", weights_only=True))
                        print(f"Loaded optimizer state from {optimizer_state_path}")
                    else:
                        print("No optimizer state found, starting fresh (momentum will be reset)")

                    # Load scheduler state if using warmup
                    if scheduler:
                        scheduler_state_path = os.path.join(resume_checkpoint_path, "scheduler.pt")
                        if os.path.exists(scheduler_state_path):
                            scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu", weights_only=True))
                            print(f"Loaded scheduler state from {scheduler_state_path}")
                        else:
                            # Fast-forward scheduler to current position (for checkpoints saved before this feature)
                            completed_steps = start_epoch * num_update_steps_per_epoch
                            if completed_steps > 0:
                                print(f"Fast-forwarding scheduler by {completed_steps} steps (no saved state found)")
                                for _ in range(completed_steps):
                                    scheduler.step()

                if scheduler:
                    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
                        qwen3tts.model, optimizer, train_dataloader, scheduler
                    )
                else:
                    model, optimizer, train_dataloader = accelerator.prepare(
                        qwen3tts.model, optimizer, train_dataloader
                    )
                
                model.train()

                target_speaker_embedding = None

                # Calculate total epochs for this run
                end_epoch = start_epoch + epochs
                print(f"Starting training from epoch {start_epoch + 1} to {end_epoch}...")

                # Helper function to save a training checkpoint (also inference-ready)
                def save_training_checkpoint(checkpoint_name):
                    """Save checkpoint for resuming training. Also inference-ready."""
                    ckpt_path = os.path.join(full_output_dir, checkpoint_name)
                    os.makedirs(ckpt_path, exist_ok=True)

                    # Save training weights with speaker embedding injected
                    unwrapped = accelerator.unwrap_model(model)
                    state_dict = {k: v.cpu() for k, v in unwrapped.state_dict().items()}

                    # Inject speaker embedding at index 3000 (for inference)
                    if target_speaker_embedding is not None:
                        weight = state_dict['talker.model.codec_embedding.weight']
                        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().cpu().to(weight.dtype)

                    torch.save(state_dict, os.path.join(ckpt_path, "pytorch_model.bin"))

                    # Save config for inference (speaker mapping)
                    base_cfg_path = os.path.join(init_model_path, "config.json")
                    with open(base_cfg_path, 'r', encoding='utf-8') as f:
                        ckpt_cfg = json.load(f)

                    ckpt_cfg["tts_model_type"] = "custom_voice"
                    spk_key = speaker_name.lower()
                    ckpt_cfg["talker_config"]["spk_id"] = {spk_key: 3000}
                    ckpt_cfg["talker_config"]["spk_is_dialect"] = {spk_key: False}

                    with open(os.path.join(ckpt_path, "config.json"), 'w', encoding='utf-8') as f:
                        json.dump(ckpt_cfg, f, indent=2, ensure_ascii=False)

                    if save_optimizer_state:
                        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
                        if scheduler:
                            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

                    # Save training config with step_offset for resume
                    training_config = {
                        "step_offset": resume_from_step,
                    }
                    with open(os.path.join(ckpt_path, "training_config.json"), 'w') as f:
                        json.dump(training_config, f, indent=2)

                    print(f"Training checkpoint saved: {ckpt_path}")
                    return ckpt_path

                # Helper function to save final inference-ready model
                def save_final_model(checkpoint_name):
                    """Save complete model ready for inference and resume."""
                    ckpt_path = os.path.join(full_output_dir, checkpoint_name)

                    # Copy config files only (exclude speech_tokenizer, model files)
                    def ignore_files(directory, files):
                        ignored = set()
                        if directory == init_model_path:
                            if "speech_tokenizer" in files:
                                ignored.add("speech_tokenizer")
                            if "model.safetensors" in files:
                                ignored.add("model.safetensors")
                            if "pytorch_model.bin" in files:
                                ignored.add("pytorch_model.bin")
                        return ignored

                    shutil.copytree(init_model_path, ckpt_path, ignore=ignore_files, dirs_exist_ok=True)

                    # Modify config.json for custom voice
                    ckpt_cfg_path = os.path.join(ckpt_path, "config.json")
                    with open(ckpt_cfg_path, 'r', encoding='utf-8') as f:
                        ckpt_cfg = json.load(f)

                    ckpt_cfg["tts_model_type"] = "custom_voice"
                    spk_key = speaker_name.lower()
                    ckpt_cfg["talker_config"]["spk_id"] = {spk_key: 3000}
                    ckpt_cfg["talker_config"]["spk_is_dialect"] = {spk_key: False}

                    with open(ckpt_cfg_path, 'w', encoding='utf-8') as f:
                        json.dump(ckpt_cfg, f, indent=2, ensure_ascii=False)

                    # Save weights with speaker embedding injected (keeps speaker_encoder for resume)
                    unwrapped = accelerator.unwrap_model(model)
                    state_dict = {k: v.cpu() for k, v in unwrapped.state_dict().items()}

                    # Inject speaker embedding at index 3000 (for inference)
                    if target_speaker_embedding is not None:
                        weight = state_dict['talker.model.codec_embedding.weight']
                        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().cpu().to(weight.dtype)

                    # Save as pytorch_model.bin (works for both inference and resume)
                    torch.save(state_dict, os.path.join(ckpt_path, "pytorch_model.bin"))

                    if save_optimizer_state:
                        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
                        if scheduler:
                            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

                    # Save training config with step_offset for resume
                    training_config = {
                        "step_offset": resume_from_step,
                    }
                    with open(os.path.join(ckpt_path, "training_config.json"), 'w') as f:
                        json.dump(training_config, f, indent=2)

                    print(f"Final model saved: {ckpt_path}")
                    return ckpt_path

                # Calculate total optimizer steps and global step counter
                # Use num_update_steps_per_epoch (optimizer steps) not len(train_dataloader) (micro-batches)
                total_optimizer_steps = num_update_steps_per_epoch * end_epoch + resume_from_step
                global_step = start_epoch * num_update_steps_per_epoch + resume_from_step  # Resume from correct optimizer step

                for epoch in range(start_epoch, end_epoch):
                    epoch_loss = 0
                    steps = 0
                    send_status(f"Epoch {epoch + 1}/{end_epoch} - Training...")
                    for batch in train_dataloader:
                        with accelerator.accumulate(model):
                            # Debug info (only on first batch of first epoch in this run)
                            if steps == 0 and epoch == start_epoch:
                                 print(f"DEBUG: Grad Enabled: {torch.is_grad_enabled()}")
                                 print(f"DEBUG: Inference Mode: {torch.is_inference_mode_enabled()}")
                                 for n, p in model.named_parameters():
                                     if p.requires_grad:
                                         print(f"DEBUG: Parameter {n} requires grad.")
                                         break

                            # Data extraction logic from sft_12hz.py
                            input_ids = batch['input_ids']
                            codec_ids = batch['codec_ids']
                            ref_mels = batch['ref_mels']
                            text_embedding_mask = batch['text_embedding_mask']
                            codec_embedding_mask = batch['codec_embedding_mask']
                            attention_mask = batch['attention_mask']
                            codec_0_labels = batch['codec_0_labels']
                            codec_mask = batch['codec_mask']
                            
                            # Unwrap model to access attributes (DDP/FSDP wrappers hide them)
                            unwrapped_model = accelerator.unwrap_model(model)

                            # Get device/dtype from model parameters (DDP wrappers don't expose these directly)
                            model_dtype = next(unwrapped_model.parameters()).dtype
                            model_device = next(unwrapped_model.parameters()).device
                            speaker_embedding = unwrapped_model.speaker_encoder(ref_mels.to(model_device).to(model_dtype)).detach()
                            if target_speaker_embedding is None:
                                target_speaker_embedding = speaker_embedding

                            input_text_ids = input_ids[:, :, 0]
                            input_codec_ids = input_ids[:, :, 1]

                            # Use unwrapped model for attribute access (DDP/FSDP wrappers hide them)
                            current_model = unwrapped_model
                            
                            # Debug Gradient Flow
                            if steps == 0 and epoch == start_epoch:
                                print(f"DEBUG: Model Training Mode: {current_model.training}")
                                # Check embedding layer grad
                                emb_layer = current_model.talker.model.text_embedding
                                print(f"DEBUG: Text Embedding Layer Weight requires_grad: {emb_layer.weight.requires_grad}")

                            # 0.6B model requires text_projection for dimension matching (1024 -> 2048)
                            raw_text_embedding = current_model.talker.model.text_embedding(input_text_ids)
                            if "0.6B" in init_model:
                                input_text_embedding = current_model.talker.text_projection(raw_text_embedding) * text_embedding_mask
                            else:
                                input_text_embedding = raw_text_embedding * text_embedding_mask
                            input_codec_embedding = current_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                            input_codec_embedding[:, 6, :] = speaker_embedding
                            
                            input_embeddings = input_text_embedding + input_codec_embedding
                            
                            if steps == 0 and epoch == start_epoch:
                                 print(f"DEBUG: input_text_embedding requires_grad: {input_text_embedding.requires_grad}")
                                 print(f"DEBUG: input_codec_embedding requires_grad: {input_codec_embedding.requires_grad}")
                                 print(f"DEBUG: input_embeddings requires_grad: {input_embeddings.requires_grad}")
                            
                            for i in range(1, 16):
                                codec_i_embedding = current_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                                input_embeddings = input_embeddings + codec_i_embedding
                                
                            outputs = current_model.talker(
                                inputs_embeds=input_embeddings[:, :-1, :],
                                attention_mask=attention_mask[:, :-1],
                                labels=codec_0_labels[:, 1:],
                                output_hidden_states=True
                            )
                            
                            hidden_states = outputs.hidden_states[0][-1]
                            talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                            talker_codec_ids = codec_ids[codec_mask]
                            
                            sub_talker_logits, sub_talker_loss = current_model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                            
                            loss = outputs.loss + sub_talker_loss
                            
                            if steps == 0 and epoch == start_epoch:
                                print(f"DEBUG: Loss requires_grad: {loss.requires_grad}")
                                if not loss.requires_grad:
                                    print(f"DEBUG: outputs.loss requires_grad: {outputs.loss.requires_grad if outputs.loss is not None else 'None'}")
                                    print(f"DEBUG: sub_talker_loss requires_grad: {sub_talker_loss.requires_grad}")
                            
                            accelerator.backward(loss)

                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                            optimizer.step()
                            if scheduler:
                                scheduler.step()
                            optimizer.zero_grad()

                            epoch_loss += loss.item()
                            steps += 1

                            # --- LOGICA DE TARGET LOSS ---
                            if target_loss > 0 and loss.item() <= target_loss:
                                print(f"\n🎯 [Qwen3-TTS] OBJETIVO ALCANZADO: Loss {loss.item():.4f} <= {target_loss}")
                                send_status(f"🎯 Target Loss reached: {loss.item():.4f}")
                                print(f"💾 Guardando checkpoint final y deteniendo...")

                                final_path = save_final_model(f"target_reached_loss_{loss.item():.4f}")

                                # Clean up and exit
                                accelerator.free_memory()
                                del model, optimizer, train_dataloader, qwen3tts
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()

                                print(f"Fine-tuning complete (Target Reached). Model saved to {final_path}")
                                send_status("Training complete (Target Reached)!")
                                return (final_path, speaker_name)
                            # -----------------------------

                            # Only count optimizer steps (after gradient accumulation completes)
                            if accelerator.sync_gradients:
                                global_step += 1

                                # Show step progress periodically
                                if log_every_steps > 0 and global_step % log_every_steps == 0:
                                    lr_val = optimizer.param_groups[0]['lr']
                                    status = f"Step {global_step}/{total_optimizer_steps}, Loss: {loss.item():.4f}, LR: {lr_val:.8f}"
                                    print(status)
                                    send_status(status)

                                # Step-based saving: only lightweight checkpoints during training
                                # (final model is always saved as epoch_N after training loop)
                                if save_every_steps > 0 and global_step % save_every_steps == 0:
                                    send_status(f"Saving checkpoint step {global_step}...")
                                    save_training_checkpoint(f"ckpt_step_{global_step}")

                    avg_loss = epoch_loss/steps if steps > 0 else 0
                    print(f"Epoch {epoch + 1}/{end_epoch} - Avg Loss: {avg_loss}")
                    send_status(f"Epoch {epoch + 1}/{end_epoch} - Loss: {avg_loss:.4f}")

                    # Epoch-based saving: intermediate checkpoints only (final saved after loop)
                    if save_every_steps == 0 and save_every_epochs > 0:
                        is_final_epoch = (epoch + 1) == end_epoch
                        should_save_checkpoint = ((epoch + 1) % save_every_epochs == 0) and not is_final_epoch
                        if should_save_checkpoint:
                            send_status(f"Saving checkpoint epoch {epoch + 1}...")
                            save_training_checkpoint(f"ckpt_epoch_{epoch + 1}")

                # Always save final model as epoch_N for consistent resume
                send_status(f"Saving final model epoch {end_epoch}...")
                save_final_model(f"epoch_{end_epoch}")
                final_output_path = os.path.join(full_output_dir, f"epoch_{end_epoch}")

                # Cleanup: free accelerator resources and synchronize CUDA
                accelerator.free_memory()
                del model, optimizer, train_dataloader, qwen3tts
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                print(f"Fine-tuning complete. Model saved to {final_output_path}")
                send_status("Training complete!")
                return (final_output_path, speaker_name)


class Qwen3SaveAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio", "multiline": False}),
            },
            "optional": {
                "output_subfolder": ("STRING", {"default": "Qwen3-TTS/output", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_audio"
    CATEGORY = "Qwen3-TTS/Utils"

    def save_audio(self, audio, filename_prefix, output_subfolder=""):
        # Determine output directory
        base_output = folder_paths.get_output_directory()
        if output_subfolder:
            out_dir = os.path.join(base_output, output_subfolder)
        else:
            out_dir = base_output

        os.makedirs(out_dir, exist_ok=True)

        # Handle audio input
        # ComfyUI Audio format: {"waveform": tensor[B, C, T], "sample_rate": int}
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Ensure we have a batch dimension
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]

        # Save each item in batch
        # Generate unique counter/timestamp to avoid overwrites within session if needed,
        # or rely on ComfyUI standard prefix logic.
        # For simplicity and batch handling:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(batch_size):
            # Extract single waveform: [Channels, Samples]
            wav = waveform[i]

            # Convert to numpy for soundfile
            # Soundfile expects [Samples, Channels] usually
            wav_np = wav.cpu().numpy()
            if wav_np.shape[0] < wav_np.shape[1]:
                wav_np = wav_np.transpose(1, 0) # [C, T] -> [T, C]

            file_name = f"{filename_prefix}_{timestamp}_{i:04d}.wav"
            file_path = os.path.join(out_dir, file_name)

            sf.write(file_path, wav_np, sample_rate)
            print(f"[Qwen3-TTS] Saved audio to: {file_path}")

        return (out_dir,)


class Qwen3LoadAudioFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"
    CATEGORY = "Qwen3-TTS/Utils"

    def load_audio(self, audio_path):
        print(f"[Qwen3-TTS DEBUG] LoadAudioFromPath [RAW]: {audio_path}")
        audio_path = fix_wsl_path(audio_path)
        print(f"[Qwen3-TTS DEBUG] LoadAudioFromPath [FIXED]: {audio_path}")

        if not audio_path or not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Use soundfile to load
        wav, sr = sf.read(audio_path)

        # Convert to float32 if needed
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        return (convert_audio(wav, sr),)


class Qwen3LoadAudioFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_folder"
    CATEGORY = "Qwen3-TTS/Utils"

    def load_folder(self, folder_path):
        folder_path = fix_wsl_path(folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))])

        if not files:
            raise ValueError(f"No audio files found in {folder_path}")

        print(f"[Qwen3-TTS] Loading {len(files)} audio files from {folder_path}")

        output_list = []

        # Helper to process single file
        def process_wav(w):
            if w.dtype != np.float32: w = w.astype(np.float32)
            if w.ndim == 1: w = w[np.newaxis, :] # (1, samples)
            if w.shape[0] > w.shape[1]: w = w.T # Ensure (channels, samples)
            return torch.from_numpy(w).unsqueeze(0) # Add batch dim: [1, C, T]

        for fname in files:
            p = os.path.join(folder_path, fname)
            try:
                w, s = sf.read(p)
                wt = process_wav(w)
                output_list.append({"waveform": wt, "sample_rate": s})
            except Exception as e:
                print(f"Error loading {fname}: {e}")

        if not output_list:
            raise ValueError("Failed to load any valid audio files.")

        return (output_list,)


class Qwen3LoadVideoFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_video"
    CATEGORY = "Qwen3-TTS/Utils"

    def load_video(self, video_path):
        if not HAS_WHISPER_PYDUB:
             raise ImportError("Please install 'pydub' (and ffmpeg) to use video loading nodes.")

        video_path = fix_wsl_path(video_path)
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        try:
            audio = AudioSegment.from_file(video_path)
            # Convert to float32 numpy
            samples = np.array(audio.get_array_of_samples())
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels)).T
            else:
                samples = samples.reshape((1, -1))

            # Normalize to -1.0 ... 1.0 (pydub gives int)
            samples = samples.astype(np.float32) / (1 << (8 * audio.sample_width - 1))

            return (convert_audio(samples, audio.frame_rate),)

        except Exception as e:
            raise RuntimeError(f"Error extracting audio from video: {e}")


class Qwen3LoadVideoFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_folder"
    CATEGORY = "Qwen3-TTS/Utils"

    def load_folder(self, folder_path):
        if not HAS_WHISPER_PYDUB:
             raise ImportError("Please install 'pydub' (and ffmpeg) to use video loading nodes.")

        folder_path = fix_wsl_path(folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm'))])

        if not files:
            raise ValueError(f"No video files found in {folder_path}")

        output_list = []

        print(f"[Qwen3-TTS] Processing {len(files)} video files...")

        for fname in files:
            p = os.path.join(folder_path, fname)
            try:
                audio = AudioSegment.from_file(p)

                samples = np.array(audio.get_array_of_samples())
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels)).T
                else:
                    samples = samples.reshape((1, -1)) # (Channels, Samples)

                samples = samples.astype(np.float32) / (1 << (8 * audio.sample_width - 1))
                samples_tensor = torch.from_numpy(samples).unsqueeze(0) # [1, C, T]

                output_list.append({"waveform": samples_tensor, "sample_rate": audio.frame_rate})

            except Exception as e:
                print(f"Error processing {fname}: {e}")

        if not output_list:
            raise ValueError("Failed to process any video files.")

        return (output_list,)


class Qwen3AudioCompare:
    # Class-level cache for speaker encoder
    _speaker_encoder = None
    _speaker_encoder_cache_key = None

    @classmethod
    def INPUT_TYPES(s):
        # Get available Base models
        base_models = [k for k in QWEN3_TTS_MODELS.keys() if "Base" in k]
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "generated_audio": ("AUDIO",),
                "speaker_encoder_model": (base_models, {"default": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "tooltip": "Base model to load speaker encoder from (only loads ~76 weights, not the full model)"}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "", "tooltip": "Optional custom path to model directory. If empty, uses default models/Qwen3-TTS/ location."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "Qwen3-TTS/Utils"

    def _load_speaker_encoder(self, model_repo, local_model_path=""):
        """Load only the speaker encoder from a Base model (not the full model)."""
        from safetensors import safe_open
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

        # Get local model path - use provided path if non-empty, otherwise fall back to default
        if local_model_path and local_model_path.strip():
            model_path = os.path.abspath(local_model_path.strip())
        else:
            model_path = get_local_model_path(model_repo)

        # Check if already cached (use resolved path as cache key)
        if Qwen3AudioCompare._speaker_encoder is not None and Qwen3AudioCompare._speaker_encoder_cache_key == model_path:
            return Qwen3AudioCompare._speaker_encoder
        if not os.path.exists(model_path):
            raise ValueError(f"Base model not found at {model_path}. Please download it first using Qwen3-TTS Loader.")

        # Load config to get speaker encoder config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Validate this is a Base model with speaker encoder
        if "speaker_encoder_config" not in config_dict:
            raise ValueError(f"Model at {model_path} does not contain speaker_encoder_config. Only Base models (e.g., Qwen3-TTS-12Hz-0.6B-Base) include the speaker encoder.")

        # Create speaker encoder config
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        speaker_config = Qwen3TTSSpeakerEncoderConfig(**config_dict["speaker_encoder_config"])

        # Instantiate speaker encoder
        speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config)

        # Load only speaker encoder weights from safetensors (selective loading to save memory)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        speaker_weights = {}
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("speaker_encoder."):
                    speaker_weights[key[len("speaker_encoder."):]] = f.get_tensor(key)

        speaker_encoder.load_state_dict(speaker_weights)
        speaker_encoder.eval()

        # Move to GPU if available
        device = mm.get_torch_device()
        speaker_encoder = speaker_encoder.to(device)

        # Cache it (use resolved path as key)
        Qwen3AudioCompare._speaker_encoder = speaker_encoder
        Qwen3AudioCompare._speaker_encoder_cache_key = model_path

        print(f"Loaded speaker encoder from {model_repo} ({len(speaker_weights)} weights)")
        return speaker_encoder

    def _extract_speaker_embedding(self, speaker_encoder, audio, sr):
        """Extract speaker embedding from audio using the speaker encoder."""
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        # Resample to 24kHz if needed
        if sr != 24000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=24000)

        # Compute mel spectrogram
        mel = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000
        ).transpose(1, 2)

        # Get embedding
        device = next(speaker_encoder.parameters()).device
        mel = mel.to(device)
        with torch.no_grad():
            embedding = speaker_encoder(mel)
        return embedding

    def compare(self, reference_audio, generated_audio, speaker_encoder_model, local_model_path=""):
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        # Extract waveforms from ComfyUI audio format
        def extract_wav(audio_input):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
            wav = waveform[0]  # Take first batch
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0)  # Mix to mono
            else:
                wav = wav.squeeze(0)
            return wav.numpy(), sr

        ref_wav, ref_sr = extract_wav(reference_audio)
        gen_wav, gen_sr = extract_wav(generated_audio)

        # 1. Speaker Similarity using speaker encoder from Base model
        speaker_encoder = self._load_speaker_encoder(speaker_encoder_model, local_model_path)

        ref_emb = self._extract_speaker_embedding(speaker_encoder, ref_wav, ref_sr)
        gen_emb = self._extract_speaker_embedding(speaker_encoder, gen_wav, gen_sr)

        speaker_sim = torch.nn.functional.cosine_similarity(
            ref_emb.flatten().unsqueeze(0),
            gen_emb.flatten().unsqueeze(0)
        ).item()

        # 2. Mel Spectrogram Distance
        target_sr = 24000
        if ref_sr != target_sr:
            ref_wav_mel = librosa.resample(ref_wav.astype(np.float32), orig_sr=ref_sr, target_sr=target_sr)
        else:
            ref_wav_mel = ref_wav
        if gen_sr != target_sr:
            gen_wav_mel = librosa.resample(gen_wav.astype(np.float32), orig_sr=gen_sr, target_sr=target_sr)
        else:
            gen_wav_mel = gen_wav

        with torch.no_grad():
            ref_mel = mel_spectrogram(
                torch.from_numpy(ref_wav_mel).unsqueeze(0),
                n_fft=1024, num_mels=128, sampling_rate=target_sr,
                hop_size=256, win_size=1024, fmin=0, fmax=12000
            )
            gen_mel = mel_spectrogram(
                torch.from_numpy(gen_wav_mel).unsqueeze(0),
                n_fft=1024, num_mels=128, sampling_rate=target_sr,
                hop_size=256, win_size=1024, fmin=0, fmax=12000
            )

            min_len = min(ref_mel.shape[-1], gen_mel.shape[-1])
            ref_mel = ref_mel[..., :min_len]
            gen_mel = gen_mel[..., :min_len]
            mel_mse = torch.nn.functional.mse_loss(ref_mel, gen_mel).item()

        # Determine quality rating
        if speaker_sim > 0.85:
            rating = "Excellent voice match"
        elif speaker_sim > 0.75:
            rating = "Good voice match"
        elif speaker_sim > 0.65:
            rating = "Moderate voice match"
        else:
            rating = "Poor voice match"

        # Calculate speaking rate
        ref_duration = len(ref_wav) / ref_sr
        gen_duration = len(gen_wav) / gen_sr
        rate_ratio = ref_duration / gen_duration

        if rate_ratio > 1.05:
            rate_desc = f"generated is {((rate_ratio - 1) * 100):.0f}% faster"
        elif rate_ratio < 0.95:
            rate_desc = f"generated is {((1 - rate_ratio) * 100):.0f}% slower"
        else:
            rate_desc = "similar pace"

        # Build report
        report = f"""Audio Comparison Report
========================
Speaker Similarity: {speaker_sim:.4f} (0-1, higher=better)
Mel Distance (MSE): {mel_mse:.6f} (lower=better)
Speaking Rate: {rate_ratio:.2f}x ({rate_desc})
Rating: {rating}

Interpretation Guide:
- Speaker Sim > 0.85: Excellent voice match
- Speaker Sim > 0.75: Good voice match
- Speaker Sim > 0.65: Moderate voice match
- Speaker Sim < 0.65: Poor voice match
- Speaking Rate ~1.0x: Ideal pacing match

Audio Details:
- Reference duration: {ref_duration:.2f}s
- Generated duration: {gen_duration:.2f}s"""

        print(report)
        return (report,)

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3Loader": Qwen3Loader,
    "Qwen3LoadFineTuned": Qwen3LoadFineTuned,
    "Qwen3CustomVoice": Qwen3CustomVoice,
    "Qwen3VoiceDesign": Qwen3VoiceDesign,
    "Qwen3PromptMaker": Qwen3PromptMaker,
    "Qwen3SavePrompt": Qwen3SavePrompt,
    "Qwen3LoadPrompt": Qwen3LoadPrompt,
    "Qwen3VoiceClone": Qwen3VoiceClone,
    "Qwen3LoadDatasetAudio": Qwen3LoadDatasetAudio,
    "Qwen3TranscribeWhisper": Qwen3TranscribeWhisper,
    "Qwen3AutoLabelEmotions": Qwen3AutoLabelEmotions,
    "Qwen3ExportJSONL": Qwen3ExportJSONL,
    "Qwen3DataPrep": Qwen3DataPrep,
    "Qwen3FineTune": Qwen3FineTune,
    "Qwen3SaveAudio": Qwen3SaveAudio,
    "Qwen3LoadAudioFromPath": Qwen3LoadAudioFromPath,
    "Qwen3LoadAudioFolder": Qwen3LoadAudioFolder,
    "Qwen3LoadVideoFromPath": Qwen3LoadVideoFromPath,
    "Qwen3LoadVideoFolder": Qwen3LoadVideoFolder,
    "Qwen3AudioCompare": Qwen3AudioCompare,
    "Qwen3AudioToDataset": Qwen3AudioToDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3Loader": "🎙️ Qwen3-TTS Loader",
    "Qwen3LoadFineTuned": "🎓 Qwen3-TTS Load Fine-Tuned",
    "Qwen3CustomVoice": "🗣️ Qwen3-TTS Custom Voice",
    "Qwen3VoiceDesign": "🎨 Qwen3-TTS Voice Design (Advanced)",
    "Qwen3PromptMaker": "📝 Qwen3-TTS Prompt Maker",
    "Qwen3SavePrompt": "💾 Qwen3-TTS Save Prompt",
    "Qwen3LoadPrompt": "📂 Qwen3-TTS Load Prompt",
    "Qwen3VoiceClone": "👥 Qwen3-TTS Voice Clone",
    "Qwen3LoadDatasetAudio": "📁 Qwen3-TTS Step 1: Load Audio Folder",
    "Qwen3TranscribeWhisper": "🎙️ Qwen3-TTS Step 2: Transcribe (Whisper)",
    "Qwen3AutoLabelEmotions": "🎭 Qwen3-TTS Step 3: Label Emotions (Qwen2-Audio)",
    "Qwen3ExportJSONL": "💾 Qwen3-TTS Step 4: Export JSONL",
    "Qwen3DataPrep": "⚙️ Qwen3-TTS Data Prep",
    "Qwen3FineTune": "🎓 Qwen3-TTS Fine-Tune",
    "Qwen3SaveAudio": "📁 Qwen3-TTS Save Audio",
    "Qwen3LoadAudioFromPath": "📁 Qwen3-TTS Load Audio (Path)",
    "Qwen3LoadAudioFolder": "📁 Qwen3-TTS Load Audio Folder (Path)",
    "Qwen3LoadVideoFromPath": "🎥 Qwen3-TTS Load Video (Path)",
    "Qwen3LoadVideoFolder": "🎥 Qwen3-TTS Load Video Folder (Path)",
    "Qwen3AudioCompare": "📊 Qwen3-TTS Audio Compare",
    "Qwen3AudioToDataset": "📁 Qwen3-TTS Dataset Maker",
}
