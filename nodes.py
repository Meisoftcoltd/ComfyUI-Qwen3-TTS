import os
import torch
import soundfile as sf
import numpy as np
import folder_paths
from qwen_tts import Qwen3TTSModel

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
    if wav.shape[0] > wav.shape[1]: 
        # assume (samples, channels) - verify this assumption
        wav = wav.transpose(0, 1)
        
    # If it's just (samples,), we made it (1, samples). 
    # ComfyUI often expects [Batch, Channels, Samples]. 
    # Let's wrap in batch.
    wav = wav.unsqueeze(0) # (1, channels, samples)
    
    return {"waveform": wav, "sample_rate": sr}

def load_audio_input(audio_input):
    # audio_input is {"waveform": tensor, "sample_rate": int}
    # waveform is [batch, channels, samples]
    # We need (samples,) or (channels, samples) numpy for Qwen?
    # Qwen accepts numpy array.
    
    if audio_input is None:
        return None
        
    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    
    # Take first batch item
    wav = waveform[0] # (channels, samples)
    
    # If multi-channel, maybe mix down or take first?
    # For cloning, mono is usually fine.
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0) # Mix to mono
    else:
        wav = wav.squeeze(0) # (samples,)
        
    return (wav.numpy(), sr)


class Qwen3Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                ], {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-TTS"

    def load_model(self, repo_id, source, precision, attention):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        model_path = repo_id
        if source == "ModelScope":
            from modelscope import snapshot_download
            try:
                model_path = snapshot_download(repo_id)
            except Exception as e:
                print(f"ModelScope download failed: {e}. Falling back to HF or local.")

        print(f"Loading Qwen3-TTS model: {repo_id} from {model_path} on {device} as {dtype}")
        
        # Determine attention implementation
        attn_impl = "sdpa" # Default to sdpa (torch 2.0+) as it's usually available and fast
        
        if attention != "auto":
            attn_impl = attention
        else:
            # Auto-detect
            try:
                import flash_attn
                # Also check version metadata as transformers works 
                import importlib.metadata
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except Exception:
                # Fallback to sdpa if flash_attn missing or metadata broken
                attn_impl = "sdpa"

        print(f"Using attention implementation: {attn_impl}")

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )
        
        return (model,)


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
                "speaker": ([
                    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", 
                    "Ryan", "Aiden", "Ono_Anna", "Sohee"
                ], {"default": "Vivian"}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, language, speaker, instruct=""):
        lang = language if language != "Auto" else None
        inst = instruct if instruct.strip() != "" else None
        
        # Ensure model is compatible? CustomVoice models only?
        # We'll just try and let it fail if wrong model.
        
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=speaker,
            instruct=inst
        )
        # wavs is a list of numpy arrays (batch size 1 here)
        return (convert_audio(wavs[0], sr),)


class Qwen3VoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "instruct": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, instruct, language):
        lang = language if language != "Auto" else None
        
        wavs, sr = model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct
        )
        return (convert_audio(wavs[0], sr),)


class Qwen3PromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"

    def create_prompt(self, model, ref_audio, ref_text):
        audio_tuple = load_audio_input(ref_audio)
        
        prompt = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text
        )
        return (prompt,)


class Qwen3VoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
                "prompt": ("QWEN3_PROMPT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, language="Auto", ref_audio=None, ref_text=None, prompt=None):
        lang = language if language != "Auto" else None
        
        wavs = None
        sr = 0
        
        if prompt is not None:
            # Use pre-calculated prompt
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=prompt
            )
        elif ref_audio is not None and ref_text is not None and ref_text.strip() != "":
            # Use on-the-fly prompt creation
            audio_tuple = load_audio_input(ref_audio)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=audio_tuple,
                ref_text=ref_text
            )
        else:
             raise ValueError("For Voice Clone, you must provide either 'prompt' OR ('ref_audio' AND 'ref_text').")
             
        return (convert_audio(wavs[0], sr),)
