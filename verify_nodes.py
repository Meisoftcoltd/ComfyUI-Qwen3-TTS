
import sys
import os
from unittest.mock import MagicMock

# Mock ComfyUI environment
sys.path.append(os.getcwd())

# Mock modules BEFORE importing nodes
module_names = [
    "torch", "torch.utils", "torch.utils.data", "torch.optim", "torch.nn",
    "soundfile", "numpy", "librosa", "transformers", "transformers.utils", "accelerate",
    "whisper", "pydub", "bitsandbytes", "safetensors", "safetensors.torch",
    "tqdm", "qwen_tts", "qwen_tts.inference", "qwen_tts.inference.qwen3_tts_model",
    "qwen_tts.core", "qwen_tts.core.models",
    "qwen_tts.core.models.configuration_qwen3_tts",
    "qwen_tts.core.models.modeling_qwen3_tts"
]

for m in module_names:
    sys.modules[m] = MagicMock()

# Setup specific mocks that are accessed as attributes or types
sys.modules["torch"].float32 = "float32"
sys.modules["torch"].float16 = "float16"
sys.modules["torch"].bfloat16 = "bfloat16"
sys.modules["torch"].utils = MagicMock()
sys.modules["torch"].utils.data = MagicMock()
sys.modules["torch"].utils.data.Dataset = object # Used in subclassing if any (actually in nodes.py logic, not top level class definition usually)

# nodes.py imports:
# from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
# from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
sys.modules["qwen_tts"].Qwen3TTSModel = MagicMock()
sys.modules["qwen_tts"].Qwen3TTSTokenizer = MagicMock()
sys.modules["qwen_tts"].inference.qwen3_tts_model.VoiceClonePromptItem = MagicMock()

# from .dataset import TTSDataset
# We need to mock .dataset or create a dummy dataset.py if it exists (it does)
# But since we are running verify_nodes.py in root, 'from .dataset' might fail if nodes.py is imported as 'nodes' (top level).
# Wait, 'import nodes' treats it as module.
# If nodes.py does `from .dataset import ...`, it expects to be in a package.
# But here it is top level file?
# In ComfyUI custom nodes are usually loaded as a package or file.
# If I import nodes.py, `from .dataset` might fail if it's not a package.
# Let's see if we can mock `nodes.dataset`.

# Actually, I can just create a dummy dataset.py if I need to, or mock it.
# sys.modules["nodes.dataset"] = MagicMock() ? No, inside nodes.py it is relative.

# Let's rely on dataset.py existing (it does in file list).
# But if I mock torch, dataset.py might fail if it imports torch.
# So I need to ensure dataset.py imports use the mocked torch.
# Since I mocked sys.modules["torch"], any subsequent import should use it.

class MockFolderPaths:
    models_dir = "models"
    base_path = "."
    def get_output_directory(self): return "output"
    def add_model_folder_path(self, *args): pass

class MockMM:
    def get_torch_device(self): return "cpu"
    def cpu_mode(self): return True

import types
folder_paths = MockFolderPaths()
sys.modules["folder_paths"] = folder_paths
sys.modules["comfy"] = types.ModuleType("comfy")
sys.modules["comfy.model_management"] = MockMM()
sys.modules["server"] = types.ModuleType("server")
sys.modules["server"].PromptServer = type("PromptServer", (), {"instance": type("Instance", (), {"send_progress": lambda *args: None, "send_progress_text": lambda *args: None, "routes": types.SimpleNamespace(get=lambda x: lambda y: y)})})

# Create dummy directories
os.makedirs("models/tts/finetuned_model", exist_ok=True)
os.makedirs("models/Qwen3-TTS/finetuned_model", exist_ok=True)
os.makedirs("models/tts/prompts", exist_ok=True)

try:
    import nodes
    print("Successfully imported nodes.py")

    # Check Qwen3FineTune INPUT_TYPES for target_loss
    ft_inputs = nodes.Qwen3FineTune.INPUT_TYPES()
    print("Qwen3FineTune INPUT_TYPES keys:", ft_inputs["required"].keys())

    if "target_loss" in ft_inputs["required"]:
        print("✅ target_loss found in Qwen3FineTune")
    else:
        print("❌ target_loss NOT found in Qwen3FineTune")
        exit(1)

    # Basic Check of train signature
    import inspect
    sig = inspect.signature(nodes.Qwen3FineTune.train)
    print("Qwen3FineTune.train signature parameters:", sig.parameters.keys())
    if "target_loss" in sig.parameters:
        print("✅ target_loss found in train signature")
    else:
        print("❌ target_loss NOT found in train signature")
        exit(1)

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
