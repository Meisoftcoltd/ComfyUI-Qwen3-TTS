
import sys
import os

# Mock ComfyUI environment
sys.path.append(os.getcwd())
class MockFolderPaths:
    models_dir = "models"
    def get_output_directory(self): return "output"
    def add_model_folder_path(self, *args): pass

class MockMM:
    def get_torch_device(self): return "cpu"
    def cpu_mode(self): return True

import builtins
# Mock module imports that might be missing in this environment
# (ComfyUI specific or huge libraries I don't want to load fully if checking syntax)
# Actually, let's try to import nodes.py directly. If dependencies are missing, I'll mock them.
# The environment seems to have torch and basic stuff.

# Mocking ComfyUI modules
import types
folder_paths = MockFolderPaths()
sys.modules["folder_paths"] = folder_paths
sys.modules["comfy"] = types.ModuleType("comfy")
sys.modules["comfy.model_management"] = MockMM()
sys.modules["server"] = types.ModuleType("server")
sys.modules["server"].PromptServer = type("PromptServer", (), {"instance": type("Instance", (), {"send_progress": lambda *args: None, "send_progress_text": lambda *args: None})})

# Create dummy directories to prevent FileNotFoundError during import if it scans
os.makedirs("models/tts/finetuned_model", exist_ok=True)
os.makedirs("models/Qwen3-TTS/finetuned_model", exist_ok=True)
os.makedirs("models/tts/prompts", exist_ok=True)

try:
    import nodes
    print("Successfully imported nodes.py")

    # Check if classes exist
    print(f"Qwen3FineTune: {nodes.Qwen3FineTune}")
    print(f"Qwen3LoadFineTuned: {nodes.Qwen3LoadFineTuned}")

    # Check INPUT_TYPES of modified node
    inputs = nodes.Qwen3LoadFineTuned.INPUT_TYPES()
    print("Qwen3LoadFineTuned INPUT_TYPES keys:", inputs["required"].keys())

    # Verify speaker scanning works (should be empty but valid)
    print("Speakers:", inputs["required"]["speaker_name"][0])
    print("Versions:", inputs["required"]["version"][0])

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
