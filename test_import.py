import sys
from unittest.mock import MagicMock

# Mock ComfyUI modules
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["server"] = MagicMock()
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].models_dir = "/tmp/models"
sys.modules["folder_paths"].base_path = "/tmp"
sys.modules["folder_paths"].get_output_directory = MagicMock(return_value="/tmp/output")
sys.modules["folder_paths"].get_temp_directory = MagicMock(return_value="/tmp/temp")

# Mock other deps if needed
sys.modules["qwen_tts"] = MagicMock()
sys.modules["qwen_asr"] = MagicMock()

try:
    import nodes
    print("Successfully imported nodes.py")
except Exception as e:
    print(f"Failed to import nodes.py: {e}")
