import os
import json
from aiohttp import web
from server import PromptServer

from .nodes import (
    Qwen3Loader,
    Qwen3CustomVoice,
    Qwen3VoiceDesign,
    Qwen3VoiceClone,
    Qwen3PromptMaker,
    Qwen3SavePrompt,
    Qwen3LoadPrompt,
    Qwen3DatasetFromFolder,
    Qwen3DataPrep,
    Qwen3FineTune,
    Qwen3AudioCompare,
    Qwen3LoadAudioFromPath
)

NODE_CLASS_MAPPINGS = {
    "Qwen3Loader": Qwen3Loader,
    "Qwen3CustomVoice": Qwen3CustomVoice,
    "Qwen3VoiceDesign": Qwen3VoiceDesign,
    "Qwen3VoiceClone": Qwen3VoiceClone,
    "Qwen3PromptMaker": Qwen3PromptMaker,
    "Qwen3SavePrompt": Qwen3SavePrompt,
    "Qwen3LoadPrompt": Qwen3LoadPrompt,
    "Qwen3DatasetFromFolder": Qwen3DatasetFromFolder,
    "Qwen3DataPrep": Qwen3DataPrep,
    "Qwen3FineTune": Qwen3FineTune,
    "Qwen3AudioCompare": Qwen3AudioCompare,
    "Qwen3LoadAudioFromPath": Qwen3LoadAudioFromPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3Loader": "🎙️ Qwen3-TTS Loader",
    "Qwen3CustomVoice": "🗣️ Qwen3-TTS Custom Voice",
    "Qwen3VoiceDesign": "🎨 Qwen3-TTS Voice Design",
    "Qwen3VoiceClone": "👥 Qwen3-TTS Voice Clone",
    "Qwen3PromptMaker": "📝 Qwen3-TTS Prompt Maker",
    "Qwen3SavePrompt": "💾 Qwen3-TTS Save Prompt",
    "Qwen3LoadPrompt": "📂 Qwen3-TTS Load Prompt",
    "Qwen3DatasetFromFolder": "📁 Qwen3-TTS Dataset Maker",
    "Qwen3DataPrep": "⚙️ Qwen3-TTS Data Prep",
    "Qwen3FineTune": "🎓 Qwen3-TTS Finetune",
    "Qwen3AudioCompare": "📊 Qwen3-TTS Audio Compare",
    "Qwen3LoadAudioFromPath": "📁 Qwen3-TTS Load Audio (Path)"
}

WEB_DIRECTORY = "./js"

@PromptServer.instance.routes.get("/qwen3/list_dirs")
async def list_dirs(request):
    try:
        path = request.query.get("path", ".")
        if not os.path.exists(path):
            # If path doesn't exist, try to start from root or home
            if path == ".":
                path = os.getcwd()
            else:
                 return web.json_response({"error": "Path not found"}, status=404)

        # Resolve to absolute path
        abs_path = os.path.abspath(path)

        # Security check: (Optional) restrict to specific drives?
        # For now, we assume user trusts the local ComfyUI instance.

        parent_path = os.path.dirname(abs_path)

        dirs = []
        files = []

        # Add parent directory entry if not at root
        if parent_path != abs_path:
            dirs.append({
                "name": "..",
                "path": parent_path,
                "is_dir": True
            })

        for item in os.listdir(abs_path):
            # Skip hidden files
            if item.startswith("."):
                continue

            item_path = os.path.join(abs_path, item)
            is_dir = os.path.isdir(item_path)

            entry = {
                "name": item,
                "path": item_path,
                "is_dir": is_dir
            }

            if is_dir:
                dirs.append(entry)
            else:
                files.append(entry)

        # Sort: directories first (alphabetical), then files (alphabetical)
        dirs.sort(key=lambda x: x["name"].lower())
        files.sort(key=lambda x: x["name"].lower())

        return web.json_response({
            "current_path": abs_path,
            "items": dirs + files
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
