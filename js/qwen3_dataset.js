import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Qwen3.DatasetPicker",
    async nodeCreated(node, app) {
        // Definitions of which nodes and widgets get pickers
        // Format: NodeClass: [{ widget: "widget_name", type: "folder" | "file" }]
        const config = {
            "Qwen3LoadDatasetAudio": [{ widget: "folder_path", type: "folder" }],
            "Qwen3LoadAudioFromPath": [{ widget: "audio_path", type: "file" }],
            "Qwen3FineTune": [{ widget: "output_dir", type: "folder" }],
            "Qwen3TrainLoRA": [{ widget: "save_path", type: "folder" }],
            "Qwen3DataPrep": [{ widget: "jsonl_path", type: "file" }],
            "Qwen3ApplyLoRA": [{ widget: "lora_path", type: "folder" }]
        };

        if (config[node.comfyClass]) {
            const targets = config[node.comfyClass];

            targets.forEach(target => {
                const widget = node.widgets.find(w => w.name === target.widget);
                if (widget) {
                    const isFolder = target.type === "folder";
                    const btnLabel = isFolder ? "📁 Select Folder" : "🎵 Select File";

                    node.addWidget("button", btnLabel, null, () => {
                        showFilePicker(widget, isFolder);
                    });
                }
            });
        }
    }
});

// Helper function to show picker
async function showFilePicker(targetWidget, dirOnly) {
    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        position: "fixed", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
        width: "600px", height: "600px", backgroundColor: "#222", color: "#ddd",
        zIndex: "10000", padding: "20px", borderRadius: "8px", border: "1px solid #555",
        display: "flex", flexDirection: "column", fontFamily: "sans-serif",
        boxShadow: "0 0 20px rgba(0,0,0,0.5)"
    });

    const header = document.createElement("div");
    header.style.marginBottom = "10px";
    header.style.display = "flex";
    header.style.justifyContent = "space-between";
    header.style.alignItems = "center";

    const title = document.createElement("h3");
    title.style.margin = "0";
    title.innerText = `Select ${dirOnly ? 'Folder' : 'File'}`;

    const closeBtn = document.createElement("button");
    closeBtn.innerText = "✖";
    closeBtn.style.background = "transparent";
    closeBtn.style.border = "none";
    closeBtn.style.color = "#888";
    closeBtn.style.cursor = "pointer";
    closeBtn.style.fontSize = "16px";
    closeBtn.onclick = () => document.body.removeChild(dialog);

    header.appendChild(title);
    header.appendChild(closeBtn);

    const pathDisplay = document.createElement("div");
    pathDisplay.style.marginBottom = "10px";
    pathDisplay.style.padding = "8px";
    pathDisplay.style.backgroundColor = "#333";
    pathDisplay.style.borderRadius = "4px";
    pathDisplay.style.fontFamily = "monospace";
    pathDisplay.style.fontSize = "14px";
    pathDisplay.style.wordBreak = "break-all";

    const listContainer = document.createElement("div");
    Object.assign(listContainer.style, {
        flex: "1", overflowY: "auto", border: "1px solid #444",
        backgroundColor: "#111", borderRadius: "4px", padding: "5px"
    });

    const footer = document.createElement("div");
    footer.style.marginTop = "15px";
    footer.style.display = "flex";
    footer.style.justifyContent = "flex-end";
    footer.style.gap = "10px";

    const cancelButton = document.createElement("button");
    cancelButton.innerText = "Cancel";
    cancelButton.style.padding = "5px 15px";
    cancelButton.onclick = () => document.body.removeChild(dialog);

    const selectButton = document.createElement("button");
    selectButton.innerText = "Select Current Folder";
    selectButton.style.padding = "5px 15px";
    selectButton.style.backgroundColor = "#2a6";
    selectButton.style.border = "none";
    selectButton.style.color = "white";
    selectButton.style.cursor = "pointer";
    selectButton.style.borderRadius = "3px";

    // Only show "Select Current Folder" if dirOnly is true
    if (!dirOnly) {
        selectButton.style.display = "none";
    }

    footer.appendChild(cancelButton);
    footer.appendChild(selectButton);

    dialog.appendChild(header);
    dialog.appendChild(pathDisplay);
    dialog.appendChild(listContainer);
    dialog.appendChild(footer);
    document.body.appendChild(dialog);

    let currentPath = targetWidget.value || ".";

    async function loadPath(path) {
        try {
            const response = await api.fetchApi(`/qwen3/list_dirs?path=${encodeURIComponent(path)}`);
            const data = await response.json();

            if (data.error) {
                // If path invalid, try root
                if (path !== ".") {
                    loadPath(".");
                } else {
                    alert(data.error);
                }
                return;
            }

            currentPath = data.current_path;
            pathDisplay.innerText = currentPath;
            listContainer.innerHTML = "";

            data.items.forEach(item => {
                const div = document.createElement("div");
                div.style.padding = "6px 8px";
                div.style.cursor = "pointer";
                div.style.borderBottom = "1px solid #222";
                div.style.display = "flex";
                div.style.alignItems = "center";
                div.style.fontSize = "14px";

                // Icon
                const icon = document.createElement("span");
                icon.style.marginRight = "8px";
                icon.innerText = item.is_dir ? "📁" : "📄";
                div.appendChild(icon);

                // Name
                const nameSpan = document.createElement("span");
                nameSpan.innerText = item.name;
                div.appendChild(nameSpan);

                // Styling for disabled items (files when in dirOnly mode)
                if (dirOnly && !item.is_dir) {
                    div.style.opacity = "0.5";
                    div.style.cursor = "default";
                } else {
                    div.onmouseover = () => div.style.backgroundColor = "#333";
                    div.onmouseout = () => div.style.backgroundColor = "transparent";

                    div.onclick = () => {
                        if (item.is_dir) {
                            loadPath(item.path);
                        } else if (!dirOnly) {
                            // Select file
                            targetWidget.value = item.path;
                            document.body.removeChild(dialog);
                        }
                    };
                }

                listContainer.appendChild(div);
            });

            // Update select button action
            selectButton.onclick = () => {
                targetWidget.value = currentPath;
                document.body.removeChild(dialog);
            };

        } catch (err) {
            console.error(err);
            alert("Error fetching path: " + err.message);
        }
    }

    loadPath(currentPath);
}
