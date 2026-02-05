# ComfyUI-Qwen3-TTS (Español)

Nodos personalizados para [Qwen2.5-Audio / Qwen3-TTS](https://huggingface.co/Qwen/Qwen2.5-Audio-Instruct), un potente modelo de audio multimodal capaz de Text-to-Speech (TTS), Clonación de Voz y Diseño de Voz.

## Características

*   **🎙️ Texto a Voz (TTS):** Genera voz de alta calidad a partir de texto en múltiples idiomas.
*   **👥 Clonación de Voz:** Clona voces a partir de un clip de audio de referencia corto (se recomiendan 3-10s).
*   **🎨 Diseño de Voz:** Diseña voces personalizadas describiendo atributos como género, edad, tono, velocidad y emoción.
*   **🎓 Fine-Tuning y LoRA:** Flujo completo para realizar fine-tuning o entrenar adaptadores LoRA ligeros con tu propio dataset de voz.
*   **📁 Pipeline Modular de Dataset:** Automatiza la creación de datasets: Cargar audio crudo -> Transcribir con Whisper -> Etiquetar emociones con Qwen2-Audio -> Exportar JSONL.
*   **⚙️ Configuración Avanzada:** Solución para errores de "Unsupported speakers" en modelos fine-tuned y control detallado de prompts.

## Instalación

1.  **Instala ComfyUI** (si no lo tienes ya).
2.  Clona este repositorio en tu carpeta `ComfyUI/custom_nodes/`:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/your-repo/ComfyUI-Qwen3-TTS.git
    cd ComfyUI-Qwen3-TTS
    ```
3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Las funciones de entrenamiento requieren `peft`, `bitsandbytes` y `accelerate`. La creación de datasets requiere `openai-whisper` y `pydub`.*

## Resumen de Nodos

### 🎙️ Inferencia
*   **Qwen3Loader:** Carga el modelo base (ej. `Qwen/Qwen3-TTS-12Hz-1.7B-Base`).
*   **Qwen3LoadFineTuned:** Carga un modelo fine-tuned (checkpoint completo) e inyecta la configuración de hablante personalizada necesaria para la inferencia.
*   **Qwen3ApplyLoRA:** Carga un adaptador LoRA (carpeta `.safetensors`) y lo aplica a un modelo base.
*   **Qwen3VoiceDesign:** Genera voz basada en texto y un conjunto de parámetros opcionales (Género, Tono, Emoción, etc.).
*   **Qwen3VoiceClone:** Genera voz clonando un audio de referencia.

### 📁 Creación de Dataset (Pipeline Modular)
1.  **Qwen3LoadDatasetAudio:** Escanea una carpeta en busca de archivos `.wav`.
2.  **Qwen3TranscribeWhisper:** Transcribe audio usando Whisper, recorta silencios y divide archivos largos. (Requiere `openai-whisper`).
3.  **Qwen3AutoLabelEmotions:** Usa `Qwen2-Audio-Instruct` para escuchar el audio y generar etiquetas descriptivas (emoción, género, tono) automáticamente.
4.  **Qwen3ExportJSONL:** Exporta los datos procesados finales a un archivo `.jsonl` listo para entrenar.

### 🎓 Entrenamiento
*   **Qwen3DataPrep:** Pre-procesa el archivo JSONL convirtiéndolo en tensores tokenizados (`input_ids`, `labels`) para un entrenamiento eficiente.
*   **Qwen3TrainLoRA:** Entrena un adaptador LoRA con los datos pre-procesados. Soporta configuración de `rank`, `alpha`, `epochs`, etc.
*   **Qwen3FineTune:** (Legacy) Lógica de fine-tuning completo.

### 🛠️ Utilidades
*   **Qwen3SaveAudio:** Guarda lotes de audio generados en una subcarpeta específica dentro del directorio de salida.
*   **Qwen3LoadAudioFromPath:** Carga audio desde una ruta absoluta (útil para pruebas).

## Consejos de Uso
*   **Diseño de Voz:** Usa los campos individuales (Gender, Pitch, etc.) para crear una voz específica. No es necesario rellenarlos todos.
*   **Entrenamiento LoRA:** Ejecuta siempre **DataPrep** primero para generar el archivo `_codes.jsonl`. Esto acelera significativamente el entrenamiento al pre-calcular los tokens.
