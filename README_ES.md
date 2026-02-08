# ComfyUI-Qwen3-TTS (Español)

Nodos personalizados para [Qwen2.5-Audio / Qwen3-TTS](https://huggingface.co/Qwen/Qwen2.5-Audio-Instruct), un potente modelo de audio multimodal capaz de Text-to-Speech (TTS), Clonación de Voz y Diseño de Voz.

## Características

*   **🎙️ Texto a Voz (TTS):** Genera voz de alta calidad a partir de texto en múltiples idiomas (Inglés, Chino, Español, etc.).
*   **👥 Clonación de Voz:** Clona voces a partir de un clip de audio de referencia corto (se recomiendan 3-10s).
*   **🎨 Diseño de Voz:** Diseña voces personalizadas describiendo atributos como género, edad, tono, velocidad y emoción.
*   **🎓 Fine-Tuning:** Flujo completo para realizar fine-tuning del modelo con tu propio dataset de voz. El fine-tuning ofrece una estabilidad y fidelidad de tono muy superiores a la clonación "zero-shot".
*   **📁 Pipeline Modular de Dataset:** Automatiza la creación de datasets: Cargar audio crudo -> Transcribir con Whisper -> Etiquetar emociones con Qwen2-Audio -> Exportar JSONL. O usa el **Creador de Dataset** todo en uno.
*   **⚙️ Configuración Avanzada:** Solución para errores de "Unsupported speakers" en modelos fine-tuned y control detallado de prompts.
*   **📊 Análisis de Audio:** Herramientas para comparar el audio generado con el audio de referencia (Similitud de Hablante y Distancia Mel).

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
    *Nota: Las funciones de entrenamiento requieren `bitsandbytes` y `accelerate`. La creación de datasets requiere `openai-whisper` y `pydub` (además de ffmpeg instalado en tu sistema).*

---

## 📚 Descripción Detallada de los Nodos

### 🎙️ Nodos de Inferencia

#### **Qwen3Loader**
*   **Función:** Carga el modelo base (ej. `Qwen/Qwen3-TTS-12Hz-1.7B-Base`) o variantes especializadas como `CustomVoice` o `VoiceDesign`.
*   **Entradas:** `repo_id`, `precision` (se recomienda bf16), `attention` (sdpa/flash_attn).
*   **Salidas:** Objeto `QWEN3_MODEL`.
*   **Detalles:** Gestiona la descarga y caché desde HuggingFace/ModelScope. Si se proporciona una ruta de checkpoint, intenta cargarla como base (útil para debug).

#### **Qwen3LoadFineTuned**
*   **Función:** Carga un checkpoint de un modelo fine-tuned para inferencia.
*   **Entradas:** `base_model` (necesario para arquitectura/tokenizer), `speaker_name`, `version`.
*   **Salidas:** Objeto `QWEN3_MODEL` listo para generar.
*   **Detalles:** Nodo crucial para usar tus voces entrenadas. Realiza una "Inyección Profunda" de la configuración personalizada del hablante (`spk_id`) en la estructura del modelo base, evitando errores de "Unsupported speaker".

#### **Qwen3CustomVoice**
*   **Función:** Genera voz usando un ID de hablante entrenado específico.
*   **Entradas:** `model`, `text`, `language`, `speaker` (lista desplegable de hablantes detectados).
*   **Salidas:** Forma de onda de audio.
*   **Detalles:** Usado para modelos fine-tuned. Permite seleccionar el `speaker_name` específico que entrenaste.

#### **Qwen3VoiceDesign**
*   **Función:** Genera voz basada en texto y un conjunto de atributos descriptivos.
*   **Entradas:** `gender`, `pitch`, `speed`, `emotion`, `tone`, `age`, etc.
*   **Salidas:** Forma de onda de audio.
*   **Detalles:** Utiliza la variante `VoiceDesign` del modelo. No es necesario rellenar todos los campos; los vacíos se ignoran. Genial para crear personajes únicos sin audio de referencia.

#### **Qwen3VoiceClone**
*   **Función:** Clonación de voz "Zero-shot" desde un audio de referencia.
*   **Entradas:** `ref_audio` (clip de 3-10s), `ref_text` (transcripción del audio), `text` (lo que quieres que diga).
*   **Salidas:** Forma de onda de audio.
*   **Detalles:** Utiliza las variantes `Base` o `CustomVoice`. Requiere el texto de referencia para una alineación precisa del prompt.

### 📁 Dataset y Utilidades

#### **Qwen3AudioToDataset (Creador de Dataset)**
*   **Función:** Nodo todo-en-uno para crear un dataset desde una carpeta de archivos de audio.
*   **Entradas:** `audio_folder`, `model_size` (Whisper), `output_folder_name`, `min/max_duration`, `silence_threshold`.
*   **Salidas:** Ruta al archivo `dataset.jsonl` generado.
*   **Detalles:** Carga, transcribe, corta y formatea automáticamente el dataset para el entrenamiento.

#### **Qwen3TranscribeSingle**
*   **Función:** Transcribe un solo clip de audio usando Whisper.
*   **Entradas:** `audio` (entrada de AUDIO), `model_size`.
*   **Salidas:** `text` (STRING).
*   **Detalles:** Útil para preparar `ref_text` para la Clonación de Voz.

#### **Qwen3AudioCompare**
*   **Función:** Compara dos clips de audio (Referencia vs Generado) para evaluar la calidad.
*   **Entradas:** `reference_audio`, `generated_audio`, `speaker_encoder_model` (Ruta al modelo Base).
*   **Salidas:** Reporte de texto con Similitud de Hablante (Coseno) y Distancia de Espectrograma Mel (MSE).

#### **Utilidades**
*   **Qwen3LoadAudioFromPath / Folder:** Carga audio desde rutas absolutas.
*   **Qwen3LoadVideoFromPath / Folder:** Extrae audio directamente de archivos de video (requiere ffmpeg/pydub).
*   **Qwen3SavePrompt / LoadPrompt:** Guarda prompts de voz generados en .safetensors para reutilizar una voz clonada sin recalcular.

### 📁 Pipeline de Dataset (Paso a Paso)

1.  **Qwen3LoadDatasetAudio:**
    *   Escanea una carpeta local buscando archivos `.wav`. Devuelve una lista.
2.  **Qwen3TranscribeWhisper:**
    *   Usa OpenAI Whisper para transcribir el audio.
    *   Corta automáticamente audios largos (ej. < 15s) y recorta silencios.
    *   Salida: `DATASET_ITEMS` (ruta de audio + texto).
3.  **Qwen3AutoLabelEmotions:**
    *   Usa `Qwen2-Audio-Instruct` para "escuchar" cada clip.
    *   Genera etiquetas como "Male voice, angry, shouting, fast speed".
    *   Mejora la calidad del dataset permitiendo al modelo aprender condicionamiento emocional.
4.  **Qwen3ExportJSONL:**
    *   Guarda los items procesados en un archivo `dataset.jsonl`.
    *   Formato: `{"audio": "ruta/al/wav", "text": "transcripción", "instruction": "etiquetas"}`.

### 🎓 Nodos de Entrenamiento

#### **Qwen3DataPrep**
*   **Función:** Pre-tokeniza el audio y el texto.
*   **Entradas:** `jsonl_path` (del Paso 4).
*   **Salidas:** Ruta al archivo `_codes.jsonl`.
*   **Detalles:** Convierte el audio en códigos discretos usando el `speech_tokenizer` y el texto en tokens. Este paso es pesado pero asegura que el bucle de entrenamiento sea rápido y no se quede sin memoria (OOM) durante la tokenización. Gestiona errores de memoria cambiando a procesamiento secuencial si falla el lote.

#### **Qwen3FineTune**
*   **Función:** Realiza el fine-tuning completo del modelo.
*   **Entradas:** `train_jsonl` (el archivo `_codes.jsonl`), `init_model`, `epochs`, `batch_size`, `lr`, `target_loss`, `save_loss_threshold`.
*   **Salidas:** Ruta al checkpoint guardado.
*   **Detalles:**
    *   **Epochs:** Se recomienda un mínimo de 50 para convergencia en datasets pequeños.
    *   **Target Loss:** Mecanismo de parada automática. Si el loss baja de este valor (ej. 2.0), el entrenamiento se detiene y guarda el modelo.
    *   **Save Loss Threshold:** Guarda un checkpoint intermedio cuando el loss baja de este valor, sin detener el entrenamiento.
    *   **Learning Rate:** Por defecto `2e-6`. Valores más altos (ej. `1e-5`) pueden causar ruido/inestabilidad.
    *   **Mixed Precision:** Soporta `bf16` (GPUs Ampere) y `fp32`.
    *   **Guardado:** Guarda `pytorch_model.bin` y `config.json` correctamente mapeados para carga inmediata con `Qwen3LoadFineTuned`.

---

## 🧪 Ejemplos de Flujo de Trabajo

### 1. Creación de Dataset
1.  **Cargar Audio:** Conecta `Qwen3LoadDatasetAudio` apuntando a tu carpeta de wavs crudos.
2.  **Transcribir:** Conecta a `Qwen3TranscribeWhisper`. Ajusta `max_duration` a 15.0s.
3.  **Etiquetar:** Conecta a `Qwen3AutoLabelEmotions`. Esto añade etiquetas de estilo.
4.  **Exportar:** Conecta a `Qwen3ExportJSONL`.
5.  **Ejecutar:** Esto genera el archivo `dataset.jsonl`.

### 2. Entrenamiento (Fine-Tuning)
1.  **Preparar Datos:** Conecta el `dataset.jsonl` (de arriba) a `Qwen3DataPrep`.
    *   *Tip: Ejecuta esto una vez. Crea `dataset_codes.jsonl`.*
2.  **Entrenar:** Conecta la salida de `Qwen3DataPrep` a `Qwen3FineTune`.
    *   **Base Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
    *   **Speaker Name:** ej. "Batman".
    *   **Epochs:** 100.
    *   **Batch Size:** 2 o 4 (dependiendo de tu VRAM).
    *   **LR:** 2e-6.
3.  **Ejecutar:** Monitorea la consola. Guardará checkpoints en `models/tts/finetuned_model/Batman/epoch_100`.

### 3. Inferencia con Voz Fine-Tuned
1.  **Cargar:** Usa `Qwen3LoadFineTuned`.
    *   **Speaker Name:** Selecciona "Batman".
    *   **Version:** Selecciona "epoch_100".
2.  **Generar:** Conecta a `Qwen3CustomVoice`.
    *   **Text:** "Soy la venganza."
    *   **Speaker:** "Batman" (debería aparecer en la lista).
3.  **Guardar:** Conecta a `Qwen3SaveAudio`.

### 4. Inferencia con Diseño de Voz (Zero-Shot)
1.  **Cargar:** Usa `Qwen3Loader` con `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
2.  **Generar:** Conecta a `Qwen3VoiceDesign`.
    *   **Gender:** "Male"
    *   **Tone:** "Deep, raspy, intimidating"
    *   **Text:** "Esta ciudad es mía."
3.  **Guardar:** Conecta a `Qwen3SaveAudio`.
