# ComfyUI Qwen3-TTS
Una suite de nodos personalizados de ComfyUI para [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), compatible con modelos de 1.7B y 0.6B, Voz Personalizada (Custom Voice), Diseño de Voz (Voice Design), Clonación de Voz (Voice Cloning) y Fine-Tuning.

> 🎤 **¿Buscas Speech-to-Text?** Echa un vistazo a [ComfyUI-Qwen3-ASR](https://github.com/DarioFT/ComfyUI-Qwen3-ASR) para transcripción de audio con salidas compatibles.

<p align="center">
    <img src="https://raw.githubusercontent.com/DarioFT/ComfyUI-Qwen3-TTS/refs/heads/main/assets/intro.png"/>
<p>

## Características

- **Selector Dinámico de Modelos**: Detecta automáticamente modelos descargados y carpetas locales en `ComfyUI/models/tts/`.
- **Integración con ComfyUI**: Los modelos se guardan organizados junto con otros modelos de ComfyUI.
- **Descarga bajo demanda**: Solo descarga el modelo que seleccionas, no todos los variantes.
- **Soporte completo de Qwen3-TTS**:
  - **Custom Voice**: Usa 9 voces preestablecidas de alta calidad (Vivian, Ryan, etc.).
  - **Voice Design**: Crea nuevas voces usando descripciones en lenguaje natural.
  - **Voice Cloning**: Clona voces a partir de un clip de audio de referencia corto.
- **Fine-Tuning**: Entrena un modelo de voz personalizado usando tu propio conjunto de datos (carpeta de archivos .wav + .txt).
  - Reanudación de entrenamiento desde checkpoints.
  - Optimizaciones de VRAM: gradient checkpointing, AdamW de 8 bits, tamaños de lote configurables.
  - Checkpoints por época con limpieza automática.
  - Soporte para modelos de 1.7B y 0.6B.
- **Comparación de Audio**: Evalúa modelos entrenados con métricas de similitud de hablante y espectrograma mel.
- **Soporte Multilingüe**: Genera voz en chino, inglés, japonés, coreano, alemán, francés, ruso, portugués, español e italiano.
- **Atención Flexible**: soporte robusto para `flash_attention_2` con retroceso automático a `sdpa` (atención estándar de PyTorch 2.0) si faltan dependencias.
- **Soporte WSL2**: Corrección automática de rutas para usuarios de WSL2 que acceden a discos de Windows (ej. `Z:\` se mapea a `/mnt/z/`).

## Instalación

1.  Clona este repositorio en tu carpeta `ComfyUI/custom_nodes`:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/DarioFT/ComfyUI-Qwen3-TTS.git
    ```
2.  Instala las dependencias:
    ```bash
    cd ComfyUI-Qwen3-TTS
    pip install -r requirements.txt
    ```

    **For instalaciones portables/standalone de ComfyUI**, usa el Python integrado:
    ```bash
    # Desde tu carpeta raíz ComfyUI_windows_portable
    .\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Qwen3-TTS\requirements.txt
    ```

    > ⚠️ **Nota**: ComfyUI no instala automáticamente las dependencias de `requirements.txt`. Debes ejecutar el comando de instalación manualmente (o usar ComfyUI Manager).

    *Para aceleración GPU, asegúrate de tener instalado un PyTorch compatible con CUDA.*

## Almacenamiento de Modelos

Los modelos y tokenizers se guardan automáticamente en tu carpeta de modelos de ComfyUI:
```
ComfyUI/models/tts/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
├── Qwen3-TTS-12Hz-0.6B-Base/
├── Qwen3-TTS-Tokenizer-12Hz/          # Para fine-tuning
└── prompts/                           # Embeddings de voz guardados (.safetensors)
```

**Primer uso**: Cuando seleccionas un modelo y ejecutas el flujo de trabajo por primera vez, se descargará automáticamente.
**Modelos Locales**: Puedes colocar tus propias carpetas de modelos en este directorio y aparecerán automáticamente en la lista del nodo Loader con el prefijo `Local:`.

## Uso

### 1. Cargar Modelo (🎙️ Qwen3-TTS Loader)
- **repo_id**: Selecciona el modelo que deseas usar.
  - Modelos `CustomVoice`: Para usar hablantes preestablecidos.
  - Modelos `VoiceDesign`: Para diseñar voces con descripciones de texto.
  - Modelos `Base`: Para clonación de voz y fine-tuning.
  - Entradas `Local: ...`: Tus modelos personalizados detectados en la carpeta.
- **source**: Elige entre HuggingFace o ModelScope para descargar.
- **local_model_path**: (Opcional) Ruta forzada a un modelo local.
- **attention**: Deja en `auto` para el mejor rendimiento.

### 2. Generar Audio

Conecta el modelo cargado a uno de los nodos generadores:

#### **Custom Voice** (🗣️ Qwen3-TTS Custom Voice)
- **speaker**: Elige uno de los 9 preajustes (ej. Vivian, Ryan).
- **text**: El texto a hablar.
- **language**: Idioma objetivo (o Auto).
- **Parámetros de Generación**:
    - **top_p** (def: 0.8): Controla la diversidad.
    - **temperature** (def: 0.7): Controla la creatividad/aleatoriedad.
    - **repetition_penalty** (def: 1.1): Aumenta (ej. 1.2) si la generación entra en bucles infinitos o repite palabras.

#### **Voice Design** (🎨 Qwen3-TTS Voice Design)
- **instruct**: Describe la voz que quieres, ej. *"A deep, resonant male voice, narrator style, calm and professional."*
- **text**: El texto a hablar.
- **Parámetros**: Iguales que en Custom Voice.

#### **Voice Clone** (👥 Qwen3-TTS Voice Clone)
- **ref_audio**: Sube un archivo de audio de referencia (1-10 segundos ideal).
- **ref_text**: La transcripción del audio de referencia (mejora la calidad).
- **text**: El texto para que la voz clonada hable.
- **max_new_tokens**: Máximo de tokens a generar.
- **ref_audio_max_seconds**: Auto-recorta el audio de referencia.
- **Parámetros**: Iguales que en Custom Voice.

### 3. Avanzado: Caché de Prompts y Librerías de Voz

Usa el nodo **Qwen3-TTS Prompt Maker** para precalcular las características de voz de un audio de referencia. Esto es más rápido si generas muchas frases con la misma voz clonada.

#### Guardar Embeddings de Voz

Puedes guardar los prompts de clonación de voz en disco para reutilizarlos:

1. **💾 Qwen3-TTS Save Prompt**: Toma un `QWEN3_PROMPT` y lo guarda en `models/tts/prompts/`.
2. **📂 Qwen3-TTS Load Prompt**: Desplegable de prompts guardados.

## Fine-Tuning (Entrenamiento)

Entrena un modelo dedicado para una voz específica.

1.  **Preparar Dataset**: Usa **📁 Qwen3-TTS Dataset Maker** para crear un `dataset.jsonl` desde una carpeta de archivos `.wav` y `.txt`.
2.  **Procesar Datos**: Usa **⚙️ Qwen3-TTS Data Prep** para tokenizar el audio.
3.  **Fine-Tune**: Usa **🎓 Qwen3-TTS Finetune**.
    - Conecta el `*_codes.jsonl`.
    - Selecciona un modelo base (1.7B o 0.6B).
    - Ajusta epochs y batch_size según tu VRAM.
4.  **Evaluar**: Usa **📊 Qwen3-TTS Audio Compare**.
5.  **Usar Modelo Entrenado**:
    - Selecciona tu modelo en **Qwen3-TTS Loader** (aparecerá como `Local: <nombre_carpeta>`).
    - Usa el nodo **Custom Voice**. Escribe el nombre exacto de tu hablante (`speaker_name`) en el campo `custom_speaker_name` si no aparece en la lista.

## Solución de Problemas

### Bloqueos de Generación / GPU al 100%

El modelo Qwen3-TTS puede entrar ocasionalmente en bucles de generación infinita.

**Soluciones:**
1. **Aumentar `repetition_penalty`**: Prueba valores como 1.1 o 1.2. Esta es la solución más efectiva.
2. **Reducir `max_new_tokens`**: Prueba 1024 para salidas más cortas.
3. **Usar audio de referencia más corto**: Para clonación de voz, 5-15 segundos es ideal.
4. **Matar el proceso Python** y reiniciar si se queda colgado.

### Inferencia lenta en Windows

Si no tienes FlashAttention 2 (común en Windows), la inferencia puede ser más lenta.
- Configura **attention** en `sdpa` o `eager`.
- Considera usar WSL2.

### Problemas de rutas en WSL2 / "File Not Found"

Si ejecutas ComfyUI dentro de WSL2 pero accedes a archivos en un disco de Windows (ej. `Z:\mi_audio.wav`), esta suite de nodos convierte automáticamente las rutas a sus equivalentes en WSL (ej. `/mnt/z/mi_audio.wav`).

**Depuración:**
- Revisa la consola de ComfyUI en busca de mensajes que empiecen por `[Qwen3-TTS DEBUG]`. Estos registros muestran la conversión exacta de rutas, así como el conteo de archivos para la creación de datasets y el progreso del entrenamiento.

## Créditos

Basado en la librería [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) de QwenLM.
