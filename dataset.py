# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,  # wav path, URL, base64
    np.ndarray,  # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    def __init__(self, data_source, processor, config: Qwen3TTSConfig, lag_num=-1):
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self._ref_mel_cache = {}

        # --- FIX: Robust initialization logic ---
        if isinstance(data_source, str):
            # Clean path just in case (remove quotes/whitespace)
            clean_path = data_source.strip().strip('"').strip("'")

            if os.path.exists(clean_path):
                self.jsonl_path = clean_path
                self.data_list = None
                print(f"[TTSDataset] Loading lazy dataset from: {self.jsonl_path}")
                self.offsets = self._build_offsets(self.jsonl_path)
            else:
                # Raise explicit error instead of falling back to list mode
                raise FileNotFoundError(f"❌ Critical Error: Dataset file not found at path: '{data_source}' (Cleaned: '{clean_path}')")

        elif isinstance(data_source, list):
            self.data_list = data_source
            self.jsonl_path = None
            self.offsets = None
        else:
            raise ValueError(f"Invalid data_source type: {type(data_source)}. Expected str (path) or list.")

    def _build_offsets(self, path):
        offsets = []
        offset = 0
        try:
            with open(path, "rb") as f:
                for line in f:
                    offsets.append(offset)
                    offset += len(line)
        except Exception as e:
            raise RuntimeError(f"Error building offsets for {path}: {e}")
        return offsets

    def _get_item_from_jsonl(self, idx):
        offset = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            try:
                return json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON Decode Error at index {idx} in {self.jsonl_path}: {e}")

    def __len__(self):
        if self.data_list is not None:
            return len(self.data_list)
        return len(self.offsets)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:

        # Force load at 24000 to match model requirements
        target_sr = 24000
        try:
            audio, sr = librosa.load(x, sr=target_sr, mono=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file '{x}': {e}")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(
        self, audios: Union[AudioLike, List[AudioLike]]
    ) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        if sr != 24000:
            pass

        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx):
        if self.data_list is not None:
            item = self.data_list[idx]
        else:
            item = self._get_item_from_jsonl(idx)

        # Safety check for dictionary
        if not isinstance(item, dict):
             raise TypeError(f"Dataset item at index {idx} is not a dictionary. Got {type(item)}. Content: {str(item)[:100]}")

        audio_path = item["audio"]
        text = item["text"]
        audio_codes = item["audio_codes"]
        language = item.get("language", "Auto")
        ref_audio_path = item["ref_audio"]

        # Optimization: Use pre-computed text_ids if available (from Qwen3DataPrep)
        if "text_ids" in item:
            text_ids = torch.tensor(item["text_ids"], dtype=torch.long)
            if text_ids.dim() == 1:
                text_ids = text_ids.unsqueeze(0)
        else:
            text = self._build_assistant_text(text)
            text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        # Use cached ref_mel if available
        if ref_audio_path in self._ref_mel_cache:
            ref_mel = self._ref_mel_cache[ref_audio_path]
        else:
            ref_audio_list = self._ensure_list(ref_audio_path)
            normalized = self._normalize_audio_inputs(ref_audio_list)
            wav, sr = normalized[0]
            ref_mel = self.extract_mels(audio=wav, sr=sr)
            # Only cache if NOT using self-reference to save RAM?
            # Actually, with self-reference, cache grows linearly with dataset size which is bad.
            # FIX: If self-reference (many unique paths), consider not caching or clearing.
            # For now, let's keep it simple. If we run out of RAM, we disable cache.
            # self._ref_mel_cache[ref_audio_path] = ref_mel
            pass

        return {
            "text_ids": text_ids[:, :-5],  # 1 , t
            "audio_codes": audio_codes,  # t, 16
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch
        ]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = (
                self.config.tts_pad_token_id
            )
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # for speaker embedding
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = (
                self.config.talker_config.codec_pad_id
            )
            input_ids[i, 8 + text_ids_len - 3, 1] = (
                self.config.talker_config.codec_pad_id
            )
            input_ids[i, 8 + text_ids_len - 2, 1] = (
                self.config.talker_config.codec_bos_id
            )
            input_ids[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1
            ] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                self.config.talker_config.codec_eos_token_id
            )

            codec_0_labels[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
            ] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
                self.config.talker_config.codec_eos_token_id
            )

            codec_ids[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :
            ] = audio_codecs

            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # for speaker embedding

            codec_mask[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
            ] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        # --- FIX: Dynamic Padding for Variable Length ref_mels ---
        raw_ref_mels = [data["ref_mel"] for data in batch]

        # Calculate max time length in this batch (dim 1)
        max_mel_len = max([r.shape[1] for r in raw_ref_mels])

        # Initialize padded tensor [Batch, MaxTime, n_mels]
        # Using same dtype as source (float32)
        ref_mels = torch.zeros(len(batch), max_mel_len, raw_ref_mels[0].shape[2], dtype=raw_ref_mels[0].dtype)

        for i, mel in enumerate(raw_ref_mels):
            # mel shape is [1, Time, 128]
            curr_len = mel.shape[1]
            ref_mels[i, :curr_len, :] = mel[0, :, :]

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
