"""Modal deploy entry for qwen3tts.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import modal
from tongflow import deploy




_cfg: dict[str, Any] = {}


def _three_repo_ids() -> tuple[str, str, str]:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and len(repos) >= 3:
        out: list[str] = []
        for r in repos[:3]:
            if isinstance(r, dict) and r.get("repoId"):
                out.append(str(r["repoId"]))
        if len(out) == 3:
            return out[0], out[1], out[2]
    return (
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )


DESIGN_REPO_ID, CUSTOM_REPO_ID, BASE_REPO_ID = _three_repo_ids()
DESIGN_MODEL_DIR = f"/models/{DESIGN_REPO_ID}"
CUSTOM_MODEL_DIR = f"/models/{CUSTOM_REPO_ID}"
BASE_MODEL_DIR = f"/models/{BASE_REPO_ID}"

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.text_gen_speech_clone import (
    TextGenSpeechCloneInput,
    TextGenSpeechCloneOutput,
)
from tongflow.models.text_gen_speech_instruct import (
    TextGenSpeechInstructInput,
    TextGenSpeechInstructOutput,
)
from tongflow.models.text_gen_speech_preset import (
    TextGenSpeechPresetInput,
    TextGenSpeechPresetOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import node_slot


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("sox", "libsox-dev", "ffmpeg")
    .pip_install(
        "tongflow==0.1.0",
        "qwen-tts==0.1.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "soundfile==0.13.1",
        "librosa==0.10.2.post1",
        "torchaudio",
        "onnxruntime==1.22.0",
        "einops==0.8.1",
        "huggingface_hub>=0.34.0,<1.0",
        "flash-attn>=2.5.0",
    )
)

with image.imports():
    import io
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L4",
    volumes={"/models": volume},
)
class Design:
    @modal.enter()
    def load(self):
        self.tts = Qwen3TTSModel.from_pretrained(
            DESIGN_MODEL_DIR,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def _synthesize_design(
        self,
        text: str,
        language: str = "Chinese",
        instruct: str = "",
        max_new_tokens: int = 2048,
    ) -> bytes:
        wavs, sr = self.tts.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        text: str,
        language: str = "Chinese",
        instruct: str = "",
        max_new_tokens: int = 2048,
    ) -> bytes:
        return self._synthesize_design(
            text=text,
            language=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_SPEECH_INSTRUCT)
    def text_gen_speech_instruct(
        self,
        input: TextGenSpeechInstructInput,
    ) -> TextGenSpeechInstructOutput:
        text = input.text or ""
        if not text:
            return TextGenSpeechInstructOutput(success=False, error="Missing text")
        raw = self._synthesize_design(
            text=text,
            language=input.language or "Chinese",
            instruct=input.instruct or "",
            max_new_tokens=int(input.max_new_tokens)
            if input.max_new_tokens is not None
            else 2048,
        )
        return TextGenSpeechInstructOutput(
            success=True, audio=asset(raw, mime="audio/wav")
        )


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L4",
    volumes={"/models": volume},
)
class Custom:
    @modal.enter()
    def load(self):
        self.tts = Qwen3TTSModel.from_pretrained(
            CUSTOM_MODEL_DIR,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def _synthesize_custom(
        self,
        text: str,
        language: str = "Chinese",
        speaker: str = "Vivian",
        instruct: str = "",
        max_new_tokens: int = 2048,
    ) -> bytes:
        wavs, sr = self.tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        text: str,
        language: str = "Chinese",
        speaker: str = "Vivian",
        instruct: str = "",
        max_new_tokens: int = 2048,
    ) -> bytes:
        return self._synthesize_custom(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_SPEECH_PRESET)
    def text_gen_speech_preset(
        self,
        input: TextGenSpeechPresetInput,
    ) -> TextGenSpeechPresetOutput:
        text = input.text or ""
        if not text:
            return TextGenSpeechPresetOutput(success=False, error="Missing text")
        raw = self._synthesize_custom(
            text=text,
            language=input.language or "Chinese",
            speaker=input.speaker or "Vivian",
            instruct=input.instruct or "",
            max_new_tokens=int(input.max_new_tokens)
            if input.max_new_tokens is not None
            else 2048,
        )
        return TextGenSpeechPresetOutput(
            success=True, audio=asset(raw, mime="audio/wav")
        )


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L4",
    volumes={"/models": volume},
)
class Reference:
    @modal.enter()
    def load(self):
        self.tts = Qwen3TTSModel.from_pretrained(
            BASE_MODEL_DIR,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def _synthesize_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str = "",
        language: str = "Auto",
        x_vector_only: bool = False,
        max_new_tokens: int = 2048,
    ) -> bytes:
        wavs, sr = self.tts.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=x_vector_only,
            max_new_tokens=max_new_tokens,
        )
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        text: str,
        ref_audio: str,
        ref_text: str = "",
        language: str = "Auto",
        x_vector_only: bool = False,
        max_new_tokens: int = 2048,
    ) -> bytes:
        return self._synthesize_clone(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            x_vector_only=x_vector_only,
            max_new_tokens=max_new_tokens,
        )

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_SPEECH_CLONE)
    def text_gen_speech_clone(
        self,
        input: TextGenSpeechCloneInput,
    ) -> TextGenSpeechCloneOutput:
        text = (input.text or "").strip()
        if not text or input.ref_audio is None:
            return TextGenSpeechCloneOutput(
                success=False, error="Missing text or ref_audio"
            )

        ref_text = (input.ref_text or "").strip()
        # Qwen ICL mode (x_vector_only=False) requires ref_text; without a
        # transcript default to x-vector-only unless the caller opted in.
        if input.x_vector_only is None:
            x_vector_only = not ref_text
        else:
            x_vector_only = bool(input.x_vector_only)
        if not x_vector_only and not ref_text:
            return TextGenSpeechCloneOutput(
                success=False,
                error=(
                    "ref_text is required for ICL voice clone. "
                    "Add the reference transcript or use x_vector_only mode."
                ),
            )

        with asset_as_path(input.ref_audio) as ref_path:
            raw = self._synthesize_clone(
                text=text,
                ref_audio=str(ref_path),
                ref_text=ref_text,
                language=input.language or "Auto",
                x_vector_only=x_vector_only,
                max_new_tokens=int(input.max_new_tokens)
                if input.max_new_tokens is not None
                else 2048,
            )
        return TextGenSpeechCloneOutput(
            success=True, audio=asset(raw, mime="audio/wav")
        )
