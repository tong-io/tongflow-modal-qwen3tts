# tongflow-modal-qwen3tts

Official TongFlow plugin. Text-to-speech with **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-1.7B`, Base / CustomVoice / VoiceDesign variants), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Speech synthesis — preset** (`text-gen-speech-preset`) — synthesize speech in a preset voice/style.
- **Speech synthesis — voice clone** (`text-gen-speech-clone`) — clone a voice from reference audio.
- **Speech synthesis — instruction** (`text-gen-speech-instruct`) — drive the voice/style from an instruction.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The Qwen3-TTS weights are public — no Hugging Face token required.
