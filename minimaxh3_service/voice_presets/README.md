# Bundled voice references

These synthetic reference clips preserve the application's existing preset IDs
when migrating to IndexTTS 2.5. They were generated locally with
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache-2.0), not recordings of users.
See manifest.json for model revision, speaker and reference text.

IndexTTS clones these fixed reference WAVs at inference time. No Qwen3-TTS
runtime or model is required to use the presets. Voice characteristics can
differ from the previous engine. The reference speaker's language does not
extend IndexTTS's supported output languages (ZH, EN, JA, ES, AR).

Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
IndexTTS: https://github.com/index-tts/index-tts
