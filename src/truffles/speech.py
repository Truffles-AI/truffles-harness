"""Thin speech adapter. Native SDK owns authentication, transport and retries."""

from typing import Any


class OpenAISpeech:
    """OpenAI or an explicitly configured compatible audio server.

    Audio bytes stay in memory. Callers own consent, size limits and retention.
    Pass a native client to reuse its pool; otherwise this adapter owns its client.
    """

    def __init__(
        self,
        *,
        transcription_model="whisper-1",
        synthesis_model="tts-1",
        voice="alloy",
        client=None,
        **options: Any,
    ):
        if client is not None and options:
            raise ValueError("Configure an existing client before passing it in.")
        self._owns_client = client is None
        if client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                if exc.name == "openai":
                    raise ImportError("Install speech with: pip install 'truffles[audio]'") from exc
                raise
            client = OpenAI(**options)
        self.client = client
        self.transcription_model = transcription_model
        self.synthesis_model = synthesis_model
        self.voice = voice

    def transcribe(self, audio: bytes, *, filename="recording.webm", language=None) -> str:
        options = {"language": language} if language else {}
        result = self.client.audio.transcriptions.create(
            model=self.transcription_model,
            file=(filename, audio),
            response_format="json",
            **options,
        )
        return result.text

    def speak(self, text: str) -> bytes:
        return self.client.audio.speech.create(
            model=self.synthesis_model, voice=self.voice, input=text, response_format="mp3"
        ).content

    def close(self):
        if self._owns_client:
            self.client.close()


def speech_model(provider: Any = "openai", **options: Any):
    """Create a speech adapter or pass through a custom transcribe/speak implementation."""
    if not isinstance(provider, str):
        if options:
            raise ValueError("Configure an existing speech adapter before passing it in.")
        return provider
    if provider != "openai":
        raise ValueError("Use 'openai' or pass an adapter implementing transcribe/speak.")
    return OpenAISpeech(**options)
