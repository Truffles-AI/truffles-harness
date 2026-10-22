# Speech

Truffles now has a minimal speech adapter for OpenAI or an explicitly configured compatible
audio server. Provider SDKs own authentication, transport, retries and wire formats.

```sh
pip install 'truffles[audio]'
```

```python
from truffles import speech_model

speech = speech_model("openai", timeout=20, max_retries=0)
try:
    transcript = speech.transcribe(b"...audio bytes...", filename="question.wav", language="en")
    mp3_bytes = speech.speak("Your spending summary is ready.")
finally:
    speech.close()
```

Defaults: `whisper-1`, `tts-1`, voice `alloy`. Configure `transcription_model`, `synthesis_model`,
`voice`, `base_url`, `api_key`, or a native `client`. A borrowed client is not closed by the
adapter; a client created by the adapter is. The API is synchronous; async web apps should run
it in a bounded thread pool. A custom object implementing `transcribe` and `speak` can be passed
through `speech_model(custom_adapter)`.

The harness does not retain audio. Callers own consent, size/type/duration limits, timeout,
provider retention decisions and a clear disclosure that playback is AI generated. The PFM
reference demonstrates those application boundaries.

- [Official transcription contract](https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create)
- [Official speech contract](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create)
- [PFM voice implementation](../apps/pfm/docs/architecture.md)
