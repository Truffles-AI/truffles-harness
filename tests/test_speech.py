"""Real OpenAI SDK serialization against stubbed HTTP; no network or credentials."""

import json

import httpx
import pytest
from openai import InternalServerError

from truffles import speech_model


def test_transcription_and_speech_contract(respx_mock):
    transcribe = respx_mock.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "मेरे खर्च का सारांश बताओ"})
    )
    speak = respx_mock.post("https://api.openai.com/v1/audio/speech").mock(
        return_value=httpx.Response(
            200, content=b"ID3audio", headers={"content-type": "audio/mpeg"}
        )
    )
    speech = speech_model("openai", api_key="test-key-not-a-credential", max_retries=0)
    assert (
        speech.transcribe(b"RIFFaudio", filename="question.wav", language="hi")
        == "मेरे खर्च का सारांश बताओ"
    )
    request = transcribe.calls.last.request
    assert b'name="model"' in request.content and b"whisper-1" in request.content
    assert b'filename="question.wav"' in request.content and b"RIFFaudio" in request.content
    assert b'name="language"' in request.content
    assert speech.speak("Hello") == b"ID3audio"
    assert json.loads(speak.calls.last.request.content) == {
        "model": "tts-1",
        "voice": "alloy",
        "input": "Hello",
        "response_format": "mp3",
    }
    speech.close()


def test_speech_custom_adapter_and_bad_provider():
    custom = object()
    assert speech_model(custom) is custom
    with pytest.raises(ValueError):
        speech_model(custom, voice="alloy")
    with pytest.raises(ValueError):
        speech_model("unknown")
    with pytest.raises(ValueError):
        speech_model("openai", client=custom, api_key="bad")


def test_speech_compatible_endpoint_and_sdk_errors(respx_mock):
    route = respx_mock.post("https://speech.internal/v1/audio/transcriptions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "down"}})
    )
    speech = speech_model(
        "openai", base_url="https://speech.internal/v1", api_key="test", max_retries=0
    )
    with pytest.raises(InternalServerError):
        speech.transcribe(b"RIFF", filename="a.wav")
    assert route.call_count == 1
    speech.close()


def test_speech_borrowed_client_is_not_closed():
    class Client:
        def close(self):
            raise AssertionError("Caller owns this client")

    adapter = speech_model("openai", client=Client())
    adapter.close()
