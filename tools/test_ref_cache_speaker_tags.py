"""Regression test: the reference cache must not carry speaker tags across requests.

Voice A is speaker 3 in one request and speaker 2 in the next. The second
request must see its own <|speaker:2|> tag, not the cached <|speaker:3|>.
Runs without a GPU (encode_reference is stubbed).
"""
import os
import sys

os.environ["FISH_REF_MAX_S"] = "0"  # no trimming: the stub audio is not decodable
sys.path.insert(0, ".")
from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.utils.schema import ServeReferenceAudio


class Loader(ReferenceLoader):
    def encode_reference(self, reference_audio, enable_reference_audio):
        return f"codes-of-{reference_audio.decode()}"


loader = Loader()
A, B, C = b"voice-a", b"voice-b", b"voice-c"
first = [ServeReferenceAudio(audio=B, text="<|speaker:2|>b text"),
         ServeReferenceAudio(audio=A, text="<|speaker:3|>a text")]
second = [ServeReferenceAudio(audio=A, text="<|speaker:2|>a text"),
          ServeReferenceAudio(audio=C, text="<|speaker:3|>c text")]
loader.load_by_hash(first, "on")
tokens, texts = loader.load_by_hash(second, "on")
print("second request texts:", texts)
assert tokens == ["codes-of-voice-a", "codes-of-voice-c"], tokens
assert texts == ["<|speaker:2|>a text", "<|speaker:3|>c text"], "stale speaker tag from cache"
print("PASS")
