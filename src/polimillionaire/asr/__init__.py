"""Speech transcription for the speech-mode game API.

`WhisperTranscriber` loads `openai/whisper-large-v3-turbo` on demand and
turns WAV bytes (24 kHz mono PCM, as served by the assignment API) into
text. See `polimillionaire.play.speech_play_loop` for the orchestration.
"""

from polimillionaire.asr.whisper import WhisperTranscriber

__all__ = ["WhisperTranscriber"]
