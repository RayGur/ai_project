import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


# 載入音訊
audio_path = "C:/AI project/trumpt sample.wav"
y, sr = librosa.load(audio_path, sr=None)

# 檢測音訊節拍（用來劃分段落）
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

print(f"Detected Tempo: {tempo:.2f} BPM")
print(f"Beat times: {beat_times}")

# 畫出波形和節拍
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.vlines(beat_times, -1, 1, color="r", linestyle="--", label="Beats")
plt.title("Audio Waveform with Beat Times")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# 偵測音符的開始與結束
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# 繪製波形與音符起始點
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.vlines(onset_times, -1, 1, color="g", linestyle="--", label="Onsets")
plt.title("Audio Waveform with Onset Times")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# 分析每個音符的音高與其他特徵
for i, onset_time in enumerate(onset_times):
    print(f"Onset {i+1}: Time = {onset_time:.2f}s")
    # 提取音高和其他特徵（此處可使用進一步分析，如音高提取）
