import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 讀取音樂文件

audio_file = "C:/AI project/trumpt sample.wav"
y, sr = librosa.load(audio_file, sr=None)

# 計算每分鐘節拍數（BPM）
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated tempo: {tempo.item():.2f} BPM")

# 將節拍轉換為時間點
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 提取每個樂段的特徵
segment_features = []
for i in range(len(beat_times) - 1):
    start = int(beat_frames[i])
    end = int(beat_frames[i + 1])
    segment = y[start:end]

    # 提取音量（RMS能量）
    rms = librosa.feature.rms(y=segment).mean()

    # 提取頻譜質心
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()

    # 提取音高特徵
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0])  # 過濾掉0以外的音高值

    segment_features.append(
        {
            "start_time": beat_times[i],
            "end_time": beat_times[i + 1],
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "avg_pitch": avg_pitch,
        }
    )

# 輸出每個樂段的特徵
for idx, features in enumerate(segment_features):
    print(f"Segment {idx + 1}:")
    print(f"  Start Time: {features['start_time']:.2f} s")
    print(f"  End Time: {features['end_time']:.2f} s")
    print(f"  RMS Energy: {features['rms']:.2f}")
    print(f"  Spectral Centroid: {features['spectral_centroid']:.2f} Hz")
    print(f"  Average Pitch: {features['avg_pitch']:.2f} Hz")
