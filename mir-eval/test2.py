import mir_eval
import numpy as np

# Example reference and estimated beat times (in seconds)
ref_beats = np.array([0.5, 1.0, 1.5, 2.0])
est_beats = np.array([0.45, 1.05, 1.55, 2.05])

# Evaluate the beat tracking performance
beat_f_measure = mir_eval.beat.f_measure(ref_beats, est_beats)

print(f"Beat Tracking F-measure: {beat_f_measure:.2f}")

# Example for pitch estimation evaluation
ref_intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
ref_pitches = np.array([440.0, 880.0])  # Reference pitches in Hz
est_intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
est_pitches = np.array([445.0, 875.0])  # Estimated pitches in Hz

# Evaluate pitch estimation
pitch_score = mir_eval.transcription.precision_recall_f1_overlap(
    ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=0.2
)

print(f"Pitch Transcription Scores: {pitch_score}")
