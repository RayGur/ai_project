import jams

# 建立 JAMS 對象
jam = jams.JAMS()

# 填充大範圍標註
beat_annotation = jams.Annotation(namespace="beat")
for beat_time in beat_times:
    beat_annotation.append(time=beat_time, duration=0, value="beat")

# 填充細部標註
onset_annotation = jams.Annotation(namespace="onset")
for onset_time in onset_times:
    onset_annotation.append(time=onset_time, duration=0, value="onset")

# 添加標註到 JAMS
jam.annotations.append(beat_annotation)
jam.annotations.append(onset_annotation)

# 保存為 JAMS 文件
jam.save("output_annotation.jams")
