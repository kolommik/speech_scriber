import pandas as pd


def merge_results(whisper_df, diarization_df):
    # Объединение результатов
    def find_speaker(start, end):
        overlaps = diarization_df[
            (diarization_df["start"] <= end) & (diarization_df["end"] >= start)
        ]
        if not overlaps.empty:
            return overlaps.iloc[0]["speaker"]
        return "Unknown"

    whisper_df["speaker"] = whisper_df.apply(
        lambda row: find_speaker(row["start"], row["end"]), axis=1
    )

    # Объединение строк для одного и того же спикера
    merged_df = merge_speaker_segments(whisper_df)
    return merged_df


def merge_speaker_segments(df):
    merged_segments = []
    current_segment = None

    for _, row in df.iterrows():
        if current_segment is None:
            current_segment = row
        else:
            if row["speaker"] == current_segment["speaker"]:
                current_segment["end"] = row["end"]
                current_segment["text"] += " " + row["text"]
            else:
                merged_segments.append(current_segment)
                current_segment = row

    if current_segment is not None:
        merged_segments.append(current_segment)

    return pd.DataFrame(merged_segments)
