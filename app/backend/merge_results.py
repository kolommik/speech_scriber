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
    return whisper_df
