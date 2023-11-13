import pandas as pd
import hashlib


def slowly_changing_dimensions(
    primary_key: str,
    new_data: pd.DataFrame,
    existing_data: pd.DataFrame = None,
    track_columns: list = None,
    initial: bool = False,
    track_changes: bool = False,
    return_only_updated: bool = False,
) -> pd.DataFrame:
    if initial and existing_data is not None:
        raise Exception(
            "Unexpected input: 'existing_data' DataFrame provided but initial run requested. Exlusively provide 'new_data' DataFrame when requesting initial run."
        )

    if "scd_id" in new_data.columns:
        raise Exception(
            "Unexpected input: 'new_data' DataFrame contains column 'scd_id', indicating DataFrame was already processed."
        )

    if return_only_updated and existing_data is None:
        raise Exception(
            "Impossible to return only updated rows: 'existing_data' DataFrame was not provided."
        )

    if track_changes is False and "scd_change" in existing_data.columns.to_list():
        print(
            "Notification: Existing DataFrame contains 'scd_change' column, but track changes is disabled in update request."
        )

    execution_time = pd.Timestamp.now()

    if track_columns is not None:
        columns = [
            col
            for col in new_data.columns
            if col in track_columns or col in primary_key
        ]
    else:
        columns = new_data.columns.to_list()

    # print(columns)

    def _hash_row(row):
        row_str = str(row.values)
        hasher = hashlib.md5()
        hasher.update(row_str.encode())
        return hasher.hexdigest()

    def _scd_initial(df: pd.DataFrame) -> pd.DataFrame:
        df["scd_id"] = df[columns].apply(_hash_row, axis=1)
        df["scd_start"] = execution_time
        df["scd_end"] = None
        df["scd_end"] = pd.to_datetime(df["scd_end"])
        df["scd_active"] = True
        df["scd_record"] = 1
        if track_changes:
            df["scd_change"] = "first load"
        return df

    def _selective_merge(df: pd.DataFrame) -> pd.DataFrame:
        all_columns = new_data.columns.to_list()
        all_columns = [col for col in all_columns if col not in track_columns]
        all_columns.remove(primary_key)
        for col in all_columns:
            df[col] = df[f"{col}_scd_new"].fillna(df[f"{col}_scd_old"])
        cols_to_drop = [
            col for col in df.columns if "_scd_new" in col or "_scd_old" in col
        ]
        df = df.drop(cols_to_drop, axis=1)
        return df

    def _scd_update(df: pd.DataFrame) -> pd.DataFrame:
        df["scd_id"] = df[columns].apply(_hash_row, axis=1)
        df["scd_start"] = df["scd_start"].fillna(execution_time)
        df["scd_active"] = df["scd_active"].fillna(True)
        left_only_mask = df["_merge"] == "left_only"
        df.loc[left_only_mask, "scd_end"] = execution_time
        df.loc[left_only_mask, "scd_active"] = False
        df.sort_values(by=[primary_key, "scd_start"])
        df["scd_record"] = df.groupby(primary_key).cumcount() + 1

        if track_changes:
            df = df.sort_values(by=["id", "scd_start"])

            def _find_changed_cols(group):
                if len(group) > 1:
                    changes = group.iloc[-1] != group.iloc[-2]
                    changed_cols = changes[changes].index.tolist()
                    changed_cols = [
                        col
                        for col in changed_cols
                        if "scd_" not in col and "_merge" not in col
                    ]
                    group.iloc[-1, group.columns.get_loc("scd_change")] = ", ".join(
                        changed_cols
                    )
                return group

            df = df.groupby("id").apply(_find_changed_cols).reset_index(drop=True)

        if return_only_updated:
            df = df[(df["_merge"] != "both")]
        else:
            df = df

        df = df.drop("_merge", axis=1)

        return df

    if initial:
        return _scd_initial(df=new_data)
    else:
        if track_columns:
            merge_input_dataframes = existing_data.merge(
                new_data,
                on=columns,
                how="outer",
                indicator=True,
                suffixes=("_scd_old", "_scd_new"),
            )
            merge_input_dataframes = _selective_merge(merge_input_dataframes)
            updated_df = _scd_update(merge_input_dataframes)
            updated_df = updated_df[existing_data.columns.to_list()]
            return updated_df
        else:
            merge_input_dataframes = existing_data.merge(
                new_data, on=columns, how="outer", indicator=True
            )
            return _scd_update(merge_input_dataframes)
