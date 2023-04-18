import pandas as pd


def freq_to_seconds(freq, unit):
    return pd.to_timedelta(freq, unit=unit).total_seconds()
