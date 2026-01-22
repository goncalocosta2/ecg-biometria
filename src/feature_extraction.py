import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import cheby2, filtfilt, find_peaks
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler

def read_csv(folder=None, filename=None, skiprows=None):
    if filename is None:
        print("No filename!")
        return
    filepath = filename
    if folder is not None:
        filepath = os.path.join(folder, filename)
    with open(filepath, "r") as file:
        if skiprows:
            data = pd.read_csv(file, names=["red", "ir"], skiprows=skiprows)
        else:
            data = pd.read_csv(file, names=["red", "ir"])
    return data

def plot_data(data1, data2=None, folder=None, filename=None, peaks=None):
    filepath = folder if folder is not None else ""
    plt.figure(figsize=(6,3))
    ax = sns.lineplot(data1)
    if data2 is not None:
        sns.lineplot(data2)
    ax.set(xlabel="Sample number")
    plt.tight_layout()
    if peaks is not None:
        sns.scatterplot(x=peaks, y=np.take(data1, peaks), color="red")
    if filename:
        plt.savefig(os.path.join(filepath, filename))
        plt.close()
    else:
        plt.show()

def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=4, rs=40):
    nyq = 0.5 * fs
    if lowcut > 0:
        b_high, a_high = cheby2(order, rs, lowcut / nyq, btype='high')
        data = filtfilt(b_high, a_high, data)
    b_low, a_low = cheby2(order, rs, highcut / nyq, btype='low')
    data = filtfilt(b_low, a_low, data)
    return data

def find_manual_foot_notch(signal, peaks_idx):
    foot_idx = []
    notch_idx = []
    for i, peak in enumerate(peaks_idx):
        start = peaks_idx[i-1] if i > 0 else max(0, peak-50)
        foot = start + np.argmin(signal[start:peak+1])
        foot_idx.append(foot)
        end = peaks_idx[i+1] if i < len(peaks_idx)-1 else min(len(signal)-1, peak+50)
        notch = peak + np.argmin(signal[peak:end+1])
        notch_idx.append(notch)
    return np.array(foot_idx), np.array(notch_idx)

# LIMPEZA DE OUTLIERS (MAD)
def clean_outliers_mad(df, subject_col, exclude_cols=None, k=3, p_low=25, p_high=75):
    df_clean = df.copy()
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    for subject in df_clean[subject_col].unique():
        idx = df_clean[subject_col] == subject
        for col in feature_cols:
            x = df_clean.loc[idx, col].values
            if np.all(np.isnan(x)):
                continue
            med = np.nanmedian(x)
            mad = median_abs_deviation(x, nan_policy='omit')
            if mad == 0 or np.isnan(mad):
                continue
            outliers = np.abs(x - med) > k * mad
            if np.any(outliers):
                low = np.nanpercentile(x, p_low)
                high = np.nanpercentile(x, p_high)
                x[outliers] = np.clip(x[outliers], low, high)
            df_clean.loc[idx, col] = x
    return df_clean

# REMOÇÃO DE SUJEITOS COM DISPERSÃO MULTIVARIADA (DETERMINANTE) APÓS NORMALIZAÇÃO
def remove_bad_subjects_by_dispersion(df, subject_col, exclude_cols=None, factor=3.0):
    if exclude_cols is None:
        exclude_cols = []

    # Seleciona apenas features contínuas (ignora colunas categóricas e contagem de picos)
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
        and not c.startswith("num_peaks")
    ]

    subject_variances = {}
    for subject in df[subject_col].unique():
        X = df.loc[df[subject_col] == subject, feature_cols].dropna().values
        if X.shape[0] < 2:
            subject_variances[subject] = np.nanmean(np.nanvar(X, axis=0))
        else:
            cov = np.cov(X, rowvar=False)
            subject_variances[subject] = np.linalg.det(cov)

    values = np.array([v for v in subject_variances.values() if not np.isnan(v)])
    med = np.median(values)
    mad_val = median_abs_deviation(values)
    if mad_val < 1e-12:
        print("MAD da dispersão é zero — nenhum sujeito removido")
        return df

    threshold = med + factor * mad_val
    bad_subjects = [s for s, v in subject_variances.items() if v > threshold]

    for s in bad_subjects:
        print(f"Sujeito {s} removido (det(cov)={subject_variances[s]:.4e}, threshold={threshold:.4e})")

    df_clean = df[~df[subject_col].isin(bad_subjects)].reset_index(drop=True)
    print(f"Sujeitos removidos: {len(bad_subjects)}")
    print(f"Sujeitos mantidos: {df_clean[subject_col].nunique()}")
    return df_clean

# MAIN
def main():
    plot_data_raw = False  
    plot_data_filtered = False  
    extract_features = True
    
    acquisitions_folder = "data"
    fs = 100
    order = 2
    lowcut = 0.8
    highcut = 10
    win_size_time = 25
    overlap_percentage = 95
    win_frames = int(win_size_time) * fs
    overlap_frames = int(win_frames * overlap_percentage / 100)
    
    all_features = []

    for file in os.listdir(acquisitions_folder):
        if not file.endswith(".csv"):
            continue
        print(file)
        filename = os.path.splitext(file)[0]
        data = read_csv(acquisitions_folder, file)
        data = data[1*fs:]
        red_data = data["red"]
        ir_data = data["ir"]

        if extract_features:
            counter = 1
            for i in range(0, int(len(data.index)) - win_frames + 1, win_frames - overlap_frames):
                initial_ind = i
                final_ind = i + win_frames - 1
                data_red_win = red_data[initial_ind:final_ind]
                data_ir_win = ir_data[initial_ind:final_ind]

                red_data_filt = cheby2_bandpass_filter(data_red_win, lowcut, highcut, fs, order=4)
                ir_data_filt = cheby2_bandpass_filter(data_ir_win, lowcut, highcut, fs, order=4)

                peaks_red, _ = find_peaks(red_data_filt, distance=fs*0.5)
                peaks_ir, _ = find_peaks(ir_data_filt, distance=fs*0.5)
                if len(peaks_red) < 2 or np.std(red_data_filt) < 0.01:
                    continue
                if len(peaks_ir) < 2 or np.std(ir_data_filt) < 0.01:
                    continue

                rr_intervals_red = np.diff(peaks_red)/fs
                rr_intervals_ir = np.diff(peaks_ir)/fs

                subject_id = file[:4]
                groups = file[5]

                features = {
                    "sujeito": subject_id,
                    "window": counter,
                    "groups": groups,
                    "red_mean": np.mean(red_data_filt),
                    "red_std": np.std(red_data_filt),
                    "num_peaks_red": len(peaks_red),
                    "rr_mean_red": np.mean(rr_intervals_red) if len(rr_intervals_red)>0 else np.nan,
                    "rr_std_red": np.std(rr_intervals_red) if len(rr_intervals_red)>0 else np.nan,
                    "ir_mean": np.mean(ir_data_filt),
                    "ir_std": np.std(ir_data_filt),
                    "num_peaks_ir": len(peaks_ir),
                    "rr_mean_ir": np.mean(rr_intervals_ir) if len(rr_intervals_ir)>0 else np.nan,
                    "rr_std_ir": np.std(rr_intervals_ir) if len(rr_intervals_ir)>0 else np.nan
                }

                all_features.append(features)
                counter += 1

    df_features = pd.DataFrame(all_features)

    # LIMPEZA DE OUTLIERS POR SUJEITO
    df_features = clean_outliers_mad(
        df_features,
        subject_col="sujeito",
        exclude_cols=["sujeito", "window", "groups"]
    )

    # NORMALIZAÇÃO GLOBAL DAS FEATURES CONTÍNUAS
    exclude = ["sujeito", "window", "groups"]
    feature_cols = [c for c in df_features.columns if c not in exclude and not c.startswith("num_peaks")]
    scaler = StandardScaler()
    df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])

    # REMOÇÃO DE SUJEITOS APÓS NORMALIZAÇÃO
    df_features = remove_bad_subjects_by_dispersion(
        df_features,
        subject_col="sujeito",
        exclude_cols=["sujeito", "window", "groups"],
        factor=1.0  #ajustar para mais ou menos restritivo
    )

    df_features.to_csv("features_dataset.csv", index=False)

if __name__ == "__main__":
    main()
