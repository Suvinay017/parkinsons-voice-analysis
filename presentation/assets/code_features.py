def extract_features(path):
    y, sr = sf.read(path, dtype='float32')
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.size == 0:
        raise ValueError('Empty audio')
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-9)

    out = {'duration_sec': float(len(y) / sr)}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc, mode='nearest')
    mean_std_features(mfcc, 'mfcc', out)
    mean_std_features(delta, 'delta_mfcc', out)

    for name, arr in {
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr),
        'zcr': librosa.feature.zero_crossing_rate(y),
        'rms': librosa.feature.rms(y=y),
        'chroma': librosa.feature.chroma_stft(y=y, sr=sr)
    }.items():
        mean_std_features(arr, name, out)
    return out
