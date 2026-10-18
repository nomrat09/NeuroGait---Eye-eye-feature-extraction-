import os
import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import fft

# Configuration
DATASET_DIR = 'ds004584-download'
EEG_DIR_NAME = 'eeg'
EEG_FILE_SUFFIX = '_task-Rest_eeg.set'
FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF8', 'F7', 'F8']
SAMPLE_SUBJECT = 'sub-101'  # Change as needed
EPOCH_DURATION = 30.0  # seconds
OVERLAP = 0.5  # 50% overlap between epochs
BLINK_REMOVAL_MS = 100  # ms to remove around blink peak

# Wavelet denoising function
def wavelet_denoise(data, wavelet='db4', level=2):
    coeffs = pywt.wavedec(data, wavelet, mode='per')
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    return pywt.waverec(denoised_coeffs, wavelet, mode='per')[:len(data)]

# Feature extraction functions
def calculate_zero_crossings(signal_data):
    """Calculate number of zero crossings in the signal"""
    return np.sum(np.diff(np.signbit(signal_data).astype(int)))

def calculate_form_factor(signal_data):
    """Calculate form factor (RMS / mean absolute value)"""
    rms = np.sqrt(np.mean(signal_data**2))
    mean_abs = np.mean(np.abs(signal_data))
    return rms / mean_abs if mean_abs != 0 else 0

def calculate_power_bands(signal_data, sfreq, bands=None):
    """Calculate power in different frequency bands"""
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    # Compute power spectral density
    freqs, psd = signal.welch(signal_data, sfreq, nperseg=min(256, len(signal_data)//4))
    
    power_bands = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Find indices for the frequency band
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(band_mask):
            power_bands[f'power_{band_name}'] = np.trapz(psd[band_mask], freqs[band_mask])
        else:
            power_bands[f'power_{band_name}'] = 0
    
    return power_bands

# --- Synthetic Blink Template ---
def generate_blink_template(sfreq, duration=0.4):
    t = np.linspace(0, duration, int(sfreq * duration), endpoint=False)
    # 3-sine sum: fast up, slow down, overshoot
    template = (
        np.sin(2 * np.pi * 1.5 * t) +
        0.5 * np.sin(2 * np.pi * 3 * t + np.pi/4) +
        0.2 * np.sin(2 * np.pi * 6 * t + np.pi/2)
    )
    # Only keep the main lobe (positive part)
    template[template < 0] = 0
    # Normalize
    template = template / np.max(np.abs(template))
    return template

# --- Blink Detection via Matched Filter ---
def detect_blinks_matched_filter(signal_data, sfreq, template=None, threshold_factor=3):
    if template is None:
        template = generate_blink_template(sfreq)
    # 2nd derivative
    d2 = np.diff(np.diff(signal_data))
    # Matched filter (correlate)
    matched_output = signal.correlate(d2, template, mode='same')
    # Threshold
    thresh = np.mean(matched_output) + threshold_factor * np.std(matched_output)
    peaks, props = signal.find_peaks(matched_output, height=thresh, distance=int(0.3*sfreq))
    # For each peak, find start/end (where output crosses 0 before/after peak)
    blink_events = []
    for peak in peaks:
        # Find start
        start = peak
        while start > 0 and matched_output[start] > 0:
            start -= 1
        # Find end
        end = peak
        while end < len(matched_output)-1 and matched_output[end] > 0:
            end += 1
        blink_events.append({'start': start, 'peak': peak, 'end': end, 'amplitude': matched_output[peak]})
    return blink_events, matched_output

# --- Blink Feature Extraction ---
def extract_blink_features(blink_events, sfreq, signal_len):
    if not blink_events:
        return {
            'blink_count': 0,
            'blink_rate_per_min': 0,
            'ibi_mean': np.nan,
            'ibi_std': np.nan,
            'ibi_entropy': np.nan,
            'blink_duration_mean': np.nan,
            'blink_duration_std': np.nan,
            'blink_amplitude_mean': np.nan,
            'blink_amplitude_std': np.nan,
            'blink_regularity': np.nan
        }
    blink_times = np.array([b['peak'] for b in blink_events]) / sfreq
    blink_amps = np.array([b['amplitude'] for b in blink_events])
    blink_durations = np.array([(b['end'] - b['start']) / sfreq for b in blink_events])
    ibis = np.diff(blink_times) if len(blink_times) > 1 else np.array([np.nan])
    blink_count = len(blink_events)
    duration_min = signal_len / sfreq / 60
    blink_rate = blink_count / duration_min if duration_min > 0 else 0
    return {
        'blink_count': blink_count,
        'blink_rate_per_min': blink_rate,
        'ibi_mean': np.nanmean(ibis),
        'ibi_std': np.nanstd(ibis),
        'ibi_entropy': entropy(np.histogram(ibis[~np.isnan(ibis)], bins=10)[0] + 1e-10) if len(ibis[~np.isnan(ibis)]) > 1 else np.nan,
        'blink_duration_mean': np.mean(blink_durations),
        'blink_duration_std': np.std(blink_durations),
        'blink_amplitude_mean': np.mean(blink_amps),
        'blink_amplitude_std': np.std(blink_amps),
        'blink_regularity': np.var(ibis)
    }

# --- Remove Blink Segments from VEOG ---
def remove_blink_segments(signal_data, blink_events, sfreq, removal_ms=100):
    cleaned = signal_data.copy()
    removal_samples = int(removal_ms / 1000 * sfreq)
    for b in blink_events:
        start = max(0, b['peak'] - removal_samples)
        end = min(len(cleaned), b['peak'] + removal_samples)
        # Interpolate over the blink segment
        if start > 0 and end < len(cleaned):
            cleaned[start:end] = np.interp(np.arange(start, end), [start-1, end], [cleaned[start-1], cleaned[end]])
        else:
            cleaned[start:end] = 0
    return cleaned

# --- Extended Feature Extraction for Epochs ---
def detect_saccades(epoch_data, sfreq, velocity_thresh_factor=6, min_duration_ms=10):
    """Detect saccades using velocity thresholding."""
    velocity = np.diff(epoch_data) * sfreq  # first derivative, in uV/sec
    velocity = np.concatenate([[0], velocity])  # pad to match length
    mad = np.median(np.abs(velocity - np.median(velocity)))
    thresh = velocity_thresh_factor * mad if mad > 0 else velocity_thresh_factor * np.std(velocity)
    above = np.abs(velocity) > thresh
    saccades = []
    in_saccade = False
    for i, val in enumerate(above):
        if val and not in_saccade:
            start = i
            in_saccade = True
        elif not val and in_saccade:
            end = i
            if (end - start) / sfreq * 1000 >= min_duration_ms:
                amp = np.abs(epoch_data[end-1] - epoch_data[start])
                dur = (end - start) / sfreq
                peak_vel = np.max(np.abs(velocity[start:end]))
                saccades.append({'start': start, 'end': end, 'amplitude': amp, 'duration': dur, 'peak_velocity': peak_vel})
            in_saccade = False
    if in_saccade:
        end = len(epoch_data)
        if (end - start) / sfreq * 1000 >= min_duration_ms:
            amp = np.abs(epoch_data[end-1] - epoch_data[start])
            dur = (end - start) / sfreq
            peak_vel = np.max(np.abs(velocity[start:end]))
            saccades.append({'start': start, 'end': end, 'amplitude': amp, 'duration': dur, 'peak_velocity': peak_vel})
    return saccades

def extract_saccade_features(saccades, sfreq, epoch_len):
    if not saccades:
        return {
            'saccade_count': 0,
            'saccade_rate': 0,
            'mean_saccade_amplitude': np.nan,
            'mean_saccade_duration': np.nan,
            'max_saccade_velocity': np.nan,
            'ISI_std': np.nan
        }
    amps = np.array([s['amplitude'] for s in saccades])
    durs = np.array([s['duration'] for s in saccades])
    vels = np.array([s['peak_velocity'] for s in saccades])
    times = np.array([s['start'] for s in saccades]) / sfreq
    isis = np.diff(times) if len(times) > 1 else np.array([np.nan])
    return {
        'saccade_count': len(saccades),
        'saccade_rate': len(saccades) / (epoch_len / sfreq),
        'mean_saccade_amplitude': np.mean(amps),
        'mean_saccade_duration': np.mean(durs),
        'max_saccade_velocity': np.max(vels),
        'ISI_std': np.nanstd(isis)
    }

def extract_gaze_features(epoch_data, saccades, sfreq):
    """Extract fixation/gaze features between saccades."""
    if not saccades:
        return {
            'fixation_count': 1,
            'mean_fixation_duration': len(epoch_data) / sfreq,
            'fixation_instability': np.std(epoch_data),
            'fixation_drift': np.polyfit(np.arange(len(epoch_data)), epoch_data, 1)[0]
        }
    fix_starts = [0] + [s['end'] for s in saccades]
    fix_ends = [s['start'] for s in saccades] + [len(epoch_data)]
    fix_durations = []
    fix_instabilities = []
    fix_drifts = []
    for start, end in zip(fix_starts, fix_ends):
        if end > start:
            seg = epoch_data[start:end]
            fix_durations.append((end - start) / sfreq)
            fix_instabilities.append(np.std(seg))
            if len(seg) > 1:
                drift = np.polyfit(np.arange(len(seg)), seg, 1)[0]
            else:
                drift = 0
            fix_drifts.append(drift)
    return {
        'fixation_count': len(fix_durations),
        'mean_fixation_duration': np.mean(fix_durations) if fix_durations else np.nan,
        'fixation_instability': np.mean(fix_instabilities) if fix_instabilities else np.nan,
        'fixation_drift': np.mean(fix_drifts) if fix_drifts else np.nan
    }

def extract_epoch_features(epoch_data, sfreq):
    features = {}
    features['mean'] = np.mean(epoch_data)
    features['std'] = np.std(epoch_data)
    features['skewness'] = skew(epoch_data)
    features['kurtosis'] = kurtosis(epoch_data)
    features['entropy'] = entropy(np.histogram(epoch_data, bins=20)[0] + 1e-10)
    features['zero_crossings'] = calculate_zero_crossings(epoch_data)
    features['form_factor'] = calculate_form_factor(epoch_data)
    # Frequency domain: wavelet power in subbands
    bands = [(0.1,1.25), (1.25,2.5), (2.5,5), (5,10), (10,20)]
    for i, (low, high) in enumerate(bands):
        power = calculate_power_bands(epoch_data, sfreq, bands={f'subb_{i+1}': (low, high)})
        features.update(power)
    features['spectral_entropy'] = calculate_spectral_entropy(epoch_data, sfreq)
    # --- Saccade and Gaze Features ---
    saccades = detect_saccades(epoch_data, sfreq)
    saccade_feats = extract_saccade_features(saccades, sfreq, len(epoch_data))
    features.update(saccade_feats)
    gaze_feats = extract_gaze_features(epoch_data, saccades, sfreq)
    features.update(gaze_feats)
    return features

def calculate_spectral_entropy(signal_data, sfreq):
    """Calculate spectral entropy"""
    freqs, psd = signal.welch(signal_data, sfreq, nperseg=min(256, len(signal_data)//4))
    # Normalize PSD to create a probability distribution
    psd_norm = psd / np.sum(psd)
    # Remove zeros to avoid log(0)
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) > 0:
        return -np.sum(psd_norm * np.log2(psd_norm))
    else:
        return 0

def create_epochs(signal_data, sfreq, epoch_duration, overlap=0.5):
    """Create epochs from continuous signal"""
    epoch_samples = int(epoch_duration * sfreq)
    step_samples = int(epoch_samples * (1 - overlap))
    
    epochs = []
    start_idx = 0
    
    while start_idx + epoch_samples <= len(signal_data):
        epoch = signal_data[start_idx:start_idx + epoch_samples]
        epochs.append(epoch)
        start_idx += step_samples
    
    return epochs

# --- New: Gather all EEG files and labels from all datasets ---
def gather_all_eeg_files():
    eeg_files = []
    # 1. Iowa (BrainVision)
    iowa_dir = 'iowa'
    if os.path.exists(iowa_dir):
        for fname in os.listdir(iowa_dir):
            if fname.endswith('.eeg'):
                base = fname[:-4]
                vhdr = os.path.join(iowa_dir, base + '.vhdr')
                vmrk = os.path.join(iowa_dir, base + '.vmrk')
                eeg = os.path.join(iowa_dir, fname)
                if os.path.exists(vhdr) and os.path.exists(vmrk):
                    label = 1 if base.lower().startswith('pd') else 0
                    eeg_files.append({
                        'dataset': 'iowa',
                        'subject': base,
                        'file_path': vhdr,  # MNE loads BrainVision from .vhdr
                        'label': label
                    })
    # 2. data31 (BDF)
    data31_dir = 'data31'
    if os.path.exists(data31_dir):
        for subj in os.listdir(data31_dir):
            subj_path = os.path.join(data31_dir, subj)
            if subj.startswith('sub-hc'):
                eeg_dir = os.path.join(subj_path, 'ses-hc', 'eeg')
                label = 0
            elif subj.startswith('sub-pd'):
                eeg_dir = os.path.join(subj_path, 'ses-on', 'eeg')
                label = 1
            else:
                continue
            if os.path.exists(eeg_dir):
                for fname in os.listdir(eeg_dir):
                    if fname.endswith('.bdf'):
                        eeg_files.append({
                            'dataset': 'data31',
                            'subject': subj,
                            'file_path': os.path.join(eeg_dir, fname),
                            'label': label
                        })
    # 3. ds004584-download (EEGLAB)
    ds_dir = 'ds004584-download'
    if os.path.exists(ds_dir):
        subjs = [d for d in os.listdir(ds_dir) if d.startswith('sub-')]
        subjs.sort()
        for subj in subjs:
            subj_num = int(subj.split('-')[1])
            eeg_dir = os.path.join(ds_dir, subj, 'eeg')
            if os.path.exists(eeg_dir):
                for fname in os.listdir(eeg_dir):
                    if fname.endswith('_task-Rest_eeg.set'):
                        label = 1 if subj_num <= 100 else 0
                        eeg_files.append({
                            'dataset': 'ds004584',
                            'subject': subj,
                            'file_path': os.path.join(eeg_dir, fname),
                            'label': label
                        })
    return eeg_files

def process_multiple_subjects(subject_list=None, max_subjects=None):
    """Process multiple subjects and combine all features into one CSV"""
    eeg_file_list = gather_all_eeg_files()
    if max_subjects:
        eeg_file_list = eeg_file_list[:max_subjects]

    all_subject_features = []
    all_blink_features = []
    all_blink_epoch_features = []

    for idx, entry in enumerate(eeg_file_list):
        subject = entry['subject']
        file_path = entry['file_path']
        label = entry['label']
        dataset = entry['dataset']
        try:
            print(f"\nProcessing {subject} from {dataset}...")
            # Load EEG data
            if dataset == 'iowa':
                raw = mne.io.read_raw_brainvision(file_path, preload=True)
            elif dataset == 'data31':
                raw = mne.io.read_raw_bdf(file_path, preload=True)
            elif dataset == 'ds004584':
                raw = mne.io.read_raw_eeglab(file_path, preload=True)
            else:
                print(f"Unknown dataset for {subject}, skipping...")
                continue

            # 2. Bandpass filter (0.5–15 Hz)
            raw_filt = raw.copy().filter(0.5, 15., fir_design='firwin')

            # 3. VEOG: Bipolar derivation (Fp1-Fp2, or fallback)
            if 'Fp1' in raw.ch_names and 'Fp2' in raw.ch_names:
                veog_chs = ('Fp1', 'Fp2')
                veog_signal = (raw_filt.get_data(picks='Fp1')[0] + raw_filt.get_data(picks='Fp2')[0]) / 2.0
                veog_label = 'Fp1+Fp2_avg'
            elif 'Fp1' in raw.ch_names:
                veog_chs = ('Fp1',)
                veog_signal = raw_filt.get_data(picks='Fp1')[0]
                veog_label = 'Fp1'
            elif 'Fp2' in raw.ch_names:
                veog_chs = ('Fp2',)
                veog_signal = raw_filt.get_data(picks='Fp2')[0]
                veog_label = 'Fp2'
            else:
                print(f"No VEOG channel found for {subject}, skipping...")
                continue

            # 4. Mean removal
            veog_signal = veog_signal - np.mean(veog_signal)

            # 5. Wavelet denoising
            veog_denoised = wavelet_denoise(veog_signal)

            # 6. Blink detection (matched filter)
            blink_template = generate_blink_template(raw.info['sfreq'])
            blink_events, matched_output = detect_blinks_matched_filter(veog_denoised, raw.info['sfreq'], blink_template)

            # --- New: Epoch the matched filter output and extract blink features per epoch ---
            matched_epochs = create_epochs(matched_output, raw.info['sfreq'], EPOCH_DURATION, OVERLAP)
            blink_epoch_features = []
            for i, epoch in enumerate(matched_epochs):
                # Detect blinks in this epoch only
                epoch_blink_events, _ = detect_blinks_matched_filter(epoch, raw.info['sfreq'], blink_template)
                epoch_blink_features = extract_blink_features(epoch_blink_events, raw.info['sfreq'], len(epoch))
                epoch_blink_features['subject'] = subject
                epoch_blink_features['epoch'] = i
                epoch_blink_features['signal_type'] = 'BLINK_MATCHED'
                epoch_blink_features['label'] = label
                blink_epoch_features.append(epoch_blink_features)

            # 7. Blink feature extraction (whole signal, for summary)
            blink_features = extract_blink_features(blink_events, raw.info['sfreq'], len(veog_denoised))
            blink_features['subject'] = subject
            blink_features['label'] = label
            all_blink_features.append(blink_features)

            # 8. Save blink events and blink features to CSV
            blink_events_df = pd.DataFrame(blink_events)
            blink_events_df['subject'] = subject
            blink_events_df.to_csv(f'{subject}_blink_events.csv', index=False)
            pd.DataFrame([blink_features]).to_csv(f'{subject}_blink_features_summary.csv', index=False)

            # 9. Remove ±100 ms around blink peaks (interpolate)
            veog_cleaned = remove_blink_segments(veog_denoised, blink_events, raw.info['sfreq'], BLINK_REMOVAL_MS)

            # 10. Epoch cleaned VEOG
            sfreq = raw.info['sfreq']
            veog_epochs = create_epochs(veog_cleaned, sfreq, EPOCH_DURATION, OVERLAP)

            # 11. Extract epoch features (VEOG)
            veog_epoch_features = []
            for i, epoch in enumerate(veog_epochs):
                feat = extract_epoch_features(epoch, sfreq)
                feat['subject'] = subject
                feat['epoch'] = i
                feat['signal_type'] = 'VEOG_CLEANED'
                feat['channel'] = veog_label
                feat['label'] = label
                all_subject_features.append(feat)
                veog_epoch_features.append(feat)

            # 12. HEOG: Bipolar derivation (AF7-AF8 or F7-F8)
            heog_chs = None
            for pair in [('AF7', 'AF8'), ('F7', 'F8')]:
                if pair[0] in raw.ch_names and pair[1] in raw.ch_names:
                    heog_chs = pair
                    break
            if heog_chs is None:
                print(f"No HEOG channel pair found for {subject}, skipping...")
                continue
            heog_signal = raw_filt.get_data(picks=heog_chs[0])[0] - raw_filt.get_data(picks=heog_chs[1])[0]

            # 13. Mean removal
            heog_signal = heog_signal - np.mean(heog_signal)

            # 14. Wavelet denoising
            heog_denoised = wavelet_denoise(heog_signal)

            # 15. Epoch denoised HEOG
            heog_epochs = create_epochs(heog_denoised, sfreq, EPOCH_DURATION, OVERLAP)

            # 16. Extract epoch features (HEOG)
            for i, epoch in enumerate(heog_epochs):
                feat = extract_epoch_features(epoch, sfreq)
                feat['subject'] = subject
                feat['epoch'] = i
                feat['signal_type'] = 'HEOG'
                feat['channel'] = f"{heog_chs[0]}-{heog_chs[1]}"
                feat['label'] = label
                all_subject_features.append(feat)

            # 17. Save all features to CSV (per subject)
            # No per-subject CSV saving; only combine at the end
            all_subject_features.extend(veog_epoch_features)
            all_subject_features.extend([f for f in all_subject_features if f['subject'] == subject and f['signal_type'] == 'HEOG'])
            all_blink_epoch_features.extend(blink_epoch_features)
            print(f"Processed features for {subject}")

        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
            continue

    # Save two combined files: one for EOG (VEOG_CLEANED, HEOG), one for BLINK_MATCHED (epochwise)
    eog_epochwise = [f for f in all_subject_features if f.get('signal_type') in ('VEOG_CLEANED', 'HEOG')]
    blink_epochwise = all_blink_epoch_features
    if eog_epochwise:
        eog_df = pd.DataFrame(eog_epochwise)
        eog_filename = "all_subjects_eog_epochwise_features.csv"
        eog_df.to_csv(eog_filename, index=False)
        print(f"\nEOG epochwise features saved to {eog_filename}")
        print(f"Total EOG feature vectors: {len(eog_df)}")
        print(f"Subjects processed: {eog_df['subject'].nunique()}")
    else:
        print("No EOG features extracted from any subject")
    if blink_epochwise:
        blink_df = pd.DataFrame(blink_epochwise)
        blink_filename = "all_subjects_blink_epochwise_features.csv"
        blink_df.to_csv(blink_filename, index=False)
        print(f"Blink epochwise features saved to {blink_filename}")
        print(f"Total blink feature vectors: {len(blink_df)}")
        print(f"Subjects processed: {blink_df['subject'].nunique()}")
    else:
        print("No blink features extracted from any subject")

if _name_ == "_main_":
    process_multiple_subjects()
    