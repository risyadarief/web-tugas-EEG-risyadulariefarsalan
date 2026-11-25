import io
import os
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import loadmat
from scipy.signal import firwin, filtfilt, medfilt
from scipy.stats import chi2

# ==============================================================================
# KONFIGURASI HALAMAN WEB & CSS KUSTOM
# ==============================================================================
st.set_page_config(
    page_title="EEG Signal Processing Pipeline",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Daftar pilihan di dalam dropdown popover rata kiri */
    div[data-testid="stPopoverBody"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.75rem !important;
    }

    /* Memperlebar modal pop-up menjadi hampir full screen */
    div[data-testid="stDialog"] div[role="dialog"] {
        width: 96vw !important;
        max-width: 96vw !important;
    }

    /* Slot kosong berbentuk kertas A4 */
    .a4-paper-slot {
        width: 100%;
        max-width: 794px;
        aspect-ratio: 210 / 297;
        margin: 16px auto 32px auto;
        background-color: #ffffff;
        color: #555555;
        border: 1px solid #d0d7de;
        border-radius: 4px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 48px;
        box-sizing: border-box;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# INISIALISASI SESSION STATE (SWITCH MENU & DROPDOWN)
# ==============================================================================
MENU_OPTIONS = ["1. Paper", "2. Database", "3. Upload"]

if "active_menu" not in st.session_state:
    st.session_state.active_menu = "1. Paper"

for opt in MENU_OPTIONS:
    switch_key = f"switch_{opt}"
    if switch_key not in st.session_state:
        st.session_state[switch_key] = (st.session_state.active_menu == opt)

if "db_file_idx" not in st.session_state:
    st.session_state.db_file_idx = 0
if "db_ch_idx" not in st.session_state:
    st.session_state.db_ch_idx = 0
if "up_ch_idx" not in st.session_state:
    st.session_state.up_ch_idx = 0


def on_switch_menu(chosen_menu):
    st.session_state.active_menu = chosen_menu
    for opt in MENU_OPTIONS:
        st.session_state[f"switch_{opt}"] = (opt == chosen_menu)


EEG_CHANNEL_FULL_NAMES = {
    'FP1': 'Left Frontopolar',
    'FP2': 'Right Frontopolar',
    'FPZ': 'Midline Frontopolar',
    'AF3': 'Left Anterior Frontal',
    'AF4': 'Right Anterior Frontal',
    'F7':  'Left Lateral Frontal',
    'F3':  'Left Frontal',
    'FZ':  'Midline Frontal',
    'F4':  'Right Frontal',
    'F8':  'Right Lateral Frontal',
    'FC5': 'Left Fronto-Central Lateral',
    'FC1': 'Left Fronto-Central',
    'FC2': 'Right Fronto-Central',
    'FC6': 'Right Fronto-Central Lateral',
    'T3':  'Left Mid-Temporal',
    'T7':  'Left Mid-Temporal',
    'C3':  'Left Central',
    'CZ':  'Midline Central',
    'C4':  'Right Central',
    'T4':  'Right Mid-Temporal',
    'T8':  'Right Mid-Temporal',
    'CP5': 'Left Centro-Parietal Lateral',
    'CP1': 'Left Centro-Parietal',
    'CP2': 'Right Centro-Parietal',
    'CP6': 'Right Centro-Parietal Lateral',
    'T5':  'Left Posterior Temporal',
    'P7':  'Left Posterior Temporal',
    'P3':  'Left Parietal',
    'PZ':  'Midline Parietal',
    'P4':  'Right Parietal',
    'T6':  'Right Posterior Temporal',
    'P8':  'Right Posterior Temporal',
    'PO3': 'Left Parieto-Occipital',
    'PO4': 'Right Parieto-Occipital',
    'O1':  'Left Occipital',
    'OZ':  'Midline Occipital',
    'O2':  'Right Occipital',
    'A1':  'Left Auricular',
    'A2':  'Right Auricular',
    'M1':  'Left Mastoid',
    'M2':  'Right Mastoid',
}


def format_channel_label(ch_code):
    clean_key = re.sub(r'^(EEG|CH)[_\-\s]*', '', str(ch_code).strip(), flags=re.IGNORECASE)
    clean_key = re.split(r'[\-\s]', clean_key)[0].upper()
    full_name = EEG_CHANNEL_FULL_NAMES.get(clean_key)
    return f"{ch_code} - {full_name}" if full_name else str(ch_code)


def get_local_dataset_files(root_dir):
    valid_exts = ('.dat', '.txt', '.csv', '.mat')
    ignored_files = {'requirements.txt', 'readme.txt', 'license.txt'}
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in sorted(files):
            if f.lower().endswith(valid_exts) and f.lower() not in ignored_files:
                file_list.append(os.path.join(root, f))
    return file_list


@st.cache_data(show_spinner=False)
def create_database_zip(file_paths, root_dir):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            arcname = os.path.relpath(path, root_dir)
            zf.write(path, arcname)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ==============================================================================
# DROPDOWN TANPA KEYBOARD & KURSOR (HANYA BISA DIPILIH)
# ==============================================================================
def readonly_dropdown(label, options, state_key, reset_key=None):
    st.markdown(
        f"<div style='font-size:0.875rem; margin-bottom:0.35rem;'>{label}</div>",
        unsafe_allow_html=True,
    )
    idx = st.session_state.get(state_key, 0)
    if idx >= len(options):
        idx = 0
        st.session_state[state_key] = 0

    with st.popover(options[idx], use_container_width=True):
        box_height = min(280, max(90, len(options) * 44))
        with st.container(height=box_height, border=False):
            for i, opt in enumerate(options):
                is_selected = (i == idx)
                if st.button(
                    opt,
                    key=f"{state_key}_opt_{i}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state[state_key] = i
                    if reset_key is not None:
                        st.session_state[reset_key] = 0
                    st.rerun()

    return st.session_state[state_key]


# ==============================================================================
# POP-UP FULL SCREEN GAMBAR
# ==============================================================================
@st.dialog("Full Screen EEG Plot", width="large")
def show_fullscreen_plot(png_bytes, caption_text):
    st.image(png_bytes, caption=caption_text, use_container_width=True)


# ==============================================================================
# PEMBACA DATASET DARI BYTES
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_eeg_from_bytes(raw_bytes, filename, default_fs=250):
    fs = default_fs
    ch_names = None
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.dat':
        header_preview = raw_bytes[:300].decode('latin1', errors='ignore')

        if 'HeaderLen=' in header_preview and 'SourceCh=' in header_preview:
            header_len = int(re.search(r'HeaderLen=\s*(\d+)', header_preview).group(1))
            source_ch = int(re.search(r'SourceCh=\s*(\d+)', header_preview).group(1))
            state_len_match = re.search(r'StatevectorLen=\s*(\d+)', header_preview)
            state_len = int(state_len_match.group(1)) if state_len_match else 0

            full_header = raw_bytes[:header_len].decode('latin1', errors='ignore')

            sr_match = re.search(r'SamplingRate=\s*(\d+)', full_header)
            if sr_match:
                fs = int(sr_match.group(1))

            ch_match = re.search(r'ChannelNames=\s*(\d+)\s+([^\r\n/]+)', full_header)
            if ch_match:
                n_names = int(ch_match.group(1))
                tokens = ch_match.group(2).strip().split()
                if n_names > 0 and len(tokens) >= n_names:
                    ch_names = tokens[:n_names]

            fmt_match = re.search(r'DataFormat=\s*(\w+)', full_header)
            fmt_str = fmt_match.group(1).lower() if fmt_match else 'int16'
            dtype = np.float32 if 'float32' in fmt_str else (np.int32 if 'int32' in fmt_str else np.int16)

            data_bytes = raw_bytes[header_len:]
            bytes_per_sample = (source_ch * np.dtype(dtype).itemsize) + state_len
            n_samp = len(data_bytes) // bytes_per_sample
            raw_matrix = np.frombuffer(
                data_bytes[:n_samp * bytes_per_sample], dtype=np.uint8
            ).reshape((n_samp, bytes_per_sample))
            eeg_bytes = raw_matrix[:, :source_ch * np.dtype(dtype).itemsize].tobytes()
            eeg_data = np.frombuffer(eeg_bytes, dtype=dtype).reshape((n_samp, source_ch)).T.astype(float)
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python', comment='%', encoding='latin1')
            num_df = df.select_dtypes(include=[np.number])
            eeg_data = num_df.values.T
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T

    elif ext in ['.txt', '.csv']:
        df = pd.read_csv(io.BytesIO(raw_bytes), comment='%', skipinitialspace=True)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 17:
            eeg_data = num_df.iloc[:, 1:17].values.T
            ch_names = list(num_df.columns[1:17])
        elif num_df.shape[1] >= 9:
            eeg_data = num_df.iloc[:, 1:9].values.T
            ch_names = list(num_df.columns[1:9])
        else:
            eeg_data = num_df.values.T
            ch_names = list(num_df.columns)

    elif ext == '.mat':
        mat = loadmat(io.BytesIO(raw_bytes))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        eeg_data = np.array(mat[keys[0]], dtype=float)
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if not ch_names or len(ch_names) != eeg_data.shape[0]:
        standard_16 = ['Fp1', 'F3', 'F7', 'T3', 'T5', 'O1', 'C3', 'P3',
                       'Fp2', 'F4', 'F8', 'T4', 'T6', 'O2', 'C4', 'P4']
        ch_names = standard_16 if eeg_data.shape[0] == 16 else [f"Ch{i+1}" for i in range(eeg_data.shape[0])]

    eeg_data = eeg_data[:, ~np.isnan(eeg_data).any(axis=0)]
    non_zero_ch = np.std(eeg_data, axis=1) > 0
    if np.any(non_zero_ch):
        eeg_data = eeg_data[non_zero_ch]
        ch_names = [name for name, keep in zip(ch_names, non_zero_ch) if keep]

    return eeg_data, fs, ch_names


# ==============================================================================
# PIPELINE DSP: MEDIAN + FIR HAMMING + SSP + MAHALANOBIS DISTANCE
# ==============================================================================
@st.cache_data(show_spinner=False)
def process_pipeline(eeg_raw, fs):
    n_ch, n_samp = eeg_raw.shape

    # 1. Filtering: Median Filter + FIR Hamming Bandpass (0.5 - 45 Hz)
    filtered_data = np.zeros_like(eeg_raw)
    nyquist = 0.5 * fs
    numtaps = 251 if n_samp > 760 else max(11, (n_samp // 4) | 1)
    taps = firwin(numtaps, [0.5 / nyquist, 45.0 / nyquist], window='hamming', pass_zero='bandpass')
    for ch in range(n_ch):
        med_clean = medfilt(eeg_raw[ch], kernel_size=3)
        filtered_data[ch] = filtfilt(taps, [1.0], med_clean)

    # 2. Preprocessing A: Signal Space Projection (SSP)
    mean_ch = np.mean(filtered_data, axis=1, keepdims=True)
    data_centered = filtered_data - mean_ch
    U, _, _ = np.linalg.svd(data_centered, full_matrices=False)
    U_artifact = U[:, :1]
    P_ssp = np.eye(n_ch) - np.dot(U_artifact, U_artifact.T)
    eeg_ssp = np.dot(P_ssp, data_centered) + mean_ch

    # 3. Preprocessing B: Mahalanobis Distance Cleaning
    X = eeg_ssp.T
    mean_vec = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False) + np.eye(n_ch) * 1e-6
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    diff = X - mean_vec
    md = np.sqrt(np.sum(np.dot(diff, inv_cov_matrix) * diff, axis=1))

    threshold = np.sqrt(chi2.ppf(0.999, df=n_ch))
    outlier_mask = md > threshold

    eeg_final = eeg_ssp.copy()
    valid_idx = np.where(~outlier_mask)[0]
    outlier_idx = np.where(outlier_mask)[0]
    if len(outlier_idx) > 0 and len(valid_idx) > 0:
        for ch in range(n_ch):
            eeg_final[ch, outlier_idx] = np.interp(outlier_idx, valid_idx, eeg_ssp[ch, valid_idx])

    return filtered_data, eeg_final, md, threshold, int(np.sum(outlier_mask))


def create_eeg_figure(t, eeg_raw, eeg_filtered, eeg_final, md_scores, md_thresh, n_outliers,
                      file_idx, total_files, file_name, ch_idx, n_ch, ch_label, fs):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.2), sharex=True)
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.10, right=0.96, hspace=0.38)

    header_title = (
        f"({file_idx+1}/{total_files}) {file_name}  |  "
        f"Channel {ch_idx+1}/{n_ch} ({ch_label})  |  fs = {fs} Hz"
    )
    fig.suptitle(header_title, fontsize=10.5, fontweight='bold', y=0.965)

    axes[0].plot(t, eeg_raw[ch_idx], color='gray', linewidth=0.7)
    axes[0].set_title('1. Raw EEG Signal')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(t, eeg_filtered[ch_idx], color='steelblue', linewidth=0.9)
    axes[1].set_title('2. Filtering Result: Median Filter + FIR Hamming Bandpass (0.5 - 45 Hz)')
    axes[1].set_ylabel('Amplitude (uV)')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].plot(t, eeg_final[ch_idx], color='forestgreen', linewidth=0.9)
    axes[2].set_title('3. Preprocessing Result: Signal Space Projection (SSP) & Mahalanobis Distance')
    axes[2].set_ylabel('Amplitude (uV)')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    axes[3].plot(t, md_scores, color='darkorange', linewidth=0.8, label='Mahalanobis Distance')
    axes[3].axhline(md_thresh, color='red', linestyle='--', label=f'Threshold ({md_thresh:.2f}) | Outliers: {n_outliers}')
    axes[3].set_title('4. Multivariate Outlier Detection (Mahalanobis Distance)')
    axes[3].set_ylabel('Distance (D)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle='--', alpha=0.5)

    fig.align_ylabels(axes)
    return fig


def render_signal_output(eeg_raw, fs, ch_names, ch_idx, file_idx, total_files, file_name, key_prefix):
    n_ch, n_samp = eeg_raw.shape
    t = np.arange(n_samp) / fs
    ch_labels = [format_channel_label(c) for c in ch_names]

    with st.spinner("Processing EEG signal..."):
        eeg_filtered, eeg_final, md_scores, md_thresh, n_outliers = process_pipeline(eeg_raw, fs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Channels", n_ch)
    col2.metric("Sampling Rate", f"{fs} Hz")
    col3.metric("Duration", f"{t[-1]:.2f} s")
    col4.metric("Detected Outliers", f"{n_outliers} / {n_samp}")

    fig = create_eeg_figure(
        t, eeg_raw, eeg_filtered, eeg_final, md_scores, md_thresh, n_outliers,
        file_idx, total_files, file_name, ch_idx, n_ch, ch_labels[ch_idx], fs
    )
    st.pyplot(fig, use_container_width=True)

    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    png_bytes = png_buffer.getvalue()
    plt.close(fig)

    clean_stem = os.path.splitext(file_name)[0]
    btn_col1, btn_col2, _ = st.columns([1.4, 1.4, 3.2])
    with btn_col1:
        st.download_button(
            label="Download Plot as PNG",
            data=png_bytes,
            file_name=f"{clean_stem}_Ch{ch_idx+1}_{ch_names[ch_idx]}.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key_prefix}_dl_png",
        )
    with btn_col2:
        if st.button(
            "Full Screen",
            use_container_width=True,
            key=f"{key_prefix}_fullscreen_btn",
        ):
            caption = f"({file_idx+1}/{total_files}) {file_name} | Channel {ch_idx+1}/{n_ch} ({ch_labels[ch_idx]}) | fs = {fs} Hz"
            show_fullscreen_plot(png_bytes, caption)


# ==============================================================================
# SIDEBAR KIRI: JUDUL PIPELINE & 3 SWITCH NAVIGASI
# ==============================================================================
with st.sidebar:
    st.markdown("### EEG Signal Processing")
    st.caption(
        "Pipeline: Median Filter + FIR Hamming Bandpass (0.5–45 Hz) | "
        "Signal Space Projection (SSP) & Mahalanobis Distance"
    )
    st.divider()
    for opt in MENU_OPTIONS:
        st.toggle(
            opt,
            key=f"switch_{opt}",
            on_change=on_switch_menu,
            args=(opt,),
        )

menu = st.session_state.active_menu
base_dir = os.path.dirname(os.path.abspath(__file__))
local_files = get_local_dataset_files(base_dir)


# ==============================================================================
# BAGIAN TENGAH 1: PAPER (SLOT KOSONG KERTAS A4)
# ==============================================================================
if menu == "1. Paper":
    st.markdown(
        """
        <div class="a4-paper-slot">
            <h3 style="margin-bottom: 12px; color: #333333;">Paper Slot (A4)</h3>
            <p style="font-size: 14px; color: #777777;">
                Dokumen paper akan ditampilkan di area ini.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# BAGIAN TENGAH 2: DATABASE
# ==============================================================================
elif menu == "2. Database":
    st.title("EEG Database")

    if not local_files:
        st.warning("No dataset files found in the repository.")
    else:
        zip_data = create_database_zip(local_files, base_dir)
        st.download_button(
            label=f"Download Database ({len(local_files)} files - ZIP)",
            data=zip_data,
            file_name="EEG_Dataset_KAU.zip",
            mime="application/zip",
        )

        st.divider()

        file_names = [os.path.basename(p) for p in local_files]
        total_files = len(local_files)
        file_options = [f"({i+1}/{total_files}) {file_names[i]}" for i in range(total_files)]

        c1, c2 = st.columns([1, 2])
        with c1:
            default_fs = st.number_input(
                "Default Sampling Rate (Hz)",
                min_value=64,
                max_value=2048,
                value=250,
                step=1,
                disabled=True,
                key="db_fs",
            )
        with c2:
            file_idx = readonly_dropdown(
                f"Select File ({total_files} available)",
                file_options,
                state_key="db_file_idx",
                reset_key="db_ch_idx",
            )

        selected_path = local_files[file_idx]
        selected_name = os.path.basename(selected_path)

        try:
            with open(selected_path, 'rb') as f:
                selected_bytes = f.read()

            eeg_raw, fs, ch_names = load_eeg_from_bytes(
                selected_bytes, selected_name, default_fs=int(default_fs)
            )
            n_ch = eeg_raw.shape[0]
            ch_labels = [format_channel_label(c) for c in ch_names]
            ch_options = [f"Channel {i+1}/{n_ch} ({ch_labels[i]})" for i in range(n_ch)]

            ch_idx = readonly_dropdown(
                f"Select Channel ({n_ch} total)",
                ch_options,
                state_key="db_ch_idx",
            )

            st.divider()

            render_signal_output(
                eeg_raw, fs, ch_names, ch_idx, file_idx, total_files, selected_name, key_prefix="db"
            )

        except Exception as e:
            st.error(f"Failed to process file `{selected_name}`: {e}")


# ==============================================================================
# BAGIAN TENGAH 3: UPLOAD (TERBATAS 1 FILE SEMENTARA)
# ==============================================================================
elif menu == "3. Upload":
    st.title("Upload EEG File")
    st.caption("Uploaded files are processed temporarily in memory and are not saved to the database.")

    uploaded_file = st.file_uploader(
        "Drag and drop or choose 1 EEG file (.dat, .txt, .csv, .mat)",
        type=['dat', 'txt', 'csv', 'mat'],
        accept_multiple_files=False,
        key="single_eeg_uploader",
    )

    if uploaded_file is not None:
        st.markdown(
            """
            <style>
            [data-testid="stFileUploaderDropzone"] {
                pointer-events: none !important;
                opacity: 0.45 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.info("1 file is currently active. Click the **×** button on the file above (or refresh the page) to upload a different file.")

        selected_name = uploaded_file.name
        selected_bytes = uploaded_file.getvalue()

        try:
            c1, c2 = st.columns([1, 2])
            with c1:
                default_fs = st.number_input(
                    "Default Sampling Rate (Hz)",
                    min_value=64,
                    max_value=2048,
                    value=250,
                    step=1,
                    disabled=True,
                    key="up_fs",
                )

            eeg_raw, fs, ch_names = load_eeg_from_bytes(
                selected_bytes, selected_name, default_fs=int(default_fs)
            )
            n_ch = eeg_raw.shape[0]
            ch_labels = [format_channel_label(c) for c in ch_names]
            ch_options = [f"Channel {i+1}/{n_ch} ({ch_labels[i]})" for i in range(n_ch)]

            with c2:
                ch_idx = readonly_dropdown(
                    f"Select Channel ({n_ch} total)",
                    ch_options,
                    state_key="up_ch_idx",
                )

            st.divider()

            render_signal_output(
                eeg_raw, fs, ch_names, ch_idx, 0, 1, selected_name, key_prefix="up"
            )

        except Exception as e:
            st.error(f"Failed to process file `{selected_name}`: {e}")
    else:
        st.info("Upload 1 EEG file (`.dat`, `.txt`, `.csv`, `.mat`) above to view the processed signal.")