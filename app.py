
import os
import hashlib
import subprocess
from pathlib import Path
from model_downloader import ensure_model
import pandas as pd
import streamlit as st

from main import run_pipeline

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TactiVision AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_videos"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- DARK THEME ----------------

st.markdown("""
<style>
    .stApp {
        background: #0B1220;
        color: #E8EEF7;
    }

    [data-testid="stSidebar"] {
        background: #111D30;
        border-right: 1px solid #24344B;
    }

    h1, h2, h3 {
        color: #F1F5FA !important;
    }

    [data-testid="stMetric"] {
        background: #17263B;
        border: 1px solid #293C55;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #A8B8CC;
    }

    .hero {
        background: linear-gradient(110deg, #162A42, #102238);
        border: 1px solid #28465E;
        border-radius: 18px;
        padding: 26px;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        color: #F5F8FC;
    }

    .hero-subtitle {
        color: #A7C0D4;
        font-size: 15px;
    }

    .stButton > button[kind="primary"] {
        background: #0FA987;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton > button[kind="primary"]:hover {
        background: #0BC69A;
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: #142238;
        border-radius: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------

def get_match_dir(file_bytes):
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    return OUTPUT_DIR / f"match_{file_hash}"


def convert_to_mp4(source, destination):
    """Convert AVI to browser-compatible MP4 using FFmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(source),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_heatmaps(directory):
    if not directory.exists():
        return []
    return sorted(directory.glob("*.png"))


def load_stats(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, ValueError):
            return pd.DataFrame()
    return pd.DataFrame()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚽ TactiVision AI")
    st.caption("SPORTS INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🎬 Match Analysis",
            "📊 Player Statistics",
            "🔥 Heatmaps",
            "📁 Match History",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Powered by YOLO • ByteTrack • OpenCV")
    st.caption("Deep Learning Project")


# ---------------- SESSION STATE ----------------

if "selected_match" not in st.session_state:
    st.session_state.selected_match = None

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None


def current_match():
    selected = st.session_state.selected_match
    if selected:
        path = OUTPUT_DIR / selected
        if path.is_dir():
            return path
    return None


# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">TactiVision AI</div>
        <div class="hero-subtitle">
            Sports Player Tracking & Tactical Event Analyser
        </div>
        <p style="color:#D3E0EE;margin-top:14px;">
            Analyze football footage with deep learning.
            Explore player movements, tracking data,
            team heatmaps and performance statistics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    matches = [
        p for p in OUTPUT_DIR.iterdir()
        if p.is_dir() and (p / "match_stats.csv").exists()
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Analyzed Matches", len(matches))
    col2.metric("Detection Model", "YOLOv5su")
    col3.metric("Tracking System", "ByteTrack")

    st.subheader("Analysis Capabilities")

    a, b, c = st.columns(3)
    with a:
        st.info("🎯 **Player Detection**\n\nDetect players, goalkeepers, referees and the ball.")
    with b:
        st.info("🏃 **Player Tracking**\n\nTrack player identities and estimate movements.")
    with c:
        st.info("📈 **Match Analytics**\n\nExplore heatmaps, speed, distance and possession.")

    st.caption("Select Match Analysis in the sidebar to upload a video.")


# ---------------- MATCH ANALYSIS ----------------

elif page == "🎬 Match Analysis":
    st.title("🎬 Match Analysis")
    st.caption("Upload a football match clip and run the AI analysis pipeline.")

    uploaded = st.file_uploader(
        "Upload football footage",
        type=["mp4", "avi"],
        help="Maximum 200 MB",
    )

    if uploaded is not None:
        if uploaded.size > 200 * 1024 * 1024:
            st.error("Video exceeds the 200 MB limit.")
            st.stop()

        file_bytes = uploaded.getvalue()
        match_dir = get_match_dir(file_bytes)
        match_id = match_dir.name

        input_path = match_dir / ("input" + Path(uploaded.name).suffix.lower())
        output_avi = match_dir / "output_video.avi"
        output_mp4 = match_dir / "output_video.mp4"
        heatmap_dir = match_dir / "heatmaps"
        stats_path = match_dir / "match_stats.csv"

        st.subheader("Video Preview")
        st.video(file_bytes)

        already_processed = (
            output_avi.exists()
            and stats_path.exists()
        )

        if already_processed:
            st.success("This video has already been analyzed.")

        if st.button(
            "🚀 Analyze Match",
            type="primary",
            disabled=already_processed,
        ):
            match_dir.mkdir(parents=True, exist_ok=True)

            with open(input_path, "wb") as f:
                f.write(file_bytes)

            status = st.status("Analyzing football match...", expanded=True)

            try:
                with status:
                    st.write("Running YOLO detection and player tracking...")
                    model_path = ensure_model()
                    run_pipeline(
                        input_path=str(input_path),
                        output_video_path=str(output_avi),
                        heatmap_dir=str(heatmap_dir),
                        stats_path=str(stats_path),
                        model_path=model_path,
                        track_stub=None,
                        cam_stub=None,
                    )

                    st.write("Preparing browser-compatible video...")

                    if not convert_to_mp4(output_avi, output_mp4):
                        st.warning(
                            "MP4 conversion unavailable. "
                            "Install FFmpeg to enable browser playback."
                        )

                    status.update(
                        label="Analysis completed!",
                        state="complete",
                    )

                st.session_state.selected_match = match_id
                st.success("Your match has been analyzed successfully!")

            except Exception as e:
                status.update(
                    label="Analysis failed",
                    state="error",
                )
                st.error(f"Pipeline error: {e}")
                st.stop()

        if already_processed or stats_path.exists():
            st.session_state.selected_match = match_id

            st.divider()
            st.subheader("Analysis Results")

            if output_mp4.exists():
                st.video(str(output_mp4))
            elif output_avi.exists():
                st.info("Download the AVI video to view it in VLC.")

            if output_avi.exists():
                with open(output_avi, "rb") as f:
                    st.download_button(
                        "⬇ Download Annotated Video",
                        data=f,
                        file_name="annotated_match.avi",
                        mime="video/x-msvideo",
                    )

            df = load_stats(stats_path)
            if not df.empty:
                st.subheader("Player Statistics")
                st.dataframe(df, width="stretch")

            heatmaps = get_heatmaps(heatmap_dir)
            if heatmaps:
                st.subheader("Team Heatmaps")
                cols = st.columns(min(len(heatmaps), 2))
                for i, image in enumerate(heatmaps):
                    with cols[i % len(cols)]:
                        st.image(str(image), width="stretch")


# ---------------- PLAYER STATISTICS ----------------

elif page == "📊 Player Statistics":
    st.title("📊 Player Statistics")

    match_dir = current_match()

    if match_dir is None:
        st.info("Analyze or select a match first.")
    else:
        df = load_stats(match_dir / "match_stats.csv")

        if df.empty:
            st.warning("No statistics available.")
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Tracked Players", len(df))
            col2.metric(
                "Maximum Speed",
                f"{df['max_speed_kmh'].max():.1f} km/h",
            )
            col3.metric(
                "Total Distance",
                f"{df['total_distance_m'].sum():.1f} m",
            )

            st.subheader("Player Performance Table")
            st.dataframe(
                df.sort_values(
                    "max_speed_kmh", ascending=False
                ),
                width="stretch",
            )

            st.subheader("Distance Covered by Player")
            chart_df = df.set_index("player_id")
            st.bar_chart(chart_df["total_distance_m"])

            st.subheader("Maximum Player Speed")
            st.bar_chart(chart_df["max_speed_kmh"])

            st.download_button(
                "⬇ Download Statistics CSV",
                data=df.to_csv(index=False),
                file_name="match_stats.csv",
                mime="text/csv",
            )


# ---------------- HEATMAPS ----------------

elif page == "🔥 Heatmaps":
    st.title("🔥 Player Movement Heatmaps")

    match_dir = current_match()

    if match_dir is None:
        st.info("Analyze or select a match first.")
    else:
        heatmaps = get_heatmaps(match_dir / "heatmaps")

        if not heatmaps:
            st.warning("No heatmaps available for this match.")
        else:
            for image in heatmaps:
                st.subheader(image.stem.replace("_", " ").title())
                st.image(str(image), width="stretch")

                with open(image, "rb") as f:
                    st.download_button(
                        f"Download {image.name}",
                        data=f,
                        file_name=image.name,
                        mime="image/png",
                        key=f"download_{image.name}",
                    )


# ---------------- MATCH HISTORY ----------------

elif page == "📁 Match History":
    st.title("📁 Match History")

    matches = sorted(
        [
            p for p in OUTPUT_DIR.iterdir()
            if p.is_dir() and (p / "match_stats.csv").exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not matches:
        st.info("No analyzed matches found yet.")
    else:
        st.write(f"Saved analyses: {len(matches)}")

        selected = st.selectbox(
            "Select a previously analyzed match",
            options=matches,
            format_func=lambda p: p.name,
        )

        if st.button("Open Selected Match", type="primary"):
            st.session_state.selected_match = selected.name
            st.success(f"Selected {selected.name}")

        if selected:
            df = load_stats(selected / "match_stats.csv")
            if not df.empty:
                st.dataframe(df, width="stretch")

            video = selected / "output_video.mp4"
            if video.exists():
                st.video(str(video))

            avi = selected / "output_video.avi"
            if avi.exists():
                with open(avi, "rb") as f:
                    st.download_button(
                        "Download Annotated Video",
                        data=f,
                        file_name="annotated_match.avi",
                    )
