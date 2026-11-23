
import csv
import gc
import os
from collections import defaultdict

import cv2
import numpy as np

from football_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from football_analysis.heatmap_generator.heatmap_generator import HeatmapGenerator
from football_analysis.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from football_analysis.speed_distance_estimator.speed_distance_estimator import SpeedAndDistanceEstimator
from football_analysis.team_assigner.team_assigner import TeamAssigner
from football_analysis.trackers.tracker import Tracker
from football_analysis.utils.video_utils import read_video
from football_analysis.view_transformer.view_transformer import ViewTransformer


def run_pipeline(
    input_path,
    output_video_path,
    heatmap_dir,
    stats_path,
    model_path="models/best_trained.pt",
    track_stub=None,
    cam_stub=None,
):
    # Validate input video
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Video not found: {input_path}")

    cap_check = cv2.VideoCapture(input_path)
    if not cap_check.isOpened():
        cap_check.release()
        raise RuntimeError(f"Cannot open video: {input_path}")

    ret, frame0 = cap_check.read()
    cap_check.release()

    if not ret:
        raise RuntimeError("Video contains no readable frames.")

    use_track_stub = bool(track_stub and os.path.exists(track_stub))
    use_cam_stub = bool(cam_stub and os.path.exists(cam_stub))

    # Load frames for detection and camera movement
    if not use_track_stub or not use_cam_stub:
        print("Reading video frames...")
        frames = read_video(input_path)

        if not frames:
            raise RuntimeError("No frames found in the input video.")

        frame0 = frames[0]
    else:
        frames = None

    # Object tracking
    print("Tracking objects...")

    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(
        frames if frames is not None else [],
        read_from_stub=use_track_stub,
        stub_path=track_stub,
    )

    if not tracks["players"]:
        raise RuntimeError("No tracking frames were generated.")

    # Camera movement
    print("Estimating camera movement...")

    cam_est = CameraMovementEstimator(frame0)

    camera_movement = cam_est.get_camera_movement(
        frames if frames is not None else [],
        read_from_stub=use_cam_stub,
        stub_path=cam_stub,
    )

    # Release frame memory
    if frames is not None:
        del frames
        gc.collect()

    # Interpolate ball detections before calculating positions
    print("Interpolating ball positions...")

    tracks["ball"] = tracker.interpolate_ball_positions(
        tracks["ball"]
    )

    # Calculate object positions
    print("Calculating player and ball positions...")

    tracker.add_position_to_tracks(tracks)

    # Adjust positions for camera movement
    print("Adjusting camera movement...")

    cam_est.add_adjust_positions_to_tracks(
        tracks,
        camera_movement,
    )

    # Perspective transformation
    print("Applying perspective transformation...")

    ViewTransformer().add_transformed_position_to_tracks(tracks)

    # Speed and distance
    print("Computing speed and distance...")

    SpeedAndDistanceEstimator().add_speed_and_distance_to_tracks(
        tracks
    )

    # Team assignment
    print("Assigning teams...")

    team_assigner = TeamAssigner()

    # Find first frame containing tracked players
    first_player_frame = next(
        (
            i for i, players in enumerate(tracks["players"])
            if len(players) >= 2
        ),
        None,
    )

    if first_player_frame is None:
        raise RuntimeError(
            "No players were tracked in the video. "
            "Try another football video."
        )

    cap = cv2.VideoCapture(input_path)

    try:
        # Read the frame used for team color initialization
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_player_frame)
        ret, team_frame = cap.read()

        if not ret:
            raise RuntimeError(
                "Could not read frame for team assignment."
            )

        team_assigner.assign_team_color(
            team_frame,
            tracks["players"][first_player_frame],
        )

        # Start reading from first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_num, player_track in enumerate(tracks["players"]):
            ret, frame = cap.read()

            if not ret:
                break

            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    frame,
                    track["bbox"],
                    player_id,
                )

                track["team"] = team
                track["team_color"] = team_assigner.team_colors.get(
                    team, (128, 128, 128)
                )

    finally:
        cap.release()

    # Ball possession
    print("Assigning ball possession...")

    pba = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_info = tracks["ball"][frame_num].get(1)

        if ball_info is None:
            team_ball_control.append(
                team_ball_control[-1]
                if team_ball_control else 0
            )
            continue

        ball_bbox = ball_info["bbox"]

        assigned = pba.assign_ball_to_player(
            player_track,
            ball_bbox,
        )

        if assigned != -1 and assigned in player_track:
            player_track[assigned]["has_ball"] = True

            team_ball_control.append(
                player_track[assigned].get("team", 0)
            )
        else:
            team_ball_control.append(
                team_ball_control[-1]
                if team_ball_control else 0
            )

    team_ball_control = np.array(team_ball_control)

    # Annotated output video
    print("Drawing annotations and saving video...")

    output_dir = os.path.dirname(output_video_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    _stream_annotated_video(
        output_video_path,
        input_path,
        tracks,
        team_ball_control,
        camera_movement,
        tracker,
        cam_est,
    )

    # Heatmaps
    print("Generating heatmaps...")

    os.makedirs(heatmap_dir, exist_ok=True)

    HeatmapGenerator().generate(
        tracks,
        heatmap_dir,
    )

    # Player statistics
    print("Exporting stats CSV...")

    _export_stats(
        tracks,
        team_ball_control,
        stats_path,
    )

    print("TactiVision AI analysis completed successfully!")

    return tracks, team_ball_control


def _stream_annotated_video(
    output_path,
    input_path,
    tracks,
    team_ball_control,
    camera_movement,
    tracker,
    cam_est,
):
    """Process and save annotated video frame by frame."""

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video for annotation.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 24

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create output video: {output_path}"
        )

    speed_est = SpeedAndDistanceEstimator()
    frame_num = 0

    try:
        while frame_num < len(tracks["players"]):
            ret, frame = cap.read()

            if not ret:
                break

            single = [frame]

            single_tracks = {
                key: [values[frame_num]]
                for key, values in tracks.items()
            }

            annotated = tracker.draw_annotations(
                single,
                single_tracks,
                team_ball_control,
                frame_offset=frame_num,
            )

            annotated = cam_est.draw_camera_movement(
                annotated,
                [camera_movement[frame_num]],
            )

            speed_est.draw_speed_and_distance(
                annotated,
                single_tracks,
            )

            out.write(annotated[0])

            frame_num += 1

    finally:
        cap.release()
        out.release()


def _export_stats(
    tracks,
    team_ball_control,
    output_path,
):
    """Export player speed, distance and team possession."""

    player_data = defaultdict(
        lambda: {
            "speeds": [],
            "distance": 0.0,
            "team": 0,
        }
    )

    for frame_tracks in tracks["players"]:
        for player_id, info in frame_tracks.items():

            if "speed" in info:
                player_data[player_id]["speeds"].append(
                    info["speed"]
                )

            player_data[player_id]["distance"] = max(
                player_data[player_id]["distance"],
                info.get("distance", 0.0),
            )

            player_data[player_id]["team"] = info.get(
                "team",
                0,
            )

    possession = {
        1: int(np.sum(team_ball_control == 1)),
        2: int(np.sum(team_ball_control == 2)),
    }

    total = sum(possession.values()) or 1

    rows = []

    for player_id, data in player_data.items():

        speeds = data["speeds"]
        team = data["team"]

        rows.append({
            "player_id": player_id,
            "team": team,
            "avg_speed_kmh": (
                round(float(np.mean(speeds)), 2)
                if speeds else 0.0
            ),
            "max_speed_kmh": (
                round(float(max(speeds)), 2)
                if speeds else 0.0
            ),
            "total_distance_m": round(
                data["distance"],
                2,
            ),
            "possession_pct": round(
                possession.get(team, 0) / total * 100,
                2,
            ),
        })

    fieldnames = [
        "player_id",
        "team",
        "avg_speed_kmh",
        "max_speed_kmh",
        "total_distance_m",
        "possession_pct",
    ]

    stats_dir = os.path.dirname(output_path)

    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_pipeline(
        input_path="input_videos/test.mp4",
        output_video_path="output_videos/output_video.avi",
        heatmap_dir="output_videos/heatmaps",
        stats_path="output_videos/match_stats.csv",
        model_path="models/best_trained.pt",
        track_stub=None,
        cam_stub=None,
    )
