
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def _get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)

        if len(image_2d) < 2:
            raise ValueError("Not enough pixels for clustering.")

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42
        )
        kmeans.fit(image_2d)
        return kmeans

    def _get_player_color(self, frame, bbox):
        height, width = frame.shape[:2]

        x1 = max(0, min(width, int(bbox[0])))
        y1 = max(0, min(height, int(bbox[1])))
        x2 = max(0, min(width, int(bbox[2])))
        y2 = max(0, min(height, int(bbox[3])))

        if x2 <= x1 or y2 - y1 < 2:
            return None

        crop = frame[y1:y2, x1:x2]
        top_half = crop[:crop.shape[0] // 2, :]

        if top_half.size == 0 or len(top_half.reshape(-1, 3)) < 2:
            return None

        kmeans = self._get_clustering_model(top_half)

        labels = kmeans.labels_.reshape(
            top_half.shape[0],
            top_half.shape[1]
        )

        corners = [
            labels[0, 0],
            labels[0, -1],
            labels[-1, 0],
            labels[-1, -1]
        ]

        bg_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - bg_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for info in player_detections.values():
            color = self._get_player_color(
                frame,
                info["bbox"]
            )

            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 2:
            raise RuntimeError(
                "Not enough valid player crops to initialize teams."
            )

        self.kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        self.kmeans.fit(player_colors)

        self.team_colors[1] = tuple(
            int(c) for c in self.kmeans.cluster_centers_[0]
        )

        self.team_colors[2] = tuple(
            int(c) for c in self.kmeans.cluster_centers_[1]
        )

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            raise RuntimeError(
                "Team colors have not been initialized."
            )

        color = self._get_player_color(
            frame,
            player_bbox
        )

        if color is None:
            return 0

        team_id = int(
            self.kmeans.predict(
                color.reshape(1, -1)
            )[0]
        ) + 1

        self.player_team_dict[player_id] = team_id

        return team_id
