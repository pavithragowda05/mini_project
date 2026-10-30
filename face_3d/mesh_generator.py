import os

import cv2
import numpy as np
import trimesh


class Face3DMeshGenerator:
    """Build an inferred closed head shell with sketch-conditioned front relief."""

    def __init__(self, img_size=(256, 256), grid_res=64):
        self.img_size = img_size
        self.grid_res = max(12, int(grid_res))
        self._face_cascade = None
        self._eye_cascade = None
        self._cascades_loaded = False

    @staticmethod
    def _to_uint8(image):
        image = np.asarray(image)
        if image.dtype == np.uint8:
            return image
        if image.size and image.min() < 0:
            return ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        return image.clip(0, 255).astype(np.uint8)

    def _load_cascades(self):
        if self._cascades_loaded:
            return
        self._cascades_loaded = True
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
            return
        cascade_dir = cv2.data.haarcascades
        self._face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        )
        self._eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_eye_tree_eyeglasses.xml")
        )

    @staticmethod
    def _estimate_dark_feature(gray, bounds, expected, radius):
        height, width = gray.shape
        x0, y0, x1, y1 = bounds
        left = max(0, int(x0 * width))
        top = max(0, int(y0 * height))
        right = min(width, int(x1 * width))
        bottom = min(height, int(y1 * height))
        region = gray[top:bottom, left:right]
        if region.size < 16:
            return expected, 0.0

        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5).astype(np.float32)
        response = np.maximum(smooth - gray.astype(np.float32), 0.0)
        region_response = response[top:bottom, left:right]
        rows, columns = np.mgrid[top:bottom, left:right]
        radius_x = max(radius[0] * width, 1.0)
        radius_y = max(radius[1] * height, 1.0)
        prior = np.exp(
            -0.5 * (
                ((columns - expected[0]) / radius_x) ** 2
                + ((rows - expected[1]) / radius_y) ** 2
            )
        )
        threshold = np.percentile(region_response, 55)
        weights = np.maximum(region_response - threshold, 0.0) * prior
        total_weight = float(weights.sum())
        contrast = float(np.percentile(region_response, 95))
        support = float(np.mean(region_response > 3.0))
        confidence = float(
            np.clip((contrast - 3.0) / 12.0, 0.0, 1.0)
            * np.clip((support - 0.01) / 0.08, 0.0, 1.0)
        )
        if total_weight <= 1e-6 or confidence < 0.12:
            return expected, 0.0

        point = (
            float((weights * columns).sum() / total_weight),
            float((weights * rows).sum() / total_weight),
        )
        return point, confidence

    def extract_landmarks(self, face_img_np):
        """Detect a face and eyes when possible; other feature locations are proxies."""
        image = self._to_uint8(face_img_np)
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Expected an RGB or grayscale face image.")

        height, width = gray.shape
        self._load_cascades()
        detected_faces = []
        if self._face_cascade is not None and not self._face_cascade.empty():
            detected_faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(max(32, width // 8), max(32, height // 8))
            )

        if len(detected_faces):
            x, y, box_width, box_height = max(
                detected_faces, key=lambda box: box[2] * box[3]
            )
            detection_source = "opencv_haar_face"
        else:
            box_width = width * 0.72
            box_height = height * 0.88
            x = (width - box_width) / 2.0
            y = (height - box_height) / 2.0
            detection_source = "centered_face_fallback"

        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        fallback_eyes = [
            (center_x - box_width * 0.23, center_y - box_height * 0.16),
            (center_x + box_width * 0.23, center_y - box_height * 0.16),
        ]
        eyes = []
        if self._eye_cascade is not None and not self._eye_cascade.empty():
            face_region = gray[
                max(0, int(y)):min(height, int(y + box_height * 0.65)),
                max(0, int(x)):min(width, int(x + box_width))
            ]
            if face_region.size:
                found_eyes = self._eye_cascade.detectMultiScale(
                    face_region, scaleFactor=1.1, minNeighbors=4,
                    minSize=(max(8, int(box_width * 0.07)), max(8, int(box_height * 0.04)))
                )
                candidates = [
                    (x + eye_x + eye_width / 2.0, y + eye_y + eye_height / 2.0, eye_width * eye_height)
                    for eye_x, eye_y, eye_width, eye_height in found_eyes
                ]
                for left_eye in sorted(candidates, key=lambda eye: eye[0]):
                    for right_eye in sorted(candidates, key=lambda eye: eye[0], reverse=True):
                        separation = right_eye[0] - left_eye[0]
                        if 0.18 * box_width <= separation <= 0.62 * box_width:
                            if abs(left_eye[1] - right_eye[1]) <= 0.16 * box_height:
                                eyes = [left_eye[:2], right_eye[:2]]
                                break
                    if eyes:
                        break

        if not eyes:
            local_left, left_confidence = self._estimate_dark_feature(
                gray,
                (x / width + 0.08 * box_width / width, y / height + 0.24 * box_height / height,
                 x / width + 0.48 * box_width / width, y / height + 0.50 * box_height / height),
                fallback_eyes[0],
                (0.075, 0.08),
            )
            local_right, right_confidence = self._estimate_dark_feature(
                gray,
                (x / width + 0.52 * box_width / width, y / height + 0.24 * box_height / height,
                 x / width + 0.92 * box_width / width, y / height + 0.50 * box_height / height),
                fallback_eyes[1],
                (0.075, 0.08),
            )
            if left_confidence >= 0.12 and right_confidence >= 0.12:
                eyes = [local_left, local_right]
                detection_source += "_contrast_eyes"
            else:
                eyes = fallback_eyes
                left_confidence = right_confidence = 0.0
        else:
            left_confidence = right_confidence = 1.0
            detection_source += "_and_eyes"

        fallback_mouth = (center_x, center_y + box_height * 0.25)
        mouth, mouth_confidence = self._estimate_dark_feature(
            gray,
            (x / width + 0.25 * box_width / width, y / height + 0.54 * box_height / height,
             x / width + 0.75 * box_width / width, y / height + 0.84 * box_height / height),
            fallback_mouth,
            (0.13, 0.07),
        )
        if mouth_confidence < 0.12:
            mouth = fallback_mouth
        else:
            detection_source += "_contrast_mouth"

        return {
            "center": (center_x, center_y),
            "face_width": float(box_width),
            "face_height": float(box_height),
            "face_bounds": (
                max(0, int(round(x))), max(0, int(round(y))),
                min(width, int(round(x + box_width))),
                min(height, int(round(y + box_height))),
            ),
            "nose_bridge": (center_x, center_y - box_height * 0.02),
            "nose_tip": (center_x, center_y + box_height * 0.10),
            "left_eye": eyes[0],
            "right_eye": eyes[1],
            "eye_confidence": min(left_confidence, right_confidence),
            "mouth_center": mouth,
            "mouth_confidence": mouth_confidence,
            "chin": (center_x, center_y + box_height * 0.43),
            "landmark_source": detection_source,
        }

    @staticmethod
    def _face_coordinates(point, center, width, height):
        return (
            float(np.clip((point[0] - center[0]) / (width * 0.5), -1.0, 1.0)),
            float(np.clip((center[1] - point[1]) / (height * 0.5), -1.0, 1.0)),
        )

    def generate_mesh(self, face_img_np):
        """Return a closed ellipsoidal head mesh, UVs, and detected/proxy landmarks."""
        image = self._to_uint8(face_img_np)
        landmarks = self.extract_landmarks(image)
        center = landmarks["center"]
        box_width = landmarks["face_width"]
        box_height = landmarks["face_height"]
        face_ratio = float(np.clip(box_width / max(box_height, 1.0), 0.62, 0.92))

        eye_left = self._face_coordinates(landmarks["left_eye"], center, box_width, box_height)
        eye_right = self._face_coordinates(landmarks["right_eye"], center, box_width, box_height)
        nose_bridge = self._face_coordinates(landmarks["nose_bridge"], center, box_width, box_height)
        nose_tip = self._face_coordinates(landmarks["nose_tip"], center, box_width, box_height)
        mouth = self._face_coordinates(landmarks["mouth_center"], center, box_width, box_height)
        chin = self._face_coordinates(landmarks["chin"], center, box_width, box_height)

        latitude_count = self.grid_res
        longitude_count = self.grid_res * 2
        latitudes = np.linspace(
            -np.pi / 2.0 + np.pi / (latitude_count + 1),
            np.pi / 2.0 - np.pi / (latitude_count + 1),
            latitude_count,
        )
        longitudes = np.linspace(-np.pi, np.pi, longitude_count, endpoint=False)
        latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")

        x_norm = np.cos(latitude_grid) * np.sin(longitude_grid)
        y_norm = np.sin(latitude_grid)
        front = np.clip(np.cos(longitude_grid), 0.0, 1.0) ** 1.5
        jaw_taper = 1.0 - 0.13 * np.clip((-y_norm - 0.05) / 0.95, 0.0, 1.0) ** 1.2
        x = face_ratio * x_norm * jaw_taper
        y = y_norm
        z = 0.62 * np.cos(latitude_grid) * np.cos(longitude_grid)

        def relief(point_x, point_y, width, height, amount):
            return amount * np.exp(
                -0.5 * (((x_norm - point_x) / width) ** 2 + ((y_norm - point_y) / height) ** 2)
            )

        relief_map = relief(0.0, 0.68, 0.48, 0.30, 0.035)
        relief_map += relief(0.0, nose_bridge[1], 0.15, 0.32, 0.16)
        relief_map += relief(0.0, nose_bridge[1] + 0.12, 0.11, 0.20, 0.045)
        relief_map += relief(nose_tip[0], nose_tip[1], 0.16, 0.12, 0.15)
        relief_map += relief(-0.12, nose_tip[1] - 0.025, 0.09, 0.07, 0.045)
        relief_map += relief(0.12, nose_tip[1] - 0.025, 0.09, 0.07, 0.045)
        relief_map += relief(eye_left[0], eye_left[1], 0.17, 0.095, -0.075)
        relief_map += relief(eye_right[0], eye_right[1], 0.17, 0.095, -0.075)
        relief_map += relief(eye_left[0], eye_left[1] + 0.10, 0.21, 0.065, 0.055)
        relief_map += relief(eye_right[0], eye_right[1] + 0.10, 0.21, 0.065, 0.055)
        relief_map += relief(-0.49, -0.04, 0.23, 0.20, 0.11)
        relief_map += relief(0.49, -0.04, 0.23, 0.20, 0.11)
        relief_map += relief(mouth[0], mouth[1] + 0.04, 0.22, 0.045, 0.045)
        relief_map += relief(mouth[0], mouth[1] - 0.035, 0.23, 0.05, 0.04)
        relief_map += relief(mouth[0], mouth[1] + 0.105, 0.07, 0.09, 0.025)
        relief_map -= relief(mouth[0], mouth[1], 0.20, 0.022, 0.025)
        relief_map -= relief(-0.19, mouth[1], 0.06, 0.045, 0.012)
        relief_map -= relief(0.19, mouth[1], 0.06, 0.045, 0.012)
        relief_map += relief(chin[0], chin[1], 0.31, 0.17, 0.075)
        relief_map += relief(-0.90, 0.0, 0.12, 0.22, 0.045)
        relief_map += relief(0.90, 0.0, 0.12, 0.22, 0.045)

        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        smooth_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0).astype(np.float32)
        detail = (gray.astype(np.float32) - smooth_gray) / 255.0
        detail = cv2.GaussianBlur(detail, (0, 0), sigmaX=4.0)
        pixel_x = np.clip(
            np.rint(center[0] + x_norm * box_width * 0.5).astype(np.int32),
            0, gray.shape[1] - 1,
        )
        pixel_y = np.clip(
            np.rint(center[1] - y_norm * box_height * 0.5).astype(np.int32),
            0, gray.shape[0] - 1,
        )
        image_shading = smooth_gray[pixel_y, pixel_x] / 255.0 - 0.5
        image_relief = np.clip(image_shading, -0.5, 0.5) * 0.10
        sketch_relief = np.clip(detail[pixel_y, pixel_x], -0.08, 0.08) * 0.2
        relief_map += front * (image_relief + sketch_relief)
        z += front * relief_map

        ring_vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        south_index = len(ring_vertices)
        north_index = south_index + 1
        vertices = np.vstack((ring_vertices, [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]))

        atlas_u = np.where(
            np.cos(longitude_grid) >= 0.0,
            0.5 + 0.25 * x_norm,
            np.where(x_norm < 0.0, -0.25 * x_norm, 1.0 - 0.25 * x_norm),
        )
        atlas_v = (1.0 + y_norm) * 0.5
        ring_uvs = np.stack((atlas_u, atlas_v), axis=-1).reshape(-1, 2)
        uvs = np.vstack((ring_uvs, [0.5, 0.0], [0.5, 1.0]))

        row_width = longitude_count
        faces = []
        for row in range(latitude_count - 1):
            for column in range(longitude_count):
                lower_left = row * row_width + column
                lower_right = row * row_width + (column + 1) % longitude_count
                upper_left = lower_left + row_width
                upper_right = row * row_width + row_width + (column + 1) % longitude_count
                faces.append([lower_left, lower_right, upper_left])
                faces.append([lower_right, upper_right, upper_left])

        last_row = (latitude_count - 1) * row_width
        for column in range(longitude_count):
            next_column = (column + 1) % longitude_count
            faces.append([south_index, next_column, column])
            faces.append([north_index, last_row + next_column, last_row + column])

        mesh = trimesh.Trimesh(
            vertices=vertices, faces=np.asarray(faces, dtype=np.int32), process=False
        )
        mesh.fix_normals(multibody=True)
        landmarks["face_ratio"] = face_ratio
        return vertices, mesh.faces.astype(np.int32), uvs, landmarks
