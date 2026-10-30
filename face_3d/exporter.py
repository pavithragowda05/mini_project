import os
import trimesh
import numpy as np
from PIL import Image

class Face3DExporter:
    """
    Exports reconstructed 3D Face Meshes with UV Texture maps into standard 3D formats:
    - Wavefront OBJ (.obj + .mtl + texture.png)
    - Polygon File Format PLY (.ply)
    - GL Transmission Format GLTF/GLB (.gltf / .glb)
    """
    def __init__(self, output_dir="output/3d"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _texture_atlas(texture_pil):
        """Place the face image on the front UV region and soften the inferred back."""
        front = texture_pil.convert("RGB")
        width, height = front.size
        atlas = Image.new("RGB", (width * 2, height))
        atlas.paste(front, (width // 2, 0))

        edge_width = max(1, width // 32)
        left_edge = front.crop((0, 0, edge_width, height)).resize(
            (width // 2, height), Image.Resampling.BICUBIC
        )
        right_edge = front.crop((width - edge_width, 0, width, height)).resize(
            (width // 2, height), Image.Resampling.BICUBIC
        )
        atlas.paste(left_edge, (0, 0))
        atlas.paste(right_edge, (width + width // 2, 0))
        return atlas

    def export_all(self, vertices, faces, uvs, texture_img, base_filename="reconstructed_face_3d"):
        """
        vertices: (N, 3) float array
        faces: (F, 3) int array
        uvs: (N, 2) float array
        texture_img: PIL Image or numpy array (H, W, 3)
        base_filename: base name without extension
        Returns dict containing absolute file paths of exported 3D formats.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        uvs = np.asarray(uvs, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("3D export requires at least four XYZ vertices.")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("3D export requires triangular mesh faces.")
        if uvs.shape != (len(vertices), 2):
            raise ValueError("UV coordinates must contain one pair per vertex.")
        if not np.isfinite(vertices).all() or not np.isfinite(uvs).all():
            raise ValueError("Mesh vertices and UV coordinates must be finite.")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Mesh faces reference an invalid vertex index.")

        if isinstance(texture_img, np.ndarray):
            if texture_img.dtype != np.uint8:
                if texture_img.min() < 0:
                    texture_img = ((texture_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                else:
                    texture_img = (texture_img * 255.0).clip(0, 255).astype(np.uint8)
            texture_pil = Image.fromarray(texture_img)
        else:
            texture_pil = texture_img
        if texture_pil is None or texture_pil.width == 0 or texture_pil.height == 0:
            raise ValueError("A non-empty facial texture is required for 3D export.")

        atlas = self._texture_atlas(texture_pil)

        # Save the UV atlas used by textured mesh formats.
        tex_path = os.path.join(self.output_dir, f"{base_filename}_texture.png")
        atlas.save(tex_path)

        # Create Trimesh geometry object with visual material texture
        material = trimesh.visual.texture.SimpleMaterial(image=atlas)
        visuals = trimesh.visual.TextureVisuals(uv=uvs, image=atlas, material=material)
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, visual=visuals, validate=True
        )
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or mesh.area <= 0.0:
            raise ValueError("Generated mesh has no exportable surface area.")
        mesh.fix_normals(multibody=True)

        exported_paths = {}

        # 1. Export OBJ
        obj_path = os.path.join(self.output_dir, f"{base_filename}.obj")
        mesh.export(obj_path, file_type="obj")
        exported_paths["obj"] = os.path.abspath(obj_path)

        # 2. Export PLY
        ply_path = os.path.join(self.output_dir, f"{base_filename}.ply")
        ply_mesh = mesh.copy()
        ply_mesh.visual = mesh.visual.to_color()
        ply_mesh.export(ply_path, file_type="ply")
        exported_paths["ply"] = os.path.abspath(ply_path)

        # 3. Export GLTF / GLB
        glb_path = os.path.join(self.output_dir, f"{base_filename}.glb")
        mesh.export(glb_path, file_type="glb")
        exported_paths["glb"] = os.path.abspath(glb_path)

        exported_paths["texture"] = os.path.abspath(tex_path)
        print(f"[OK] 3D Face successfully exported to '{self.output_dir}' (OBJ, PLY, GLB).")
        return exported_paths
