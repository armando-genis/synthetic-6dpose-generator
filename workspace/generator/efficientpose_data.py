
import os
import json
import shutil
from pathlib import Path
import yaml
import re
import random
from PIL import Image

class BOPDatasetExtractor:
    def __init__(self, src_path: str, dst_path: str, model_id: int):
        self.src_path = src_path
        self.dst_path = dst_path
        self.model_id = model_id
        self.data_dir = None

    def _ensure_dirs(self):
        # Ensure destination directories exist
        os.makedirs(self.dst_path, exist_ok=True)
        model_dir = os.path.join(self.dst_path, "models")
        self.data_dir = os.path.join(self.dst_path, "data", f"{self.model_id:02d}")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "mask"), exist_ok=True)

    def _process_scene(self):
        for scene in sorted(Path(self.src_path).iterdir()):
            if scene.is_dir():
                # Load scene metadata
                with open(scene / 'scene_gt.json') as gt_file:
                    scene_gt = json.load(gt_file)
                with open(scene / 'scene_camera.json') as cam_file:
                    scene_camera = json.load(cam_file)
                with open(scene / 'scene_gt_info.json') as gt_info_file:
                    scene_gt_info = json.load(gt_info_file)

                # Save filtered metadata files
                filtered_gt = {k: v for k, v in scene_gt.items() if any(obj['obj_id'] == model_id for obj in v)}
                filtered_gt_info = {k: v for k, v in scene_gt_info.items() if k in filtered_gt}
                filtered_camera = {k: v for k, v in scene_camera.items() if k in filtered_gt}

                with open(os.path.join(self.data_dir, 'scene_gt.json'), 'w') as f:
                    json.dump(filtered_gt, f, indent=2)
                with open(os.path.join(self.data_dir, 'scene_gt_info.json'), 'w') as f:
                    json.dump(filtered_gt_info, f, indent=2)
                with open(os.path.join(self.data_dir, 'scene_camera.json'), 'w') as f:
                    json.dump(filtered_camera, f, indent=2)

                # Copy rgb, depth, and mask images for the model_id
                for frame_id in filtered_gt.keys():
                    frame_str = f"{int(frame_id):06d}"
                    # Copy RGB image
                    rgb_src = scene / 'rgb' / f"{frame_str}.jpg"
                    rgb_dst = os.path.join(self.data_dir, 'rgb', f"{frame_str}.jpg")
                    if rgb_src.exists():
                        shutil.copyfile(rgb_src, rgb_dst)

                    # Copy depth image
                    depth_src = scene / 'depth' / f"{frame_str}.png"
                    depth_dst = os.path.join(self.data_dir, 'depth', f"{frame_str}.png")
                    if depth_src.exists():
                        shutil.copyfile(depth_src, depth_dst)


                    # Copy masks only for specified model_id
                    model_index_real = model_id - 1
                    for mask in (scene / 'mask').glob(f"{frame_str}_*.png"):
                        mask_obj_id = int(mask.stem.split('_')[1])
                        if mask_obj_id == model_index_real:
                            mask_dst = os.path.join(self.data_dir, 'mask', f"{frame_str}_{mask_obj_id:06d}.png")
                            shutil.copyfile(mask, mask_dst)

        print("=> [Efficientpose 1/8] Dataset successfully structured!")

    # Custom representer for lists to use flow style only for lists of scalars
    def represent_list_gt(self, dumper, data):
        if all(isinstance(item, (int, float, str)) for item in data):
            # List of scalars, use flow style
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            # List contains non-scalars, use block style
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
    def represent_float_gt(self, dumper, data):
        if data == float('inf'):
            value = '.inf'
        elif data == float('-inf'):
            value = '-.inf'
        elif data != data:  # NaN check
            value = '.nan'
        else:
            value = '{0:.8f}'.format(data)
        return dumper.represent_scalar('tag:yaml.org,2002:float', value)

    def _parse_gt(self):
        scene_gt_path = os.path.join(self.data_dir, 'scene_gt.json')
        scene_gt_info_path = os.path.join(self.data_dir, 'scene_gt_info.json')
        output_path  = os.path.join(self.data_dir, 'gt.yml')

        # Load the scene_gt.json and scene_gt_info.json files
        with open(scene_gt_path, 'r') as gt_file:
            scene_gt = json.load(gt_file)
        with open(scene_gt_info_path, 'r') as gt_info_file:
            scene_gt_info = json.load(gt_info_file)

        # Prepare filtered data for YAML format
        filtered_data = {}
        for frame_id, annotations in scene_gt.items():
            frame_list = []
            for i, annotation in enumerate(annotations):
                if annotation['obj_id'] == self.model_id:
                    # Get the corresponding bounding box from scene_gt_info
                    bbox_obj = scene_gt_info[frame_id][i].get("bbox_obj", [-1, -1, -1, -1])
                    # Append data in the required format
                    frame_list.append({
                        'cam_R_m2c': annotation['cam_R_m2c'],
                        'cam_t_m2c': annotation['cam_t_m2c'],
                        'obj_bb': bbox_obj,
                        'obj_id': annotation['obj_id']
                    })
            if frame_list:
                filtered_data[int(frame_id)] = frame_list  # Convert frame ID to integer for YAML readability

        yaml.add_representer(list, self.represent_list_gt)
        yaml.add_representer(float, self.represent_float_gt)

        # Save the data to gt.yaml with the specified structure
        with open(output_path, 'w') as file:
            yaml.dump(filtered_data, file, sort_keys=False)

        print(f"=> [Efficientpose 2/8] Filtered data with bounding boxes saved to {output_path}")

    def represent_list_camera(self, dumper, data):
        if all(isinstance(item, (int, float, str)) for item in data):
            # List of scalars, use flow style
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            # List contains non-scalars, use block style
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
    # Custom representer for floats to control precision
    def represent_float_camera(self, dumper, data):
        if data == float('inf'):
            value = '.inf'
        elif data == float('-inf'):
            value = '-.inf'
        elif data != data:  # NaN check
            value = '.nan'
        else:
            value = '{0:.6f}'.format(data)  # Adjust precision here
        return dumper.represent_scalar('tag:yaml.org,2002:float', value)

    def _parse_camera_info(self):
        scene_camera_path = os.path.join(self.data_dir, 'scene_camera.json')
        output_path = os.path.join(self.data_dir, 'info.yml')

        # Load the scene_camera.json file
        with open(scene_camera_path, 'r') as camera_file:
            scene_camera = json.load(camera_file)

        # Prepare data for YAML format
        info_data = {}
        for frame_id, camera_info in scene_camera.items():
            # Extract cam_K and depth_scale
            cam_K = camera_info.get('cam_K', [])
            depth_scale = camera_info.get('depth_scale', 1.0)  # Default to 1.0 if not present

            # Round cam_K values to match desired precision (e.g., 6 decimal places)
            cam_K = [round(k, 6) for k in cam_K]

            # Build the frame data
            info_data[int(frame_id)] = {
                'cam_K': cam_K,
                'depth_scale': depth_scale
            }

        yaml.add_representer(list, self.represent_list_camera)
        yaml.add_representer(float, self.represent_float_camera)

        # Save the data to info.yml with the specified structure
        with open(output_path, 'w') as file:
            yaml.dump(info_data, file, sort_keys=True)

        print(f"=> [Efficientpose 3/8] Camera info saved to {output_path}")

    def _parse_fourdigit(self):
        directory_depth = os.path.join(self.data_dir, 'depth')
        directory_rgb = os.path.join(self.data_dir, 'rgb')

        
        # Regular expression pattern to match filenames like '000000.png' or '000000.jpg'
        pattern = re.compile(r'^(\d{6})\.(png|jpg)$')

        # Loop over all files in the directory
        for filename in os.listdir(directory_depth):
            match = pattern.match(filename)
            if match:
                number_str, extension = match.groups()
                number_int = int(number_str)
                # Convert to four-digit number with leading zeros
                new_number_str = f"{number_int:04d}"
                new_filename = f"{new_number_str}.{extension}"
                old_file = os.path.join(directory_depth, filename)
                new_file = os.path.join(directory_depth, new_filename)
                os.rename(old_file, new_file)
                # print(f"Renamed '{filename}' to '{new_filename}'")

        print("=> [Efficientpose 4/8] Depth images parsed to four digits")

        # Loop over all files in the directory
        for filename in os.listdir(directory_rgb):
            match = pattern.match(filename)
            if match:
                number_str, extension = match.groups()
                number_int = int(number_str)
                # Convert to four-digit number with leading zeros
                new_number_str = f"{number_int:04d}"
                new_filename = f"{new_number_str}.{extension}"
                old_file = os.path.join(directory_rgb, filename)
                new_file = os.path.join(directory_rgb, new_filename)
                os.rename(old_file, new_file)
                # print(f"Renamed '{filename}' to '{new_filename}'")

        print("=> [Efficientpose 5/8] Rbg images parsed to four digits")

    def _parse_maskdigits(self):
        directory = os.path.join(self.data_dir, 'mask')

        # Regular expression pattern to match filenames like '000000_000001.png'
        pattern = re.compile(r'^(\d+)_\d+\.(png|jpg)$')

        # Loop over all files in the directory
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                old_number_str, extension = match.groups()
                old_number_int = int(old_number_str)
                # Convert to four-digit number with leading zeros
                new_number_str = f"{old_number_int:04d}"
                new_filename = f"{new_number_str}.{extension}"
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)
                os.rename(old_file, new_file)
                # print(f"Renamed '{filename}' to '{new_filename}'")

        print("=> [Efficientpose 6/8] Mask images parsed to four digits")

    def _create_train_eval_file(self):
        directory = os.path.join(self.data_dir, 'rgb')
        train_file = os.path.join(self.data_dir, 'train.txt')
        test_file  = os.path.join(self.data_dir, 'test.txt')

        # Percentage split between training and testing
        train_percentage = 0.8
        # Gather all image filenames in the directory
        image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]

        # Extract image indices (filenames without extensions)
        image_indices = list(set(os.path.splitext(f)[0] for f in image_files))

        # Shuffle the indices
        random.shuffle(image_indices)

        # Split the indices
        split_point = int(len(image_indices) * train_percentage)
        train_indices = image_indices[:split_point]
        test_indices = image_indices[split_point:]

        # Write train.txt
        with open(train_file, 'w') as f:
            for idx in sorted(train_indices):
                f.write(f"{idx}\n")

        # Write test.txt
        with open(test_file, 'w') as f:
            for idx in sorted(test_indices):
                f.write(f"{idx}\n")

        print(f"=> [Efficientpose 7/8] Randomly split data into `{train_file}` and `{test_file}` with {len(train_indices)} and {len(test_indices)} entries respectively.")

    def _rbgtopng(self):
        directory = os.path.join(self.data_dir, 'rgb')

        # Loop over all files in the directory
        for filename in os.listdir(directory):
            if filename.lower().endswith('.jpg'):
                # Construct full file path
                jpg_file = os.path.join(directory, filename)
                # Replace the .jpg extension with .png
                png_filename = os.path.splitext(filename)[0] + '.png'
                png_file = os.path.join(directory, png_filename)
                
                # Open the jpg image and save it as png
                with Image.open(jpg_file) as img:
                    img.save(png_file, 'PNG')
                
                # Remove the original jpg file
                os.remove(jpg_file)
                
                # print(f"Converted '{filename}' to '{png_filename}' and deleted the original file")

        print("=> [Efficientpose 8/8] Rgb images converted to png")
        

    def run(self):
        self._ensure_dirs()
        self._process_scene()
        self._parse_gt()
        self._parse_camera_info()
        self._parse_fourdigit()
        self._parse_maskdigits()
        self._create_train_eval_file()
        self._rbgtopng()


if __name__ == "__main__":

    src_path = '/workspace/generator/datasets_big_3/bop_data/lm/train_pbr'
    dst_path = '/workspace/generator/datasets_big_3/efficientpose_data'
    model_id = 1 
    extractor = BOPDatasetExtractor(src_path, dst_path, model_id)
    extractor.run()