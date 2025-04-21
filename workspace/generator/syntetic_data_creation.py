#!/usr/bin/env python3

import blenderproc as bp  # must be first!
from blenderproc.python.utility.Utility import Utility
import bpy

import argparse
import cv2
import glob
import json
from math import acos, atan, cos, pi, sin, sqrt
import numpy as np
import os
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import random
import sys
from tqdm import tqdm
from time import sleep
import shutil

# Define color codes for console output
RED = "\033[1;31m"
GREEN = "\033[1;32m"
BLUE = "\033[1;34m"
YELLOW = "\033[1;33m"
PURPLE = "\033[1;35m"
RESET = "\033[0m"

# ##############################################################################
# YOLOv5-6D-Pose helper functions
# ##############################################################################

def write_folders_yolo6dpose(folder):
    # create a folder called 'JPEGImages'
    image_folder = os.path.join(folder, 'JPEGImages')
    os.makedirs(image_folder, exist_ok=True)
    # create a folder called 'labels'
    labels_folder = os.path.join(folder, 'labels')
    os.makedirs(labels_folder, exist_ok=True)
    # create a folder called 'mask'
    mask_folder = os.path.join(folder, 'mask')
    os.makedirs(mask_folder, exist_ok=True)


def write_data_yolo6dpose(yolo6dope_folder, out_directory, im, frame, width, height, camera, objects, objects_data, target_obj_idx=None):

    image_folder = os.path.join(yolo6dope_folder, 'JPEGImages')
    labels_folder = os.path.join(yolo6dope_folder, 'labels')
    mask_folder = os.path.join(yolo6dope_folder, 'mask')
    
    # save the image in the folder
    filename = os.path.join(image_folder, str(frame).zfill(6) + ".png")
    im.save(filename)

    # get the mask from the dop data folder
    frame_str = str(frame).zfill(6)
    source_mask_path = os.path.join(out_directory, 'bop_data', 'lm', 'train_pbr', '000000', 
                                   'mask', f"{frame_str}_{target_obj_idx:06d}.png")
    
    target_mask_path = os.path.join(mask_folder, f"{frame_str}_{target_obj_idx:06d}.png")
    shutil.copy(source_mask_path, target_mask_path)

    # write labels
    filename = os.path.join(labels_folder, str(frame).zfill(6) + ".txt")
    write_custom_pose_data(filename, width, height, camera, objects, objects_data, target_obj_idx)

    
def write_custom_pose_data(outf, width, height, camera, objects, objects_data, target_obj_idx=None):
    """
    Writes custom pose estimation data with 29 values per object:
    - Class label (1)
    - 9 points (centroid + 8 corners) × 2 coordinates (18)
    - Normalized object size range (2)
    - Camera intrinsics and image dimensions (8)
    """
    # Get camera intrinsics
    K = camera.get_intrinsics_as_K_matrix()
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    # Make sure target_obj_idx is valid
    if target_obj_idx < 0 or target_obj_idx >= len(objects):
        print(f"Error: Target object index {target_obj_idx} is out of range (0-{len(objects)-1})")
        return
    
    # Get the target object
    obj = objects[target_obj_idx]
    
    with open(outf, "w") as f:

        # Get all cuboid keypoints (already in image space!)
        cuboid_points = get_cuboid_image_space(obj, camera)
        
        # Extract all x and y coordinates from the projected points to calculate 2D range
        all_x = [point[0] for point in cuboid_points]
        all_y = [point[1] for point in cuboid_points]
        
        # Calculate ranges in image space and normalize
        x_range = (max(all_x) - min(all_x)) / width
        y_range = (max(all_y) - min(all_y)) / height
        
        # Create the output array with 29 values
        line = []
        
        # Class Label (1 value)
        line.append(f"{objects_data[target_obj_idx]['id']:.6f}")
        
        # All 9 points (centroid + 8 corners) - normalized (18 values)
        # Centroid coordinates
        line.extend([f"{cuboid_points[8][0] / width:.6f}", f"{cuboid_points[8][1] / height:.6f}"])
        
        # 8 corner coordinates
        for i in range(8):
            line.extend([f"{cuboid_points[i][0] / width:.6f}", f"{cuboid_points[i][1] / height:.6f}"])
        
        # Normalized Object Size Range (2 values)
        line.extend([f"{x_range:.6f}", f"{y_range:.6f}"])
        
        # Camera intrinsics and image dimensions (8 values)
        line.extend([
            f"{fx:.6f}",             # Focal length x
            f"{fy:.6f}",             # Focal length y
            f"{width:.6f}",          # Sensor width
            f"{height:.6f}",         # Sensor height
            f"{cx:.6f}",             # Focal offset x (u0)
            f"{cy:.6f}",             # Focal offset y (v0)
            f"{width:.6f}",          # Image width
            f"{height:.6f}"          # Image height
        ])
        
        # Writing all 29 values
        f.write(" ".join(line) + "\n")

# ##############################################################################
# Dope helper functions
# ##############################################################################

def get_cuboid_image_space(mesh, camera):
    # object aligned bounding box coordinates in world coordinates
    bbox = mesh.get_bound_box()
    '''
    bbox is a list of the world-space coordinates of the corners of a
    blender object's oriented bounding box
     https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box

                   TOP
           3 +-----------------+ 7
            /                 /|
           /                 / |
        2 +-----------------+ 6
          |     z    y      |  |
          |      | /        |  |
          |      |/         |  |
          |  |   +--- x     |  |
           0 +-             |  + 4
          | /               | /
          |                 |/
        1 +-----------------+ 5
                FRONT

    '''

    centroid = np.array([0.,0.,0.])
    for ii in range(8):
        centroid += bbox[ii]
    centroid = centroid / 8

    cam_pose = np.linalg.inv(camera.get_camera_pose()) # 4x4 world to camera transformation matrx.
    # rvec & tvec describe the world to camera coordinate system
    tvec = -cam_pose[0:3,3]
    rvec = -cv2.Rodrigues(cam_pose[0:3,0:3])[0]
    K = camera.get_intrinsics_as_K_matrix()

    # However these points are in a different order than the original DOPE data format,
    # so we must reorder them
    dope_order = [6, 2, 1, 5, 7, 3, 0, 4]
    cuboid = [None for ii in range(9)]
    for ii in range(8):
        cuboid[dope_order[ii]] = cv2.projectPoints(bbox[ii], rvec, tvec, K, np.array([]))[0][0][0]
    cuboid[8] = cv2.projectPoints(centroid, rvec, tvec, K, np.array([]))[0][0][0]

    return np.array(cuboid, dtype=float).tolist()

def draw_cuboid_markers(objects, camera, im):
    colors = ['yellow', 'magenta', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'white']
    R = 2 # radius
    # draw dots on image to label the cuiboid vertices
    draw = ImageDraw.Draw(im)
    for oo in objects:
        projected_keypoints = get_cuboid_image_space(oo, camera)
        for idx, pp in enumerate(projected_keypoints):
            x = int(pp[0])
            y = int(pp[1])
            draw.ellipse((x-R, y-R, x+R, y+R), fill=colors[idx])

    return im

def write_json(outf, width, height, min_pixels, camera, objects, objects_data, seg_map):
    cam_xform = camera.get_camera_pose()
    eye = -cam_xform[0:3,3]
    at = -cam_xform[0:3,2]
    up = cam_xform[0:3,0]

    K = camera.get_intrinsics_as_K_matrix()

    data = {
        "camera_data" : {
            "width" : width,
            'height' : height,
            'camera_look_at':
            {
                'at': [
                    at[0],
                    at[1],
                    at[2],
                ],
                'eye': [
                    eye[0],
                    eye[1],
                    eye[2],
            ],
                'up': [
                    up[0],
                    up[1],
                    up[2],
                ]
            },
            'intrinsics':{
                'fx':K[0][0],
                'fy':K[1][1],
                'cx':K[2][0],
                'cy':K[2][1]
            }
        },
        "objects" : []
    }

    ## Object data
    ##
    for ii, oo in enumerate(objects):
        idx = ii+1 # objects ID indices start at '1'

        num_pixels = int(np.sum((seg_map == idx)))

        if num_pixels < min_pixels:
            continue
        projected_keypoints = get_cuboid_image_space(oo, camera)

        data['objects'].append({
            'class': objects_data[ii]['class'],
            'name': objects_data[ii]['name'],
            'visibility': num_pixels,
            'projected_cuboid': projected_keypoints,
            ## 'location' and 'quaternion_xyzw' are both optional data fields,
            ## not used for training
            'location': objects_data[ii]['location'],
            'quaternion_xyzw': objects_data[ii]['quaternion_xyzw']
        })

    with open(outf, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return data

# ##############################################################################
# 3D rotation matrices and positioning functions
# ##############################################################################

def Rx(A):
    return np.array([[1,      0,      0],
                     [0, cos(A), -sin(A)],
                     [0, sin(A),  cos(A)]])

def Ry(A):
    return np.array([[ cos(A), 0, sin(A)],
                     [      0, 1,      0],
                     [-sin(A), 0, cos(A)]])

def Rz(A):
    return np.array([[cos(A), -sin(A), 0],
                     [sin(A),  cos(A), 0],
                     [0,            0, 1]])

def ur():
    return 2.0*random.random() - 1.0

def random_rotation_matrix(max_angle=180):
    mr = pi*(max_angle/180.0)
    # Orient the board so a white square (sq #0) in UL corner
    RY = Ry(-0.5*pi)
    # add some random rotations
    return RY @ Rx(mr*ur()) @ Ry(mr*ur()) @ Rz(mr*ur())

def random_object_position(distance_min=5.0, distance_max=40.0, width=10.0, height=10.0):
    """
    Specialized function to randomly place objects in a visible location.
    
    Args:
        distance_min: Minimum distance from camera (y-axis)
        distance_max: Maximum distance from camera (y-axis)
        width: Width of the placement area (x-axis range will be -width/2 to width/2)
        height: Height of the placement area (z-axis range will be -height/2 to height/2)
    
    Returns:
        numpy array with [x, y, z] coordinates
    """
    # X coordinate (horizontal position, left to right)
    x = random.uniform(-width/2, width/2)
    
    # Y coordinate (distance from camera, front to back)
    y = random.uniform(distance_min, distance_max)
    
    # Z coordinate (vertical position, bottom to top)
    z = random.uniform(-height/2, height/2)
    
    return np.array([x, y, z])

# ##############################################################################
# Load all models from a specified folder
# ##############################################################################

def load_objects_from_folder(folder_path):
    """
    Load all OBJ files from a folder into BlenderProc.
    
    Args:
        folder_path: Path to folder containing OBJ files with textures
    
    Returns:
        Tuple of (objects, objects_data) with loaded objects and their metadata
    """
    objects = []
    objects_data = []
    
    # Find all .obj files in the folder (non-recursive)
    ply_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ply')]
    ply_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    
    print(f"{GREEN}Found {len(ply_files)} OBJ files in {folder_path}{RESET}")
    
    # Load each object
    for idx, ply_file in enumerate(ply_files):
        model_path = os.path.join(folder_path, ply_file)
        print(f"{BLUE}Loading model: {model_path}{RESET}")
        
        # Load the object with BlenderProc
        obj = bp.loader.load_obj(model_path)[0]
        obj.set_cp("category_id", 1+idx)
        objects.append(obj)
        
        # Get object class from filename (remove extension)
        obj_class = os.path.splitext(ply_file)[0]
        obj_name = obj_class + "_" + str(idx).zfill(3)
        
        objects_data.append({
            'class': obj_class,
            'name': obj_name,
            'id': 1+idx
        })
    
    return objects, objects_data


# ##############################################################################
# Load background images from a specified folder and apply background into the scene
# ##############################################################################
def load_background_images(backgrounds_folder):
    """
    Load all supported background images from a folder recursively.
    
    Args:
        backgrounds_folder: Path to folder containing background images
    
    Returns:
        List of paths to background images
    """
    # Supported image file extensions
    image_types = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.hdr', '*.HDR')
    backdrop_images = []
    
    # Check if the folder exists
    if backgrounds_folder is None or not os.path.exists(backgrounds_folder):
        print(f"{RED}Background folder '{backgrounds_folder}' not provided or does not exist{RESET}")
        return backdrop_images
    
    # Recursively find all image files
    for ext in image_types:
        backdrop_images.extend(glob.glob(
            os.path.join(backgrounds_folder, os.path.join('**', ext)),
            recursive=True
        ))
    
    # Report results
    if len(backdrop_images) == 0:
        print(f"{RED}No images found in backgrounds directory '{backgrounds_folder}'{RESET}")
    else:
        print(f"{GREEN}{len(backdrop_images)} images found in backgrounds directory "
              f"'{backgrounds_folder}'{RESET}")
        
    return backdrop_images

def set_world_background_hdr(filename, strength=1.0, rotation_euler=None):
    """
    Sets the background with a Poly Haven HDRI file

    strength: The brightness of the background.
    rot_euler: Optional euler angles to rotate the background.
    """
    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(filename, check_existing=True)

    # get the background node of the world shader and link the new texture node
    background_node = Utility.get_the_one_node_with_type(nodes, "Background")
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # Set the brightness
    background_node.inputs["Strength"].default_value = strength

    # add a mapping node and a texture coordinate node
    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    #link the texture coordinate node to mapping node and vice verse
    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    mapping_node.inputs["Rotation"].default_value = rotation_euler

def crop_to_rotation(img, angle):
    # 'img' is a PIL Image of uint8 RGB values
    # 'angle' is in degrees
    angle_rad = angle*pi/180.0
    width, height = img.size

    img = img.rotate(angle)
    # Crop out black border resulting from rotation
    wr, hr = rotated_rectangle_extents(width, height, angle_rad)
    return crop_around_center(img, wr, hr)

def scale_to_original_shape(img, o_width, o_height):
    c_width, c_height = img.size
    o_ar = o_width/o_height
    c_ar = c_width/c_height
    if o_ar > c_ar:
        cropped = crop_around_center(img, c_width, c_width/o_ar)
    else:
        cropped = crop_around_center(img, c_height*o_ar, c_height)

    return cropped.resize((o_width, o_height))

def crop_around_center(image, width, height):
    """
    Crop 'image' (a PIL image) to 'width' and height' around the images center
    point
    """
    size = image.size
    center = (int(size[0] * 0.5), int(size[1] * 0.5))

    if(width > size[0]):
        width = size[0]

    if(height > size[1]):
        height = size[1]

    x1 = int(center[0] - width * 0.5)
    x2 = int(center[0] + width * 0.5)
    y1 = int(center[1] - height * 0.5)
    y2 = int(center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2)) # (left, upper, right, lower)

def rotated_rectangle_extents(w, h, angle):
    """
    Given a rectangle of size W x H that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same, it
    # suffices to only look at the first quadrant and the absolute values of
    # sin,cos:
    sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr

def randomize_background(path, width, height):
    img = Image.open(path)

    # Randomly rotate
    angle = 45.0 - random.random()*90.0
    img = crop_to_rotation(img, angle)
    img = scale_to_original_shape(img, width, height)

    # Randomly flip in horizontal and vertical directions
    if random.random() > 0.5:
        # flip horizontal
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        # flip vertical
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img
    
def detect_is_hdr(backdrop_images):
    """
    Detect if any of the provided background images are HDR.
    """

    if not backdrop_images:
        print(f"{RED}No background images provided.{RESET}")
        return False, None

    # Select a random background
    background_path = backdrop_images[random.randint(0, len(backdrop_images) - 1)]
    print(f"{BLUE}Using background: {background_path}{RESET}")
    
    # Check if it's an HDR image
    is_hdr = os.path.splitext(background_path)[1].lower() == ".hdr"

    return is_hdr, background_path

def setup_hdr_background(background_path, is_hdr=False):
    """
    Sets up an HDR background as world environment before rendering.
    """
    if is_hdr:
        # For HDR backgrounds, set up as world environment
        strength = random.random() + 0.5  # 0.5 to 1.5
        rotation = [
            random.random() * 0.2 - 0.1,  # Small random rotation in X
            random.random() * 0.2 - 0.1,  # Small random rotation in Y
            random.random() * 0.2 - 0.1   # Small random rotation in Z
        ]
        set_world_background_hdr(background_path, strength, rotation)
        bp.renderer.set_output_format(enable_transparency=False)
    else:
        # For regular images, just configure the renderer
        bp.renderer.set_output_format(enable_transparency=True)

def apply_regular_background(background_path, width, height, rendered_image):
    """
    Applies a regular (non-HDR) background to a rendered image.
    """
    background = randomize_background(background_path, width, height)
    background = background.convert('RGB') # some images may be B&W
    # Pasting the current image on the selected background
    background.paste(rendered_image, mask=rendered_image.convert('RGBA'))
    return background


# ##############################################################################
# Main function
# ##############################################################################

def main(num_id, width, height, scale, outf, nb_frames, focal_length=None, models_folder=None, backgrounds_folder=None, min_pixels=1, target_obj_idx=None):

    # Make output directories
    out_directory = os.path.join(outf)
    os.makedirs(out_directory, exist_ok=True)
    # make the output directory for the DOPE data
    dope_data_folder = os.path.join(out_directory, "dope_data")
    os.makedirs(dope_data_folder, exist_ok=True)
    # New custom 6D label dataset directory
    custom6d_folder = os.path.join(out_directory, "YOLO6DPose_data")
    os.makedirs(custom6d_folder, exist_ok=True)
    # make the output directory for the yolo6dpose data
    write_folders_yolo6dpose(custom6d_folder)

    # Set up blenderproc
    bp.init()
        
    # Renderer setup
    bp.renderer.set_output_format('PNG')
    bp.renderer.set_render_devices(desired_gpu_ids=[0])

    # Load the object
    print(f"{GREEN}Loading models from folder: {models_folder}{RESET}")
    objects, objects_data = load_objects_from_folder(models_folder)
    if len(objects) == 0:
        print(f"{RED}No valid OBJ files found in {models_folder}{RESET}")
        exit(1)  

    # Load background images
    print(f"{GREEN}Loading background images from folder: {models_folder}{RESET}")
    backdrop_images = load_background_images(backgrounds_folder)

    # Set the camera to be in front of the object
    cam_pose = bp.math.build_transformation_mat([0, -10, 0], [np.pi / 2, 0, 0])
    bp.camera.add_camera_pose(cam_pose)
    bp.camera.set_resolution(width, height)
    if focal_length:
        K = np.array([[focal_length, 0, width/2],
                      [0, focal_length, height/2],
                      [0,0,1]])
        bp.camera.set_intrinsics_from_K_matrix(K, width, height, clip_start=1.0,
                                               clip_end=1000.0)
    else:
        bp.camera.set_intrinsics_from_blender_params(lens=0.785398, # FOV in radians
                                                     lens_unit='FOV',
                                                     clip_start=1.0, clip_end=1000.0)
        
    # # sample point light on shell
    # light_point = bp.types.Light()
    # light_point.set_energy(200)
    # # Sample light source
    # light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    # location = bp.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
    #                         elevation_min = 5, elevation_max = 89)
    # light_point.set_location(location)

    # # Add a ground plane for reference
    # ground = bp.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 0, -5])
    # ground_material = bp.material.create('ground_material')
    # ground_material.set_principled_shader_value("Base Color", [0.2, 0.2, 0.2, 1.0])
    # ground_material.set_principled_shader_value("Roughness", 0.5)
    # ground.replace_materials(ground_material)

    bp.renderer.enable_depth_output(activate_antialiasing=False)
    bp.renderer.set_max_amount_of_samples(50)

    for frame in tqdm(range(nb_frames), desc="Syntetic Data Creation", unit="frame"):

        # Place object(s)
        for idx, oo in enumerate(objects):
            # Set a random pose
            xform = np.eye(4)
            xform[0:3,3] = random_object_position(distance_min=2.0, distance_max=2.0, width=10.0, height=10.0)
            xform[0:3,0:3] = random_rotation_matrix()
            oo.set_local2world_mat(xform)
            oo.set_scale([scale, scale, scale])
            # Update location and quaternion in objects_data for DOPE format
            xform_in_cam = np.linalg.inv(bp.camera.get_camera_pose()) @ xform
            objects_data[idx]['location'] = xform_in_cam[0:3,3].tolist()
            tmp_wxyz = Quaternion(matrix=xform_in_cam[0:3,0:3]).elements  # [scalar, x, y, z]
            q_xyzw = [tmp_wxyz[1], tmp_wxyz[2], tmp_wxyz[3], tmp_wxyz[0]] # [x, y, z, scalar]
            objects_data[idx]['quaternion_xyzw'] = q_xyzw
            # Scale 3D model
            
        # check if we have a HDR background
        is_hdr, background_path = detect_is_hdr(backdrop_images)
        # For HDR backgrounds, we need to apply before-rendering
        setup_hdr_background(background_path, is_hdr=is_hdr)

        # render the cameras of the current scene
        segs = bp.renderer.render_segmap()
        data = bp.renderer.render()
        im = Image.fromarray(data['colors'][0])

        # For non-HDR backgrounds, we need to apply post-rendering
        if not is_hdr:
            im = apply_regular_background(background_path, width, height, im)

        # Apply cube drawing for debbugging
        # im = draw_cuboid_markers(objects, bp.camera, im)

        # Write data in DOPE format
        filename = os.path.join(dope_data_folder, str(frame).zfill(6) + ".png")
        im.save(filename)

        ## Export JSON file
        filename = os.path.join(dope_data_folder, str(frame).zfill(6) + ".json")
        write_json(filename, width, height, min_pixels, bp.camera, objects, objects_data, segs['class_segmaps'][0])

        # Write data in bop format
        bp.writer.write_bop(os.path.join(out_directory, 'bop_data'),
                            target_objects=objects,
                            dataset='lm',
                            depth_scale=1.0,
                            depths=data["depth"],
                            colors=data["colors"],
                            color_file_format="JPEG",
                            ignore_dist_thres=20, 
                            frames_per_chunk=10000)
        

        # Write data in YOLO6DPose format. !THIS MUST BE AFTER THE BOP BECAUSE IT USE THE SAME MASK IMAGES!
        write_data_yolo6dpose(custom6d_folder, out_directory, im, frame, width, height, bp.camera, objects, objects_data, target_obj_idx)

if __name__ == "__main__":
    num_id = 0
    width = 500
    height = 500
    scale = 0.01  #the object scale is meters -> scale=0.01; if it is in cm -> scale=1.0: if if it is in mm -> scale=0.001. 
    # Also you can use whatever scale you want to make the object bigger or smaller, As long as you apply the scale consistently throughout your pipeline. 
    outf = "datasets/"
    nb_frames = 10
    focal_length = None
    models_folder = "models/"
    backgrounds_folder = "backgrounds/"
    min_pixels = 1
    target_obj_idx = 0 #for data yolo6dpose

    print("=========================================")
    print("Starting BlenderProc")

    main(
        num_id=num_id,
        width=width,
        height=height,
        scale=scale,
        outf=outf,
        nb_frames=nb_frames,
        focal_length=focal_length,
        models_folder=models_folder,
        backgrounds_folder=backgrounds_folder,
        min_pixels=min_pixels,
        target_obj_idx = target_obj_idx
    )
