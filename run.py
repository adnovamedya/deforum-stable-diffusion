import argparse
import json

parser = argparse.ArgumentParser(description='deforum informations')
parser.add_argument('--process_id', nargs='?', dest='process_id', type=str, help='process id')
parser.add_argument('--args', nargs='?', dest='args', type=str, help='deforum args')
parser.add_argument('--anim_args', nargs='?', dest='anim_args', type=str, help='deforum animation args')

command_args = parser.parse_args()

if not command_args.process_id:
    print('process_id required')
    raise RuntimeError('process_id required')

process_id = command_args.process_id

if not command_args.args:
    print('deforum args required')
    raise RuntimeError('args required')

if not command_args.anim_args:
    print('deforum animation args required')
    raise RuntimeError('deforum animation args required')

args_dict = json.loads(command_args.args)
anim_args_dict = json.loads(command_args.anim_args)

import subprocess, os, time, gc

sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(f"{sub_p_res[:-1]}")

# setup environment
from setup import Setup
Setup()

import torch
import random
import clip
from types import SimpleNamespace
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts

from root import Root
root = Root().getLocals()
root = SimpleNamespace(**root)

root.deforum_images_path = f"{root.deforum_images_path}/{process_id}"
root.deforum_videos_path = f"{root.deforum_videos_path}/{process_id}"
root.models_path, root.output_path = get_model_output_paths(root)
root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

args.outdir = f"outputs/deforum_images/{process_id}"
args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

# Load clip model if using clip guidance
if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    if (args.aesthetics_scale > 0):
        root.aesthetics_model = load_aesthetics_model(args, root)

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None
if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
    print(f"Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = 'klms'
if args.sampler != 'ddim':
    args.ddim_eta = 0

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

# clean up unused memory
gc.collect()
torch.cuda.empty_cache()

# get prompts
cond, uncond = Prompts(prompt=args.animation_prompts,neg_prompt=args.negative_prompts).as_dict()

# dispatch to appropriate renderer
if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
    render_animation(root, anim_args, args, cond, uncond)
elif anim_args.animation_mode == 'Video Input':
    render_input_video(root, anim_args, args, cond, uncond)
elif anim_args.animation_mode == 'Interpolation':
    render_interpolation(root, anim_args, args, cond, uncond)
else:
    render_image_batch(root, args, cond, uncond)

import subprocess
render_steps = False  #@param {type: 'boolean'}
path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
make_gif = False
bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"

max_frames = str(anim_args.max_frames)
mp4_path = f"outputs/deforum_videos/{process_id}/result.mp4"

if render_steps:  # render steps from a single image
    fname = f"{path_name_modifier}_%05d.png"
    all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if
                     os.path.isdir(os.path.join(args.outdir, d))]
    newest_dir = max(all_step_dirs, key=os.path.getmtime)
    image_path = os.path.join(newest_dir, fname)
    print(f"Reading images from {image_path}")
    max_frames = str(args.steps)
else:  # render images for a video
    image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")

# make video
cmd = [
    'ffmpeg',
    '-y',
    '-vcodec', bitdepth_extension,
    '-r', str(anim_args.fps),
    '-start_number', str(0),
    '-i', image_path,
    '-frames:v', max_frames,
    '-c:v', 'libx264',
    '-vf',
    f'fps={anim_args.fps}',
    '-pix_fmt', 'yuv420p',
    '-crf', '17',
    '-preset', 'veryfast',
    '-pattern_type', 'sequence',
    mp4_path
]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode != 0:
    print(stderr)
    raise RuntimeError(stderr)
