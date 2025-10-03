# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
from typing import Union, Optional
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
import math
import utils3d
from copy import deepcopy
from kaolin.visualize import IpyTurntableVisualizer
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.trellis.utils import render_utils

from sam3d_objects.utils.visualization import SceneVisualizer

__all__ = ["Inference"]

# concatenate "li" + "dra" to skip the automated string replacement
if "li" + "dra" not in sys.modules:
    sys.modules["li" + "dra"] = sam3d_objects


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
        pointmap=None,  # TODO(Pierre) : add pointmap type
    ) -> dict:
        # enable or disable layout model
        return self._pipeline.run(
            image,
            mask,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                ]
            ).cuda()
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 1, 0]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_video_ring(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,  # radius of the ring
    pitch_deg=15,  # elevation angle above the xy-plane
    fov=40,
    **kwargs,
):
    # 1) angular positions around the ring
    yaws = torch.linspace(0, 2 * torch.pi, num_frames).tolist()

    # 2) constant pitch for every frame
    pitch = torch.full((num_frames,), torch.deg2rad(torch.tensor(pitch_deg))).tolist()

    # 3) look-at cameras centred at (0,0,0)
    extrinsics, intrinsics = _yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws,
        pitch,
        r,
        fov,
    )

    # 4) render
    return render_utils.render_frames(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )


def render_video_flat(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,
    fov=40,
    pitch_deg=0,
    yaw_start_deg=-90,
    **kwargs,
):

    yaws = (
        torch.linspace(0, 2 * torch.pi, num_frames) + math.radians(yaw_start_deg)
    ).tolist()
    pitch = [math.radians(pitch_deg)] * num_frames

    extr, intr = _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)

    return render_utils.render_frames(
        sample,
        extr,
        intr,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )


def ready_gaussian_for_video_rendering(scene_gs, in_place=False):
    scene_gs = _fix_gaussian_alignment(scene_gs, in_place=in_place)
    scene_gs = normalized_gaussian(scene_gs, in_place=True)
    return scene_gs


def _fix_gaussian_alignment(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    device = scene_gs._xyz.device
    dtype = scene_gs._xyz.dtype
    scene_gs._xyz = (
        scene_gs._xyz
        @ torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], device=device, dtype=dtype).T
    )
    return scene_gs


def normalized_gaussian(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    orig_xyz = scene_gs._xyz
    orig_scale = scene_gs._scaling

    norm_scale = orig_scale / (orig_xyz.max(dim=0)[0] - orig_xyz.min(dim=0)[0]).max()
    norm_xyz = orig_xyz / (orig_xyz.max(dim=0)[0] - orig_xyz.min(dim=0)[0]).max()
    norm_xyz = norm_xyz - norm_xyz.min(dim=0)[0]
    scene_gs._xyz = norm_xyz
    scene_gs._scaling = norm_scale
    return scene_gs


# gaussian visualizer


def _make_render_fn(scene_gs):
    def _gaussian_renderer(camera: Camera):
        # convert kaolin camera extrinsics & instrinsics
        # to format undertstood by `render_gaussian_color_stay_in_device`
        # REMARK(Pierre) This is the wrong conversion, but works in the sense it displays something
        extr = camera.extrinsics.view_matrix()
        extr[:, 1, 1] *= -1
        extr[:, 2, 2] *= -1
        extr[:, 2, 3] *= -1
        intr = torch.tensor(
            [
                [
                    [0.8660, 0.0000, 0.5000],
                    [0.0000, 0.8660, 0.5000],
                    [0.0000, 0.0000, 1.0000],
                ]
            ],
            device=camera.device,
        )

        color = render_utils.render_gaussian_color_stay_in_device(
            scene_gs, extr, intr, verbose=False
        )["color"][0]
        return color

    return _gaussian_renderer


def _make_lowres_cam(in_cam, factor=8):
    lowres_cam = deepcopy(in_cam)
    lowres_cam.width = in_cam.width // factor
    lowres_cam.height = in_cam.height // factor
    return lowres_cam


def _make_lowres_render_func(render_func):
    def lowres_render_func(in_cam):
        return render_func(_make_lowres_cam(in_cam))

    return lowres_render_func


def get_gaussian_splatting_visualizer(scene_gs, device="cuda"):
    # scene_gs is supposed to be centered and normalized
    camera = Camera(
        extrinsics=CameraExtrinsics.from_lookat(
            eye=[0, 0, 2], at=[0, 0, 0], up=[0, 1, 0], device=device
        ),
        intrinsics=PinholeIntrinsics.from_fov(256, 256, fov=60, device=device),
    )

    render_fn = _make_render_fn(scene_gs)

    # Create the visualizer
    vizualizer = IpyTurntableVisualizer(
        width=256,
        height=256,
        camera=camera,
        render=render_fn,
        max_fps=10,
        fast_render=_make_lowres_render_func(render_fn),
    )

    return vizualizer


# def prepare_gaussian_outputs(outputs):
#     all_outs = []
#     for output in outputs:
#         PC = SceneVisualizer.object_pointcloud(
#             points_local=(output["gs"]._xyz - 0.5).unsqueeze(0),
#             quat_l2c=output["rotation"],
#             trans_l2c=output["translation"],
#             scale_l2c=output["scale"],
#         )

#         output["gs"]._xyz = PC.points_list()[0]
#         output["gs"]._scaling *= output["scale"]
#         all_outs.append(output)

#     scene_gs = all_outs[0]["gs"]
#     for out in all_outs[1:]:
#         out_gs = out["gs"]
#         scene_gs._xyz = torch.cat([scene_gs._xyz, out_gs._xyz], dim=0)
#         scene_gs._features_dc = torch.cat(
#             [scene_gs._features_dc, out_gs._features_dc],
#             dim=0,
#         )
#         scene_gs._scaling = torch.cat([scene_gs._scaling, out_gs._scaling], dim=0)
#         scene_gs._rotation = torch.cat([scene_gs._rotation, out_gs._rotation], dim=0)
#         scene_gs._opacity = torch.cat([scene_gs._opacity, out_gs._opacity], dim=0)

#     pose_targets = []
#     for out in all_outs:
#         converted_out = {
#             "xyz_local": (out["coords"][:, 1:] / 64 - 0.5),
#             "rotation": out["rotation"],
#             "translation": out["translation"],
#             "scale": out["scale"],
#         }
#         pose_targets.append(converted_out)

#     return scene_gs, pose_targets
