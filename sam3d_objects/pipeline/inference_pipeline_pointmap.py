# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union, Optional
from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
import numpy as np
import torch
from tqdm import tqdm
from torch.utils._pytree import tree_map_only
from loguru import logger
from functools import wraps
from PIL import Image

from pytorch3d.transforms import Transform3d

from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.model.backbone.trellis.utils import postprocessing_utils
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.trellis.pose_loader import R3
from sam3d_objects.data.dataset.tdfy.trellis.dataset import PerSubsetDataset
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import normalize_pointmap_ssi
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from copy import deepcopy


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


class InferencePipelinePointMap(InferencePipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method

        super().__init__(*args, **kwargs)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"

        # self.depth_model.model.forward = compile_wrapper(
        #     self.depth_model.model.forward,
        #     mode=compile_mode,
        #     fullgraph=True,
        # )

        for embedder, _ in self.condition_embedders[
            "ss_condition_embedder"
        ].embedder_list:
            if isinstance(embedder, PointPatchEmbed):
                logger.info("Found PointPatchEmbed")
                embedder.inner_forward = compile_wrapper(
                    embedder.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
            else:
                embedder.forward = compile_wrapper(
                    embedder.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )

        self.models["ss_generator"].reverse_fn.inner_forward = compile_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode=compile_mode,
            fullgraph=True,
        )

        if self.models["layout_model"] is not None:
            self.models["layout_model"].reverse_fn.inner_forward = compile_wrapper(
                self.models["layout_model"].reverse_fn.inner_forward,
                mode=compile_mode,
                fullgraph=True,
            )

            if self.condition_embedders["layout_condition_embedder"] is not None:
                # we move the condition embedder outside the reverse_fn
                embedder_list = self.condition_embedders[
                    "layout_condition_embedder"
                ].embedder_list
            else:
                raise NotImplementedError

            for embedder, _ in embedder_list:
                if isinstance(embedder, PointPatchEmbed):
                    logger.info("Found PointPatchEmbed")
                    embedder.inner_forward = compile_wrapper(
                        embedder.inner_forward,
                        mode=compile_mode,
                        fullgraph=True,
                    )
                else:
                    embedder.forward = compile_wrapper(
                        embedder.forward,
                        mode=compile_mode,
                        fullgraph=True,
                    )

        self.models["ss_decoder"].forward = compile_wrapper(
            self.models["ss_decoder"].forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)
        with torch.inference_mode(False):
            with torch.no_grad():
                for _ in tqdm(range(num_warmup_iters)):
                    pointmap_dict = recursive_clone(self.compute_pointmap(image))
                    pointmap = pointmap_dict["pointmap"]

                    ss_input_dict = self.preprocess_image(
                        image, self.ss_preprocessor, pointmap=pointmap
                    )
                    if self.models["layout_model"] is not None:
                        layout_input_dict = self.preprocess_image(
                            image, self.layout_preprocessor, pointmap=pointmap
                        )
                    else:
                        layout_input_dict = {}
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict, inference_steps=None
                    )

                    _ = self.run_layout_model(
                        layout_input_dict,
                        ss_return_dict,
                        inference_steps=None,
                    )

    def _preprocess_image_and_mask_pointmap(
        self, rgb_image, mask_image, pointmap, img_mask_pointmap_joint_transform
    ):
        for trans in img_mask_pointmap_joint_transform:
            rgb_image, mask_image, pointmap = trans(
                rgb_image, mask_image, pointmap=pointmap
            )
        return rgb_image, mask_image, pointmap

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        # Check if we need to normalize the pointmap for layout preprocessing
        if pointmap is not None and preprocessor.normalize_pointmap:
            pointmap, _, _ = normalize_pointmap_ssi(pointmap)

        if (
            preprocessor.img_mask_pointmap_joint_transform != (None,)
            and preprocessor.img_mask_pointmap_joint_transform is not None
            and pointmap is not None
        ):
            processed_rgb_image, processed_mask, processed_rgb_pointmap = (
                self._preprocess_image_and_mask_pointmap(
                    rgb_image,
                    rgb_image_mask,
                    pointmap,
                    preprocessor.img_mask_pointmap_joint_transform,
                )
            )
        else:
            processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
                rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
            )
            processed_rgb_pointmap = pointmap

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, preprocessor.img_transform
        )
        processed_mask = self._apply_transform(
            processed_mask, preprocessor.mask_transform
        )
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            processed_rgb_pointmap = self._apply_transform(
                processed_rgb_pointmap,
                preprocessor.pointmap_transform,
            )

        # full image, with only processing from the image
        rgb_image = self._apply_transform(rgb_image, preprocessor.img_transform)
        rgb_image_mask = self._apply_transform(
            rgb_image_mask, preprocessor.mask_transform
        )
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            full_pointmap = self._apply_transform(
                pointmap, preprocessor.pointmap_transform
            )
        item = {
            "mask": processed_mask[None].to(self.device),
            "image": processed_rgb_image[None].to(self.device),
            "rgb_image": rgb_image[None].to(self.device),
            "rgb_image_mask": rgb_image_mask[None].to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = processed_rgb_pointmap[None].to(self.device)
            item["rgb_pointmap"] = full_pointmap[None].to(self.device)

        return item

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(R3.r3_camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate loaded_image to match points_tensor size
                # points_tensor has shape [H, W, 3], we need H and W
                loaded_image = torch.nn.functional.interpolate(
                    loaded_image.unsqueeze(0),
                    size=(points_tensor.shape[0], points_tensor.shape[1]),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            intrinsics = None

        # Prepare the point map tensor
        point_map_tensor = PerSubsetDataset._prepare_pointmap(
            points_tensor, return_pointmap=True
        )
        point_map_tensor["pts_color"] = loaded_image

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor, device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]

        return point_map_tensor

    def run_post_optimization(self, mesh_glb, intrinsics, pose_dict, layout_input_dict):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal
        revised_quat, revised_t, revised_scale, final_iou, _, _ = (
            self.layout_post_optimization_method(
                mesh_glb,
                pose_dict["rotation"],
                pose_dict["translation"],
                pose_dict["scale"],
                layout_input_dict["rgb_image_mask"][0, 0],
                layout_input_dict["rgb_pointmap"][0].permute(1, 2, 0),
                intrinsics,
                min_size=518,
            )
        )
        return {
            "quaternion": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
        }

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
    ) -> dict:
        logger.info("InferencePipelinePointMap.run() called")
        # This should only happen if called from demo
        image = self.merge_image_and_mask(image, mask)
        with self.device:  # TODO(Pierre) make with context a decorator ?
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]
            pointmap_scale = pointmap_dict["pointmap_scale"]
            pointmap_shift = pointmap_dict["pointmap_shift"]

            ss_input_dict = self.preprocess_image(
                image, self.ss_preprocessor, pointmap=pointmap
            )
            if self.models["layout_model"] is not None:
                layout_input_dict = self.preprocess_image(
                    image, self.layout_preprocessor, pointmap=pointmap
                )
            else:
                layout_input_dict = {}
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            # This is for decoupling oriented shape and layout model
            # ss_input_dict["x_shape_latent"] = ss_return_dict["shape"]
            layout_return_dict = self.run_layout_model(
                layout_input_dict,
                ss_return_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )
            ss_return_dict.update(layout_return_dict)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            glb = outputs.get("glb", None)

            try:
                if (
                    with_layout_postprocess
                    and self.layout_post_optimization_method is not None
                ):
                    assert glb is not None, "require mesh to run postprocessing"
                    logger.info("Running layout post optimization method...")
                    postprocessed_pose = self.run_post_optimization(
                        deepcopy(glb),
                        pointmap_dict["intrinsics"],
                        ss_return_dict,
                        layout_input_dict,
                    )
                    ss_return_dict.update(postprocessed_pose)
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            # glb.export("sample.glb")
            logger.info("Finished!")

            pts = type(self)._down_sample_img(pointmap)
            pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

            return {
                **ss_return_dict,
                **outputs,
                "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
            }

    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor):
        # img_3chw: (3, H, W)
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        max_side = max(x.shape[2], x.shape[3])
        scale_factor = 1.0

        # heuristics
        if max_side > 3800:
            scale_factor = 0.125
        if max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )  # -> (1, 3, H/4, W/4)
        return x.squeeze(0)
