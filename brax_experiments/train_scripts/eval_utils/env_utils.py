import jax.numpy as jnp
import jax
import numpy as np
from brax import base, envs, math
import numpy as onp
from collections import deque
import os
from PIL import Image
from functools import partial

from typing import (
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Dict,
    Tuple,
    Any,
    Callable,
    Union,
)

import brax

# FOR NEW BRAX
#from brax.io import image

from brax.envs.base import PipelineEnv, State
from brax.training.acme.types import NestedArray
from brax.training.types import PRNGKey
from flax import struct
import flax

from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import (
    Renderer,
    UpAxis,
    create_capsule,
    create_cube,
    transpose_for_display,
)
import trimesh


def imread(filename, hw):
    img = Image.open(filename).resize(size=(hw, hw))
    img_np = onp.asarray(img)
    return img_np


def make_pixel_brax(
    backend: str,
    env_name: str,
    n_envs,
    seed,
    hw=84,
    distractor=None,
    video_path=None,
    video_set="train",
    alpha=0.5,
    action_repeat=1,
    return_float32=True,
    experimental=False
):
    assert backend in ["generalized", "positional", "spring"]
    assert video_set in ["train", "test"]

    # Checking the video path given. We'll be caching all of these video fraes
    if distractor == "videos":
        # 1st two folders are "train", rest(?) are "test"
        video_path_root = f"{video_path}/JPEGImages/480p/"
        full_video_path = sorted(os.listdir(video_path_root))
        train_video_folders = full_video_path[:2]
        test_video_folders = full_video_path[2:8]

        BG_FRAMES = []

        for folder in (
            train_video_folders if video_set == "train" else test_video_folders
        ):
            video_frame_files = sorted(os.listdir(f"{video_path_root}/{folder}"))
            inner_frames = []
            # These video frames are coming in as uint8, so no need to change them when return_float32 is False
            for file in video_frame_files:
                _frame = imread(f"{video_path_root}/{folder}/{file}", hw)
                inner_frames.append(jnp.array(_frame))
            BG_FRAMES.append(inner_frames)

        VIDEO_LEN = jnp.min(jnp.array([len(x) for x in BG_FRAMES]))
        BG_FRAMES = jnp.array([x[:VIDEO_LEN] for x in BG_FRAMES])
        # print(f'BG_FRAMES: {len(BG_FRAMES)} // {[len(x) for x in BG_FRAMES]}')

    else:
        BG_FRAMES = [jnp.zeros((1,)) for _ in range(n_envs)]

    # We could have some other file that stores the presets for the environments.
    # But that would be this exact code (maybe in .yaml structure) just in a different file, right?
    # If that's the case, we'll need to add some extra machinery (e.g., util to read from the other file, some
    # lookup functions). So, we'd just end up with more code && the future reader would need to sift through *yet another*
    # file. Seems like a waste of everyone's time to me.
    if env_name == "reacher":
        if backend != "generalized":
            raise AttributeError(f"Physics backend needs to be generalized.")
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([0.0, -0.01, 0.8])
        CAM_UP = jnp.array([0.0, 1.0, 1.0])
        HFOV = 45.0

    elif env_name == "swimmer":
        if backend != "generalized":
            raise AttributeError(f"Physics backend needs to be generalized.")
        CAMERA_TARGET = 1
        CAM_EYE = 1
        CAM_OFF = jnp.array([2.4, 2.4, -0.2])  # jnp.array([-1.2, 1.2, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        # The purpose of CAM_Z is to have a constant location along the z-axis for the camera
        # This is useful if the goems may move up/down (e.g., locomotion envs)
        CAM_Z = 1.25
        HFOV = 50.0

    elif env_name == "pusher":
        if backend != "generalized":
            raise AttributeError(f"Physics backend needs to be generalized.")
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([1.6, 1.6, -0.2])  # jnp.array([-1.2, 1.2, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        # The purpose of CAM_Z is to have a constant location along the z-axis for the camera
        # This is useful if the goems may move up/down (e.g., locomotion envs)
        CAM_Z = 0.75
        HFOV = 50.0

    elif env_name == "halfcheetah":
        if backend != "spring":
            raise AttributeError(f"Physics backend needs to be spring.")
        # [7, 3]
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([0.0, -2.1, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        CAM_Z = 0.8
        HFOV = 55.0

    elif env_name == "hopper":
        if backend != "positional":
            raise AttributeError(f"Physics backend needs to be positional.")
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([0.0, -2.1, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        CAM_Z = 0.8
        HFOV = 55.0

    elif env_name == "inverted_pendulum":
        if backend != "generalized":
            raise AttributeError(f"Physics backend needs to be generalized.")
        # [2, 3]
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([0.0, -2.3, 0.0])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        HFOV = 55.0

    elif env_name == "ant":
        # [9, 3]
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([1.6, -1.6, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        # The purpose of CAM_Z is to have a constant location along the z-axis for the camera
        # This is useful if the goems may move up/down (e.g., locomotion envs)
        CAM_Z = 3.5
        HFOV = 40.0
    elif env_name == "walker2d":
        if backend != "spring":
            raise AttributeError(f"Physics backend needs to be generalized.")
        # [7, 3]
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([0.0, -2.1, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        CAM_Z = 0.8
        HFOV = 55.0
    elif "humanoid" in env_name:
        # if backend != 'generalized':
        #   raise AttributeError(f'Physics backend needs to be generalized.')
        # [11, 3]
        CAMERA_TARGET = 0
        CAM_EYE = 0
        CAM_OFF = jnp.array([1.7, -1.7, -0.2])
        CAM_UP = jnp.array([0.0, 0.0, 1.0])
        # The purpose of CAM_Z is to have a constant location along the z-axis for the camera
        # This is useful if the goems may move up/down (e.g., locomotion envs)
        CAM_Z = 1.5
        HFOV = 50.0

    # Now that we have set up our rendering constants, we can create the env
    # This env comes with an autoreset wrapper by default. What to do with this>?
    env = envs.create(env_name=env_name, backend=backend, action_repeat=action_repeat)
    seed_key = jax.random.PRNGKey(seed=seed)
    ret = jax.jit(jax.random.split, static_argnames=("num",))(seed_key, num=n_envs)
    seed_key = ret[0]
    seed_keys = ret[1:]

    # print(env)

    initial_states = jax.vmap(env.reset)(seed_keys)

    # remove this. we only need to jit the **outer** step fn
    _env_step_fn = jax.jit(jax.vmap(env.step))

    """Image rendering code"""
    canvas_width = hw
    canvas_height = hw

    # The groundplane used in the environment. This component is important for the locomoation
    # tasks, as the agent needs to understand the contact between the plane and the rigid body
    def grid(grid_size: int, color) -> jnp.ndarray:
        grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.single)
        grid[:, :] = onp.array(color) / 255.0
        grid[0] = onp.zeros((grid_size, 3), dtype=onp.single)
        # to reverse texture along y direction
        grid[:, -1] = onp.zeros((grid_size, 3), dtype=onp.single)
        return jnp.asarray(grid)

    _GROUND: jnp.ndarray = grid(hw, [200, 200, 200])
    # print(f'_GROUND: {_GROUND} // {_GROUND.shape}')
    # qqq

    # TODO; make every grom primitive congruent. That way we can vmap over the objects instead of looping over
    #  them. This could grant us a large speedup.
    class Obj(NamedTuple):
        """An object to be rendered in the scene.

        Assume the system is unchanged throughout the rendering.

        col is accessed from the batched geoms `sys.geoms`, representing one geom.
        """

        
        link_idx: int
        """col.link_idx if col.link_idx is not None else -1"""
        off: jnp.ndarray
        """col.transform.rot"""
        rot: jnp.ndarray
        """col.transform.rot"""
        instance: Optional[Instance] = None
        """An instance to be rendered in the scene, defined by jaxrenderer."""

    @jax.jit
    def _build_objects(sys: brax.System) -> list[Obj]:
        """
        Converts a brax System to a list of Obj.

        Args:
          sys:

        Returns:

        """
        objs: list[Obj] = []

        def take_i(obj, i):
            return jax.tree_map(lambda x: jnp.take(x, i, axis=0), obj)

        testing = []
        for batch in sys.geoms:
            num_geoms = len(batch.friction)
            inner = []
            for i in range(num_geoms):
                inner.append(take_i(batch, i))
            testing.append(inner)

        for geom in testing:
            for col in geom:
                tex = col.rgba[:3].reshape((1, 1, 3))
                # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
                specular_map = jax.lax.full(tex.shape[:2], 2.0)

                if isinstance(col, base.Capsule):
                    half_height = col.length / 2
                    model = create_capsule(
                        radius=col.radius,
                        half_height=half_height,
                        up_axis=UpAxis.Z,
                        diffuse_map=tex,
                        specular_map=specular_map,
                    )
                elif isinstance(col, base.Box):
                    model = create_cube(
                        half_extents=col.halfsize,
                        diffuse_map=tex,
                        texture_scaling=jnp.array(16.0),
                        specular_map=specular_map,
                    )
                elif isinstance(col, base.Sphere):
                    model = create_capsule(
                        radius=col.radius,
                        half_height=jnp.array(0.0),
                        up_axis=UpAxis.Z,
                        diffuse_map=tex,
                        specular_map=specular_map,
                    )
                elif isinstance(col, base.Plane):
                    tex = _GROUND
                    model = create_cube(
                        half_extents=jnp.array([1000.0, 1000.0, 0.0001]),
                        diffuse_map=tex,
                        texture_scaling=jnp.array(8192.0),
                        specular_map=specular_map,
                    )
                elif isinstance(col, base.Convex):
                    # convex objects are not visual
                    continue
                elif isinstance(col, base.Mesh):
                    tm = trimesh.Trimesh(vertices=col.vert, faces=col.face)
                    model = RendererMesh.create(
                        verts=tm.vertices,
                        norms=tm.vertex_normals,
                        uvs=jnp.zeros((tm.vertices.shape[0], 2), dtype=int),
                        faces=tm.faces,
                        diffuse_map=tex,
                    )
                else:
                    raise RuntimeError(f"unrecognized collider: {type(col)}")

                i: int = col.link_idx if col.link_idx is not None else -1
                instance = Instance(model=model)
                off = col.transform.pos
                rot = col.transform.rot
                obj = Obj(instance=instance, link_idx=i, off=off, rot=rot)
                objs.append(obj)
        return objs

    def _with_state(objs: Iterable[Obj], x: brax.Transform) -> list[Instance]:
        """x must have at least 1 element. This can be ensured by calling
        `x.concatenate(base.Transform.zero((1,)))`. x is `state.x`.

        This function does not modify any inputs, rather, it produces a new list of
        `Instance`s.
        """
        # if (len(x.pos.shape), len(x.rot.shape)) != (2, 2):
        #   raise RuntimeError('unexpected shape in state')

        instances: list[Instance] = []

        for obj in objs:
            i = obj.link_idx
            pos = x.pos[i] + math.rotate(obj.off, x.rot[i])
            rot = math.quat_mul(x.rot[i], obj.rot)
            instance = obj.instance
            instance = instance.replace_with_position(pos)
            instance = instance.replace_with_orientation(rot)
            instances.append(instance)
        return instances

    def _eye(sys: brax.System, state: brax.State) -> jnp.ndarray:
        """
        Determines the camera location for a Brax system as a position relative to a given geom in the brax System.
        This is computed as an  "offset" from the location of the geom.
        Args:
          sys:
          state:

        Returns:

        """
        """"""
        if env_name == "reacher":
            # The geom we are attaching the camera to does not move, so we don't need anything special
            return state.x.pos[CAM_EYE, :] + CAM_OFF
        elif (
            env_name
            in ["halfcheetah", "ant", "walker2d", "pusher", "swimmer", "hopper"]
            or "humanoid" in env_name
        ):
            # All the geoms the camera can attach to are going to be moving.
            # We want the camera to be steady. This can be done with idx 2 of <x,y,z>
            cam_eye = state.x.pos[CAM_EYE, :] + CAM_OFF
            return cam_eye.at[2].set(CAM_Z)
        elif env_name == "inverted_pendulum":
            return CAM_OFF

    def _up(unused_sys: brax.System) -> jnp.ndarray:
        """Determines the up orientation of the camera."""
        # [0,1,1] [1,1,1] weird angle isometric
        # return jnp.array([0., 0., 0.])
        # return jnp.array([0., 0., 1.])
        return CAM_UP

    def get_target(state: brax.State) -> jnp.ndarray:
        """Gets target of camera. I.e., the center of the camera's viewport"""
        if env_name in ["reacher", "ant", "pusher", "swimmer"]:
            return jnp.array(
                [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 0]
            )
        elif env_name in ["halfcheetah", "walker2d", "hopper"]:
            return jnp.array(
                [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 0.6]
            )
        elif "humanoid" in env_name:
            return jnp.array(
                [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 1.1]
            )
        elif env_name == "inverted_pendulum":
            return jnp.array([0.0, 0.0, 0.0])

    def get_camera(
        sys: brax.System,
        state: brax.State,
        width: int = canvas_width,
        height: int = canvas_height,
    ) -> Camera:
        """Gets camera object."""
        eye, up = _eye(sys, state), _up(sys)

        hfov = HFOV  # orig: 58.0 -- higher == more zoomed out
        vfov = hfov * height / width

        # Position of the camera && target need to be updated
        # _eye determines the position
        # get_target updates the target
        target = get_target(state)

        camera = Camera(
            viewWidth=width,
            viewHeight=height,
            position=eye,
            target=target,
            up=up,
            hfov=hfov,
            vfov=vfov,
        )

        return camera

    @jax.default_matmul_precision("float32")
    def render_instances(
        instances: list[Instance],
        width: int,
        height: int,
        camera: Camera,
        light: Optional[Light] = None,
        shadow: Optional[Shadow] = None,
        camera_target: Optional[jnp.ndarray] = None,
        enable_shadow: bool = False,
    ) -> jnp.ndarray:
        """Renders an RGB array of sequence of instances.

        Rendered result is not transposed with `transpose_for_display`; it is in
        floating numbers in [0, 1], not `uint8` in [0, 255].
        """
        if light is None:
            direction = jnp.array([0.57735, -0.57735, 0.57735])
            light = Light(
                direction=direction,
                ambient=0.8,
                diffuse=0.8,
                specular=0.6,
            )

        img = Renderer.get_camera_image(
            objects=instances,
            light=light,
            camera=camera,
            width=width,
            height=height,
            shadow_param=shadow,
        )
        arr = jax.lax.clamp(0.0, img, 1.0)
        return arr

    _get_instances = jax.jit(
        jax.vmap(
            lambda objs, state: _with_state(
                objs, state.x.concatenate(base.Transform.zero((1,)))
            ),
            in_axes=(None, 0),
        )
    )

    _inner_inner_render = jax.jit(
        render_instances,
        static_argnames=("width", "height", "enable_shadow"),
        inline=True,
    )

    def inner_render(instances, camera, target) -> jnp.ndarray:
        img = _inner_inner_render(
            instances=instances,
            width=canvas_width,
            height=canvas_height,
            camera=camera,
            camera_target=target,
        )
        return img

    _inner_render = jax.jit(jax.vmap(inner_render))

    def render(objs, states: brax.State, batched_camera, batched_target) -> jnp.ndarray:
        batched_instances = _get_instances(objs, states)
        images = _inner_render(batched_instances, batched_camera, batched_target)
        return images

    _get_cameras = jax.jit(
        jax.vmap(lambda sys, state: get_camera(sys, state), in_axes=(None, 0))
    )
    _get_targets = jax.jit(jax.vmap(get_target))
    _render = jax.jit(render)

    @jax.jit
    def render_pixels(sys: brax.System, pipeline_states: brax.State):
        # (1) grab the cameras and the view targets. The camera object contains its own view target
        # the extra bit we grab with _get_targets() is only used to render shadows. Maybe we can remove?
        # print(f'Within render pixels')

        # The "current_frame" arg is meant to work for the "video distractor" case
        batched_camera = _get_cameras(sys, pipeline_states)
        # print(f'after batched_camera')
        batched_target = _get_targets(pipeline_states)
        # print(f'after batched_target')
        objs = _build_objects(sys)
        # print(f'after _build_objects')
        images = _render(objs, pipeline_states, batched_camera, batched_target)
        # print(f'after _render')
        # return None
        return images

    @struct.dataclass
    class State(base.Base):
        """Environment state for training and inference."""

        pipeline_state: Optional[base.State]
        obs: jax.Array
        pixels: jax.Array
        reward: jax.Array
        done: jax.Array
        key: Optional[PRNGKey]
        frame_idx: jax.Array
        video_idx: jax.Array
        metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
        info: Dict[str, Any] = struct.field(default_factory=dict)
    
    """EXPERIMENTAL BLOCK"""
    _inner_inner_render_experimental = jax.jit(
        render_instances,
        static_argnames=("width", "height", "enable_shadow"),
        inline=True,
    )
    # @partial(jax.jit, static_argnames="hw")
    def inner_render_experimental(instances, camera, target, hw) -> jnp.ndarray:
        img = _inner_inner_render_experimental(
            instances=instances,
            width=hw,
            height=hw,
            camera=camera,
            camera_target=target,
        )
        return img


    _inner_render_experimental = jax.vmap(inner_render_experimental, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnames=("geom_id", "geom_num"))
    def _vmap_build(
        sys: brax.System,
        pipeline_states,
        specular_map: jnp.ndarray,
        tex: jnp.ndarray,
        geom_id: int,
        geom_num: int,
    ):
        """ """
        # Plane
        if geom_id == 0:
            model = create_cube(
                half_extents=jnp.array([1000.0, 1000.0, 0.0001]),
                texture_scaling=jnp.array(8192.0),
                diffuse_map=_GROUND,
                specular_map=specular_map,
            )
        # HFIELD
        elif geom_id == 1:
            raise NotImplementedError("Have not impl HFIELD yet.")
        # Sphere
        elif geom_id == 2:
            # sphere geom_rbound = geom_size[0]
            print(f"======== SPHERE =========")
            print(f"geom_rbound: {sys.mj_model.geom_rbound[geom_num]}")
            print(f"geom_size: {sys.mj_model.geom_size[geom_num]}")
            print("==========================")
            # is sys.mj_model.geom_rbound == sys.mj_model.geom_size[0]?
            radius = sys.mj_model.geom_rbound[geom_num]
            model = create_capsule(
                radius=radius,
                half_height=jnp.array(0.0),
                up_axis=UpAxis.Z,
                diffuse_map=tex,
                specular_map=specular_map,
            )
        # Capsule
        elif geom_id == 3:
            # capsule geom_rbound = geom_size[0] + geom_size[1]
            # capsule geom_size[0] and geom_size[1] is not always same
            # capsule geom_size[2] is always 0
            print(f"======== CAPSULE =========")
            print(f"geom_rbound: {sys.mj_model.geom_rbound[geom_num]}")
            print(f"geom_size: {sys.mj_model.geom_size[geom_num]}")
            print("==========================")

            # geom_rbound is the radius of the "bounding sphere". This means the half_height
            # should just be the radius, right? If so, then what is radius?
            bs_radius = sys.mj_model.geom_rbound[geom_num]
            # length = sys.mj_model.geom_l

            # When using "fromto", only need to provide a single number for size: the
            # radius of the object. in panda.xml, 0.04 or 0.04
            # sys.mj_model.geom_size[geom_num] = [0.07 0.07 0.  ]

            # The "half_height" determines the length of the cylinder between the two
            # half-spheres. I.g., half_height=0 is just a sphere with a given radius
            # geom_size[0] is radius
            # geom_size[1] * 2 is height in THREE.CylinderGeometry
            model = create_capsule(
                radius=sys.mj_model.geom_size[geom_num][0],
                half_height=sys.mj_model.geom_size[geom_num][1],
                up_axis=UpAxis.Z,
                diffuse_map=tex,
                specular_map=specular_map,
            )
        elif geom_id == 4:
            raise NotImplementedError("Have not implemented ELLIPSOID yet.")
        elif geom_id == 5:
            raise NotImplementedError("Have no implemented CYLINDER yet.")
        elif geom_id == 6:
            model = create_cube(
                half_extents=sys.geom_size[geom_num] * 0.0,
                diffuse_map=tex,
                texture_scaling=jnp.array(16.0),
                specular_map=specular_map,
            )
        elif geom_id == 7:
            # nmeshvert: 399342
            # geom_dataid: (106,)
            # mesh_vertadr: (57,)
            # Get vertices:
            # (1) get the idx at which this particular mesh's idx begins and ends
            # (2) use (1) to query mj_model.mesh_vert[begin:end]

            # we can use "geom_dataid: id of geom's mesh/hfield" to determine the mesh idx
            # as well as if the mesh is the last mesh. geom_dataid: (n_geoms,)
            mesh_idx = sys.mj_model.geom_dataid[geom_num]
            last_mesh = (mesh_idx + 1) >= sys.mj_model.nmesh
            vert_idx_start = sys.mj_model.mesh_vertadr[mesh_idx]
            vert_idx_end = (
                sys.mj_model.mesh_vertadr[mesh_idx + 1]
                if not last_mesh
                else sys.mj_model.mesh_vert.shape[0]
            )
            # mesh_vert (399342, 3)
            vertices = sys.mesh_vert[vert_idx_start:vert_idx_end]

            # Get faces
            face_idx_start = sys.mj_model.mesh_faceadr[mesh_idx]
            face_idx_end = (
                sys.mj_model.mesh_faceadr[mesh_idx + 1]
                if not last_mesh
                else sys.mj_model.mesh_face.shape[0]
            )
            faces = sys.mj_model.mesh_face[face_idx_start:face_idx_end]

            # print(f"vertices: {vertices.shape}")
            # print(f"faces: {faces.shape}")
            # print(f"BEFORE: {tex.shape}")
            material_id = sys.mj_model.geom_matid[geom_num]
            tex = sys.mat_rgba[material_id][:3].reshape((1, 1, 3))
            # print(f"AFTER: {tex.shape}")

            # I am now beginning to think that we do not need **any** of the dataclasses...
            # We cannot jit the creation of this dataclass, which I think is slowing down
            # the runtime (bottlenecking it, effectively).
            face_normals, _triangles = compute_face_normals_and_triangles(vertices, faces)
            face_angles = compute_face_angles(_triangles)
            vertex_normals = compute_vertex_normals(
                vertices, faces, face_normals, face_angles
            )

            # tm = Trimesh.create(vertices, faces)
            # print("made the object.")
            ## We need the vertex_normals, which is computed with geometry.weighted_vertex_normals()
            ## The above fn takes vertex_count (len(trimesh.vertices)),
            ## face_normals (needs computing)
            ## face_angles (needs computing)

            ## Face normals:
            ## (a) triangles = vertices[faces]
            ## (b) triangles_cross = triangles.cross(triangles)

            # face_normals, _triangles = tm.compute_face_normals_and_triangles()
            # face_angles = tm.compute_face_angles(_triangles)
            # vertex_normals = tm.compute_vertex_normals(face_normals, face_angles)
            model = RendererMesh.create(
                verts=vertices,
                norms=vertex_normals,
                uvs=jnp.zeros((vertices.shape[0], 2), dtype=int),
                faces=faces,
                diffuse_map=tex,
            )
        else:
            raise NotImplementedError(f"Geom of ID {geom_id} not implemented nor known.")

        # Lets give this a funny shout... Trying to copy how they do it in the .js file
        # https://github.com/google/brax/blob/main/brax/visualizer/js/system.js#L223
        # x *does not include the ground plane*! This would explain the n_bodies - 1 (29, 3)
        # https://github.com/google/brax/blob/c87dcfc5094afffb149f98e48903fb39c2b7f7af/brax/mjx/pipeline.py#L75C17-L75C34

        # x.rot, x.pos = (29, 3), (29, 4)
        # print(
        #    f".x.rot: {pipeline_states.x.rot.shape} // {pipeline_states.x.rot[body_id - 1].shape}"
        # )
        # print(
        #    f".x.pos: {pipeline_states.x.pos.shape} // {pipeline_states.x.pos[body_id - 1].shape}"
        # )

        ## (106, 3), (106, 4)
        # print(f"sys.geom_pos: {sys.mj_model.geom_pos.shape}")
        # print(f"sys.geom_quat: {sys.mj_model.geom_quat.shape}")

        # rot_raw = pipeline_states.x.rot[body_id]
        # rot = jnp.array([rot_raw[1], rot_raw[2], rot_raw[3], rot_raw[0]])
        # off = pipeline_states.x.pos[body_id]
        # rot = quat_from_3x3(math.inv_3x3(pipeline_states.geom_xmat[geom_num]))

        # The groundplane's information is **not** within pipeline_states.x
        # if geom_id != 990:
        # rot = quat_from_3x3(pipeline_states.geom_xmat[geom_num])
        # off = pipeline_states.geom_xpos[geom_num]
        # copying this thing:
        # https://github.com/google/brax/blob/main/brax/io/json.py#L129
        rot = sys.geom_quat[geom_num]
        off = sys.geom_pos[geom_num]
        # print(f"rot: {rot.shape}")
        # print(f"off: {off.shape}")

        # Then there's this idea...
        # https://github.com/google/brax/blob/c87dcfc5094afffb149f98e48903fb39c2b7f7af/brax/contact.py#L43
        # else:
        #    # off = pipeline_states.geom_xpos[geom_num]
        #    # rot = math.ang_to_quat(pipeline_states.xd.ang[body_id - 1])

        #    def local_to_global(pos1, quat1, pos2, quat2):
        #        pos = pos1 + math.rotate(pos2, quat1)
        #        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        #        return pos, mat

        #    x = pipeline_states.x.concatenate(base.Transform.zero((1,)))
        #    pos, mat = local_to_global(
        #        x.pos[body_id - 1],
        #        x.rot[body_id - 1],
        #        sys.mj_model.geom_pos[geom_num],
        #        sys.mj_model.geom_quat[geom_num],
        #    )
        #    print(f"pos: {pos.shape}")
        #    print(f"3x3: {mat.shape}")

        #    off = pos
        #    rot = quat_from_3x3(mat)
        return model, rot, off


    @jax.jit
    def _build_objects_experimental(sys: brax.System, pipeline_states: brax.State) -> list[Obj]:
        """
        Converts a brax System to a list of Obj.

        Args:
          sys:

        Returns:

        """

        objs: list[Obj] = []

        def take_i(obj, i):
            return jax.tree_map(lambda x: jnp.take(x, i, axis=0), obj)

        # Loop through each geom type (sys.mj_model.geom_type) in the list of the
        # environment's geoms. Within each step in the loop, loop over each of the
        # batched envs, create the N geoms (N = num of parallel envs) in a list
        # final outer list is len of ngeom, and len of each inner list is len of
        # N.

        # print(f"dof_parentid: {sys.dof_parentid} // {sys.dof_parentid.shape}")
        #print(f"bbox: {sys.geom_size} // {sys.geom_size.shape}")
        # qqq
        # cuboid.verts
        for idx, geom_id in enumerate(sys.mj_model.geom_type):
            print(f"geom_id: {geom_id}")
            tex = sys.mj_model.geom_rgba[idx, :3].reshape((1, 1, 3))
            # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
            specular_map = jax.lax.full(tex.shape[:2], 2.0)

            # Can we use the idx from sys.geom_bodyid to query sys.body_parentid?
            # link_idx = sys.body_parentid[sys.geom_bodyid[idx] - 1]

            # copying this thing:
            # https://github.com/google/brax/blob/main/brax/io/json.py#L129
            link_idx = sys.geom_bodyid[idx] - 1

            # TODO: temporary for dev. remove when done
            if geom_id in [0, 1, 2, 3, 4, 5, 6, 7]:  # [0, 1, 2, 3]:
                model, rot, off = _vmap_build(
                    sys,
                    pipeline_states,
                    specular_map,
                    tex,
                    geom_id,
                    idx,
                    # sys.geom_bodyid[idx],
                )

                outs = [
                    (
                        Instance(model=jax.tree_map(lambda x: x[i], model)),
                        jax.tree_map(lambda x: x[i], rot),
                        jax.tree_map(lambda x: x[i], off),
                    )
                    for i in range(model.verts.shape[0])
                ]

                print(f"outs: {type(outs)} // {len(outs)}")
                outs = [
                    Obj(instance=instance, link_idx=link_idx, rot=rot, off=off)
                    for (instance, rot, off) in outs
                ]
            else:
                outs = []

            objs.extend(outs)

        return objs

    def build_objects_for_cache(sys: brax.System, n_envs: int):
        objs = _build_objects(sys)

        # we now have a list of Obj(), but they are not t"``racedarrays
        jax_objs = []
        for obj in objs:
            #print(f"bool: {obj.instance.double_sided}")
            #print(obj.instance.double_sided.shape)
            #print(len(obj.instance.double_sided.shape) > 0)
            obj = Obj(link_idx=obj.link_idx, off=obj.off, rot=obj.rot, instance=obj.instance._replace(double_sided=jnp.array(obj.instance.double_sided).reshape(1,)))
            #obj = obj.replace(instance=obj.instance._replace(double_sided=jnp.array(obj.instance.double_sided).reshape(1,)))
            #obj.instance = obj.instance._replace(double_sided = jnp.array(obj.instance.double_sided).reshape(1,))
            jax_objs.append(jax.tree_map(lambda x: jnp.array(x), obj))
            #print(f"bool: {obj.instance.double_sided}")
            #print(obj.instance.double_sided.shape)
            #qqq
            # try:
            #    print(
            #        f"{jax_objs[-1].instance.model.verts.shape} // {jax_objs[-1].instance.model.faces.shape}"
            #    )
            # except:
            #    print("This one has no verts...")
            #    print(jax_objs[-1])
            #    qqq
        
        vmappable_objs = Obj(
            #instance=jax.tree_map(lambda *x: jnp.concatenate([jnp.expand_dims(_x, 0) for _x in x], axis=0), *zip([i.instance for i in jax_objs]))[0],
            rot=jnp.concatenate([x.rot[None] for x in jax_objs], axis=0),
            off=jnp.concatenate([x.off[None] for x in jax_objs], axis=0),
            link_idx=jnp.concatenate(
                [jnp.array(x.link_idx)[None] for x in jax_objs], axis=0
            ),
        )
        #print(f"new instances: {vmappable_objs.instance.transform.shape}")
        #qqq

        return jax_objs, vmappable_objs
    
    def get_camera_experimental(
        state: brax.State,
        width: int,
        height: int,
    ) -> Camera:
        """Gets camera object."""
        eye, up = _eye(None, state), _up(None)

        hfov = HFOV  # orig: 58.0 -- higher == more zoomed out
        vfov = hfov * height / width

        # Position of the camera && target need to be updated
        # _eye determines the position
        # get_target updates the "lookat" target
        target = get_target(state)

        camera = Camera(
            viewWidth=width,
            viewHeight=height,
            position=eye,
            target=target,
            up=up,
            hfov=hfov,
            vfov=vfov,
        )

        return camera


    _get_cameras_experimental = jax.jit(
        jax.vmap(
            lambda state, width, height: get_camera_experimental(state, width, height),
            in_axes=(0, None, None),
        )
    )

    @partial(jax.jit, static_argnames="hw")
    def render_pixels_with_cached_objs(
        pipeline_states: brax.State,
        cached_objs: Iterable[Any],
        cached_vmappable_objs: Iterable[Any],
        hw: int,
    ):
        # Here, cached_objs has .instances attribute while vmappable does not
        batched_camera = _get_cameras_experimental(pipeline_states, hw, hw)
        batched_target = _get_targets(pipeline_states)
        
        images = _render_cached(
            cached_objs,
            cached_vmappable_objs,
            pipeline_states,
            batched_camera,
            batched_target,
            hw,
        )
        return images

    def render_cached(objs, vmappable_objs, states, batched_camera, batched_target, hw):
        batched_instances = _get_instances_vmap(objs, vmappable_objs, states)
        images = _inner_render_experimental(batched_instances, batched_camera, batched_target, hw)
        return images

    _render_cached = jax.jit(render_cached, static_argnames="hw")

    _get_instances_vmap = jax.jit(
        jax.vmap(
            lambda objs, vmappable_objs, state: _with_state_vmap(
                objs, vmappable_objs, state.x.concatenate(base.Transform.zero((1,)))
            ),
            in_axes=(None, None, 0),
        )
    )

    @partial(jax.vmap, in_axes=(0, None))
    def _inner_with_state_vmap(vmappable_objs: Iterable[Any], x: brax.Transform):
        pos = x.pos[vmappable_objs.link_idx] + math.rotate(
            vmappable_objs.off, x.rot[vmappable_objs.link_idx]
        )
        rot = math.quat_mul(x.rot[vmappable_objs.link_idx], vmappable_objs.rot)
        return pos, rot


    def _with_state_vmap(
        objs: Iterable[Obj], vmappable_objs: Iterable[Any], x: brax.Transform
    ) -> list[Instance]:
        """For this process, we only need positon and orientation!"""
        poss, rots = _inner_with_state_vmap(vmappable_objs, x)
        new_instances = [objs[i].instance.replace_with_position(poss[i]).replace_with_orientation(rots[i]) for i in range(poss.shape[0])]
        return new_instances



    class PixelEnvExperimental(PipelineEnv):
        def __init__(self, env):
            super().__init__(sys=env.sys, backend=env.backend)
            self.env = env
            self.seed = ret
            self._reset_fn = jax.jit(jax.vmap(env.reset))
            self._step_fn = jax.jit(jax.vmap(env.step))

            #self.cached_objects, self.vmappable_objects = build_objects_for_cache(
            #    self.env.sys, n_envs
            #)

        @property
        def action_size(self):
            return self.env.action_size

        @property
        def observation_sample(self):
            return jnp.zeros(
                (hw, hw, 9), dtype=jnp.float32 if return_float32 else jnp.uint8
            )

        @property
        def observation_size(self):
            return self.env.observation_size

        @property
        def max_episode_steps(self):
            return 1000

        def reset(self, rng: jax.Array):
            raw_state = self._reset_fn(rng)

            # This is only used for the video distractors. This API design is kinda gross, but oh well...
            video_idx = jax.random.choice(
                rng[0], jnp.arange(start=0, stop=len(BG_FRAMES)), shape=(n_envs,)
            )

            #frames = render_pixels_with_cached_objs(
            #    raw_state.pipeline_state,
            #    self.cached_objects,
            #    self.vmappable_objects,
            #    hw,
            #)
            frames = jax.vmap(image.render_array, in_axes=(None, 0, None, None))(self.sys, raw_state.pipeline_state, hw, hw)
            print("here...")
            qqq

            if distractor == "colors":
                # When doing color distractions, we want a singular value for each color channel [R,G,B]. However,
                # we also want a different initial color for each of the environments
                if return_float32:
                    noise = jax.random.uniform(
                        key=rng[0], shape=(n_envs, 1, 1, 3), minval=-0.3, maxval=0.3
                    )
                    _frames = jnp.clip(frames + noise, a_min=0.0, a_max=1.0)
                else:
                    noise = jax.random.uniform(
                        key=rng[0],
                        shape=(n_envs, 1, 1, 3),
                        minval=-0.3 * 255,
                        maxval=0.3 * 255,
                    ).astype(jnp.int8)
                    _frames = jnp.clip(frames + noise, a_min=0, a_max=255).astype(
                        jnp.uint8
                    )

                video_idx = jnp.zeros(shape=(n_envs,), dtype=jnp.int8)

            elif distractor == "videos":
                if return_float32:
                    frames = alpha * frames + (1 - alpha) * (
                        BG_FRAMES[video_idx][:, 0] / 255.0
                    )
                else:
                    frames = (
                        alpha * frames + (1 - alpha) * BG_FRAMES[video_idx][:, 0]
                    ).astype(jnp.uint8)

            else:
                video_idx = jnp.zeros(shape=(n_envs,), dtype=jnp.int8)

            _frames = jnp.concatenate([frames, frames, frames], axis=-1)

            return State(
                raw_state.pipeline_state,
                raw_state.obs,
                _frames,
                raw_state.reward,
                raw_state.done,
                rng,
                jnp.zeros(shape=(n_envs,), dtype=jnp.int8),
                video_idx,
                raw_state.metrics,
                raw_state.info,
            )

        def step(self, states, actions):
            raw_next_states = self._step_fn(states, actions)
            frame_idx = states.frame_idx

            next_frames = render_pixels_with_cached_objs(
                raw_next_states.pipeline_state,
                self.cached_objects,
                self.vmappable_objects,
                hw,
            )

            if not return_float32:
                next_frames = (next_frames * 255).astype(jnp.uint8)

            if distractor == "colors":
                key = jax.vmap(jax.random.split)(states.key)[:, 0]
                # noise = jax.random.uniform(key=key[0], shape=(n_envs, 1, 1, 3), minval=-0.3, maxval=0.3)
                if return_float32:
                    noise = jax.random.normal(key=key[0], shape=(n_envs, 1, 1, 3)) * 0.3
                    next_frames = jnp.clip(next_frames + noise, a_min=0.0, a_max=1.0)
                else:
                    noise = (
                        jax.random.normal(key=key[0], shape=(n_envs, 1, 1, 3))
                        * 0.3
                        * 255
                    ).astype(jnp.int8)
                    next_frames = jnp.clip(
                        next_frames + noise, a_min=0, a_max=255
                    ).astype(jnp.uint8)

            elif distractor == "videos":
                frame_idx = (states.frame_idx + 1) % VIDEO_LEN
                if return_float32:
                    next_frames = alpha * next_frames + (1 - alpha) * (
                        BG_FRAMES[states.video_idx, frame_idx] / 255.0
                    )
                else:
                    next_frames = (
                        alpha * next_frames
                        + (1 - alpha) * BG_FRAMES[states.video_idx, frame_idx]
                    ).astype(jnp.uint8)
                key = states.key
            else:
                key = states.key

            next_frames = jnp.concatenate(
                [states.pixels[:, :, :, 3:], next_frames], axis=-1
            )

            return states.replace(
                pipeline_state=raw_next_states.pipeline_state,
                obs=raw_next_states.obs,
                reward=raw_next_states.reward,
                done=raw_next_states.done,
                pixels=next_frames,
                info=raw_next_states.info,
                key=key,
                frame_idx=frame_idx,
            )


    class PixelEnv(PipelineEnv):
        def __init__(self, env):
            super().__init__(sys=env.sys, backend=env.backend)
            self.env = env
            self.seed = ret
            self._reset_fn = jax.jit(jax.vmap(env.reset))
            self._step_fn = jax.jit(jax.vmap(env.step))
            # self._frames = deque([jnp.zeros(shape=(n_envs, hw, hw, 3)) for _ in range(3)], maxlen=3)

        @property
        def action_size(self):
            return self.env.action_size

        @property
        def observation_sample(self):
            return jnp.zeros(
                (hw, hw, 9), dtype=jnp.float32 if return_float32 else jnp.uint8
            )

        @property
        def observation_size(self):
            return self.env.observation_size

        @property
        def max_episode_steps(self):
            return 1000

        def reset(self, rng: jax.Array):
            raw_state = self._reset_fn(rng)

            # This is only used for the video distractors. This API design is kinda gross, but oh well...
            video_idx = jax.random.choice(
                rng[0], jnp.arange(start=0, stop=len(BG_FRAMES)), shape=(n_envs,)
            )

            frames = render_pixels(
                self.env.sys.replace(dt=self.env.dt), raw_state.pipeline_state
            )
            if not return_float32:
                frames = (frames * 255).astype(jnp.uint8)

            if distractor == "colors":
                # When doing color distractions, we want a singular value for each color channel [R,G,B]. However,
                # we also want a different initial color for each of the environments
                if return_float32:
                    noise = jax.random.uniform(
                        key=rng[0], shape=(n_envs, 1, 1, 3), minval=-0.3, maxval=0.3
                    )
                    frames = jnp.clip(frames + noise, a_min=0.0, a_max=1.0)
                else:
                    noise = jax.random.uniform(
                        key=rng[0],
                        shape=(n_envs, 1, 1, 3),
                        minval=-0.3 * 255,
                        maxval=0.3 * 255,
                    ).astype(jnp.int8)
                    frames = jnp.clip(frames + noise, a_min=0, a_max=255).astype(
                        jnp.uint8
                    )

                video_idx = jnp.zeros(shape=(n_envs,), dtype=jnp.int8)

            elif distractor == "videos":
                if return_float32:
                    frames = alpha * frames + (1 - alpha) * (
                        BG_FRAMES[video_idx][:, 0] / 255.0
                    )
                else:
                    frames = (
                        alpha * frames + (1 - alpha) * BG_FRAMES[video_idx][:, 0]
                    ).astype(jnp.uint8)

            else:
                video_idx = jnp.zeros(shape=(n_envs,), dtype=jnp.int8)

            _frames = jnp.concatenate([frames, frames, frames], axis=-1)
            #_frames = frames

            return State(
                raw_state.pipeline_state,
                raw_state.obs,
                _frames,
                raw_state.reward,
                raw_state.done,
                rng,
                jnp.zeros(shape=(n_envs,), dtype=jnp.int8),
                video_idx,
                raw_state.metrics,
                raw_state.info,
            )

        def step(self, states, actions):
            # Annoyingly, we need to replace the pixels obs in the previous state to the env's actual obs...
            raw_next_states = self._step_fn(states, actions)

            frame_idx = states.frame_idx

            next_frames = render_pixels(
                self.env.sys.replace(dt=self.env.dt), raw_next_states.pipeline_state
            )
            if not return_float32:
                next_frames = (next_frames * 255).astype(jnp.uint8)

            if distractor == "colors":
                key = jax.vmap(jax.random.split)(states.key)[:, 0]
                # noise = jax.random.uniform(key=key[0], shape=(n_envs, 1, 1, 3), minval=-0.3, maxval=0.3)
                if return_float32:
                    noise = jax.random.normal(key=key[0], shape=(n_envs, 1, 1, 3)) * 0.3
                    next_frames = jnp.clip(next_frames + noise, a_min=0.0, a_max=1.0)
                else:
                    noise = (
                        jax.random.normal(key=key[0], shape=(n_envs, 1, 1, 3))
                        * 0.3
                        * 255
                    ).astype(jnp.int8)
                    next_frames = jnp.clip(
                        next_frames + noise, a_min=0, a_max=255
                    ).astype(jnp.uint8)

            elif distractor == "videos":
                frame_idx = (states.frame_idx + 1) % VIDEO_LEN
                if return_float32:
                    next_frames = alpha * next_frames + (1 - alpha) * (
                        BG_FRAMES[states.video_idx, frame_idx] / 255.0
                    )
                else:
                    next_frames = (
                        alpha * next_frames
                        + (1 - alpha) * BG_FRAMES[states.video_idx, frame_idx]
                    ).astype(jnp.uint8)
                key = states.key
            else:
                key = states.key

            next_frames = jnp.concatenate(
                [states.pixels[:, :, :, 3:], next_frames], axis=-1
            )


            return states.replace(
                pipeline_state=raw_next_states.pipeline_state,
                obs=raw_next_states.obs,
                reward=raw_next_states.reward,
                done=raw_next_states.done,
                pixels=next_frames,
                info=raw_next_states.info,
                key=key,
                frame_idx=frame_idx,
            )
    
    if not experimental:
        return PixelEnv(env), PixelEnv(env).reset(ret), ret
    else:
        return PixelEnvExperimental(env), PixelEnvExperimental(env).reset(ret), ret


# adapted from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
@flax.struct.dataclass
class RunningMeanStd:
    mean: np.ndarray
    var: np.ndarray
    count: float
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    @staticmethod
    def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    @classmethod
    def create(cls, x, **kwargs):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        mean, var, count = cls.update_mean_var_count_from_moments(
            kwargs["mean"],
            kwargs["var"],
            kwargs["count"],
            batch_mean,
            batch_var,
            x.shape[0],
        )
        return cls(mean, var, count)


@flax.struct.dataclass
class RewardWrapper:
    normalize: bool
    clip: bool
    clip_coef: float
    discounted_returns: np.ndarray
    return_rms: RunningMeanStd
    gamma: float

    @classmethod
    def create(cls, rewards, next_terminated, next_truncated, **kwargs):
        discounted_returns = kwargs["discounted_returns"] * kwargs["gamma"] + rewards
        kwargs["return_rms"] = kwargs["return_rms"].create(
            discounted_returns, **kwargs["return_rms"].__dict__
        )
        kwargs["discounted_returns"] = np.where(
            next_terminated + next_truncated, 0, discounted_returns
        )
        return cls(**kwargs)

    def process_rewards(self, rewards, next_terminated, next_truncated):
        newcls = self.create(rewards, next_terminated, next_truncated, **self.__dict__)
        if self.normalize:
            rewards = self.normalize_rewards(rewards, newcls.return_rms)
        if self.clip:
            rewards = self.clip_rewards(rewards, self.clip_coef)
        return newcls, rewards

    @staticmethod
    def normalize_rewards(rewards, return_rms, epsilon=1e-8):
        return rewards / np.sqrt(return_rms.var + epsilon)

    @staticmethod
    def clip_rewards(rewards, clip_coef):
        return np.clip(rewards, -clip_coef, clip_coef)
