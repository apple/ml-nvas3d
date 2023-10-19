#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
# Portions of this code are derived from SoundSpaces (CC-BY-4.0).
# Original work available at: https://github.com/facebookresearch/sound-spaces
#

import os
import sys
import math
import random
import magnum as mn
import typing as T
import matplotlib.pyplot as plt
from contextlib import contextmanager

import numpy as np
import torch
import torchaudio

import habitat_sim.sim
from habitat_sim.utils import quat_from_angle_axis

from soundspaces_nvas3d.utils import audio_utils, aihabitat_utils


@contextmanager
def suppress_stdout_and_stderr():
    """
    To suppress the logs from SoundSpaces
    """

    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        os.close(devnull)


class Receiver:
    """
    Receiver for SoundSpaces
    """

    def __init__(self,
                 position: T.Tuple[float, float, float],
                 rotation: float,
                 sample_rate: float = 48000,
                 ):

        self.position = position
        self.rotation = rotation
        self.sample_rate = sample_rate


class Source:
    """
    Source for Soundspaces
    """

    def __init__(self,
                 position: T.Tuple[float, float, float],
                 rotation: float,
                 dry_sound: str,
                 mesh: str,
                 device: torch.device
                 ):

        self.position = position
        self.rotation = rotation
        self.device = device  # where to store dry_sound
        self.dry_sound = dry_sound
        self.mesh = mesh


class Scene:
    """
    Soundspaces scene including room, receiver, and source list
    """

    def __init__(self,
                 room: str,
                 source_name_list: T.List[str],
                 receiver: Receiver = None,
                 source_list: T.List[Source] = None,
                 include_visual_sensor: bool = True,
                 add_source_mesh: bool = True,
                 device: torch.device = torch.device('cpu'),
                 add_source: bool = True,
                 image_size: T.Tuple[int, int] = (512, 256),
                 hfov: float = 90.0,
                 use_default_material: bool = False,
                 channel_type: str = 'Ambisonics',
                 channel_order: int = 1
                 ):

        # Set scene
        self.room = room
        self.n_sources = len(source_name_list)
        assert self.n_sources > 0
        self.receiver = receiver
        self.source_list = source_list
        self.source_current = None
        self.include_visual_sensor = include_visual_sensor
        self.add_source_mesh = add_source_mesh
        self.device = device  # where to store IR

        # Set channel config for soundspaces
        self.channel = {}
        self.channel['type'] = channel_type
        self.channel['order'] = channel_order
        if channel_type == 'Ambisonics':
            self.channel_count = (self.channel['order'] + 1)**2
        elif channel_type == 'Binaural':
            self.channel_count = 2

        # Set aihabitat config for soundspaces
        self.aihabitat = {}
        self.aihabitat['default_agent'] = 0
        self.aihabitat['sensor_height'] = 1.5
        self.aihabitat['height'] = image_size[0]
        self.aihabitat['width'] = image_size[1]
        self.aihabitat['hfov'] = hfov

        # Set acoustics config for soundspaces
        self.acoustic_config = {}
        self.acoustic_config['sampleRate'] = 48000
        self.acoustic_config['direct'] = True
        self.acoustic_config['indirect'] = True
        self.acoustic_config['diffraction'] = True
        self.acoustic_config['transmission'] = True
        self.acoustic_config['directSHOrder'] = 5
        self.acoustic_config['indirectSHOrder'] = 3
        self.acoustic_config['unitScale'] = 1
        self.acoustic_config['frequencyBands'] = 32
        self.acoustic_config['indirectRayCount'] = 50000

        # Set audio material
        if use_default_material:
            self.audio_material = './data/material/mp3d_material_config_default.json'
        else:
            self.audio_material = './data/material/mp3d_material_config.json'

        # Create simulation
        self.create_scene()

        # Randomly set source and receiver position
        source_position, source_rotation = None, None
        receiver_position, receiver_rotation = None, None

        # Create receiver (inside the room)
        if self.receiver is None:
            # random receiver
            self.create_receiver(receiver_position, receiver_rotation)
        else:
            # input receiver
            self.update_receiver(self.receiver)

        if add_source:
            # Create source
            if self.source_list is None:
                # random source
                self.source_list = [None] * self.n_sources
                for source_id, source_name in enumerate(source_name_list):
                    self.create_source(source_name, source_id, source_position, source_rotation)
            else:
                # input source
                for source_id, _ in enumerate(source_name_list):
                    self.update_source(self.source_list[source_id], source_id)

    def create_scene(self):
        """
        Given the configuration, create a scene for soundspaces
        """

        # Set backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = f'./data/scene_datasets/mp3d/{self.room}/{self.room}.glb'
        backend_cfg.scene_dataset_config_file = './data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        backend_cfg.load_semantic_mesh = True
        backend_cfg.enable_physics = False

        # Set agent configuration
        agent_config = habitat_sim.AgentConfiguration()

        if self.include_visual_sensor:
            # Set color sensor
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color_sensor"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = [self.aihabitat['height'], self.aihabitat['width']]
            rgb_sensor_spec.position = [0.0, self.aihabitat["sensor_height"], 0.0]
            rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            rgb_sensor_spec.hfov = self.aihabitat["hfov"]
            agent_config.sensor_specifications = [rgb_sensor_spec]

            # Set depth sensor
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.aihabitat["height"], self.aihabitat["width"]]
            depth_sensor_spec.position = [0.0, self.aihabitat["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth_sensor_spec.hfov = self.aihabitat["hfov"]
            agent_config.sensor_specifications.append(depth_sensor_spec)

            # # Set semantic sensor
            # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            # semantic_sensor_spec.uuid = "semantic_sensor"
            # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            # semantic_sensor_spec.resolution = [self.aihabitat["height"], self.aihabitat["width"]]
            # semantic_sensor_spec.position = [0.0, self.aihabitat["sensor_height"], 0.0]
            # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            # semantic_sensor_spec.hfov = self.aihabitat["hfov"]
            # agent_config.sensor_specifications.append(semantic_sensor_spec)

        # Set simulator configuration
        cfg = habitat_sim.Configuration(backend_cfg, [agent_config])

        # Set simulator
        sim = habitat_sim.Simulator(cfg)

        # set navmesh path for searching for navigatable points
        navmesh = f'./data/scene_datasets/mp3d/{self.room}/{self.room}.navmesh'
        sim.pathfinder.load_nav_mesh(navmesh)

        # seed for navmesh
        sim.seed(random.randint(0, 1024))

        # Set simulation
        self.sim = sim
        print('Scene created!')

        return self

    import torch

    def add_audio_sensor(self):
        """
        Add audio sensor to the scene
        """

        # set audio sensor
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        audio_sensor_spec.uuid = "audio_sensor"
        audio_sensor_spec.enableMaterials = True  # make sure _semantic.ply file is in the scene folder
        audio_sensor_spec.channelLayout.type = getattr(habitat_sim.sensor.RLRAudioPropagationChannelLayoutType, self.channel['type'])
        audio_sensor_spec.channelLayout.channelCount = self.channel_count  # ambisonics

        # Set acoustic configuration
        audio_sensor_spec.acousticsConfig.sampleRate = self.acoustic_config['sampleRate']
        audio_sensor_spec.acousticsConfig.direct = self.acoustic_config['direct']
        audio_sensor_spec.acousticsConfig.indirect = self.acoustic_config['indirect']
        audio_sensor_spec.acousticsConfig.diffraction = self.acoustic_config['diffraction']
        audio_sensor_spec.acousticsConfig.transmission = self.acoustic_config['transmission']
        audio_sensor_spec.acousticsConfig.directSHOrder = self.acoustic_config['directSHOrder']
        audio_sensor_spec.acousticsConfig.indirectSHOrder = self.acoustic_config['indirectSHOrder']
        audio_sensor_spec.acousticsConfig.unitScale = self.acoustic_config['unitScale']
        audio_sensor_spec.acousticsConfig.frequencyBands = self.acoustic_config['frequencyBands']
        audio_sensor_spec.acousticsConfig.indirectRayCount = self.acoustic_config['indirectRayCount']
        # audio_sensor_spec.acousticsConfig.maxIRLength = 40.0
        # audio_sensor_spec.acousticsConfig.sourceRayCount = 2000
        # audio_sensor_spec.acousticsConfig.meshSimplification = False

        # Initialize receiver
        audio_sensor_spec.position = [0.0, self.aihabitat['sensor_height'], 0.0]  # audio sensor has a height of 1.5m
        self.sim.add_sensor(audio_sensor_spec)

        audio_sensor = self.sim.get_agent(self.aihabitat['default_agent'])._sensors['audio_sensor']
        audio_sensor.setAudioMaterialsJSON(self.audio_material)

        return self

    def create_receiver(self,
                        position: T.Tuple[float, float, float] = None,
                        rotation: float = None
                        ):
        """
        Randomly sample receiver position and rotation
        """

        if position is None:
            # Randomly set receiver position in the room
            position = self.sim.pathfinder.get_random_navigable_point()
            rotation = random.uniform(0, 360)

        # Set sample rate
        sample_rate = self.acoustic_config['sampleRate']

        # Set receiver
        receiver = Receiver(position, rotation, sample_rate)

        # Update receiver
        self.update_receiver(receiver)

        return self

    def update_receiver(self,
                        receiver: Receiver
                        ):
        """
        Update receiver
        """

        agent = self.sim.get_agent(self.aihabitat["default_agent"])
        new_state = self.sim.get_agent(self.aihabitat["default_agent"]).get_state()
        new_state.position = np.array(receiver.position + np.array([0, 0.0, 0]))  # agent height is already applied in audio_sensor_spec.position
        new_state.rotation = quat_from_angle_axis(math.radians(receiver.rotation), np.array([0, 1.0, 0]))  # + -> left
        # new_state.rotation *= quat_from_angle_axis(math.radians(-30), np.array([1.0, 0, 0]))  # + -> up
        new_state.sensor_states = {}
        agent.set_state(new_state, True)

        self.receiver = receiver  # for reference

        return self

    def update_receiver_position(self,
                                 receiver_position: T.Tuple[float, float, float]
                                 ):
        """
        Update receiver position
        """

        self.receiver.position = receiver_position

        agent = self.sim.get_agent(self.aihabitat["default_agent"])
        new_state = self.sim.get_agent(self.aihabitat["default_agent"]).get_state()
        new_state.position = np.array(receiver_position + np.array([0, 0.0, 0]))  # agent height is already applied in audio_sensor_spec.position
        new_state.sensor_states = {}
        agent.set_state(new_state, True)

        return self

    def create_source(self,
                      source_name: str,
                      source_id: int,
                      position: T.Tuple[float, float, float] = None,
                      rotation: float = None
                      ):
        """
        Set source given the source name, position, and rotation
        """

        if position is None:
            # Randomly set source position in the room
            position = self.sim.pathfinder.get_random_navigable_point()
            rotation = random.uniform(0, 360)  # only for mesh as source sound is omnidirectional

        # Randomly set source sound
        dry_sound, mesh = sample_dry_sound_and_mesh(source_name)

        # Set source
        source = Source(position, rotation, dry_sound, mesh, device=self.device)

        # Save source
        self.update_source(source, source_id)

        return self

    def update_source(self,
                      source: Source,
                      source_id: int = None
                      ):
        """
        Update source
        """

        if source_id is not None:
            # update source list
            self.source_list[source_id] = source

            # Add mesh
            if self.add_source_mesh:
                ########## Add mesh (source.position, source.rotation) ##########
                obj_templates_mgr = self.sim.get_object_template_manager()
                rigid_obj_mgr = self.sim.get_rigid_object_manager()

                # Load the object template from the configuration file
                obj_templates_mgr.load_configs(str(os.path.join("data/objects")))

                # Insert the object relative to the agent
                object_ids = []
                object_orientation = mn.Quaternion.rotation(mn.Deg(source.rotation), mn.Vector3.y_axis())
                object_template_handle = obj_templates_mgr.get_template_handles(f'data/objects/{source.mesh}')[0]  # debug
                if source.mesh == 'male':
                    scale = 0.5
                    height_offset = 0.935
                elif source.mesh == 'female':
                    scale = 1.0
                    height_offset = 0.85
                elif source.mesh == 'guitar':
                    scale = 1 / 1239.1628 * 2
                    height_offset = 1.5
                    object_orientation *= mn.Quaternion.rotation(mn.Deg(-90), mn.Vector3.x_axis())
                elif source.mesh == 'drum':
                    scale = 1 / 1.8
                    height_offset = 0.6
                elif source.mesh == 'classic_microphone':
                    scale = 1 / 1.15
                    height_offset = 0.67
                elif source.mesh == 'bluetooth_speaker':
                    scale = 1 / 70
                    height_offset = 1.0

                # Scale the object to fit the scene
                scaled_object_template = obj_templates_mgr.get_template_by_handle(object_template_handle)
                scaled_object_template.scale = np.array([scale, scale, scale])
                obj_templates_mgr.register_template(scaled_object_template, "scaled")
                object = rigid_obj_mgr.add_object_by_template_handle("scaled")
                object.translation = np.array(source.position) + np.array([0, height_offset, 0])
                object.rotation = object_orientation

                object_ids.append(object.object_id)

                # rigid_obj_mgr.remove_all_objects()

        else:
            # update current source
            audio_sensor = self.sim.get_agent(self.aihabitat['default_agent'])._sensors['audio_sensor']
            audio_sensor.setAudioSourceTransform(source.position + np.array([0, self.aihabitat["sensor_height"], 0]))  # add 1.5m to the height calculation

            self.source_current = source  # for reference

        return self

    def update_source_position(self,
                               source_position
                               ):
        """
        Update Source position
        """

        audio_sensor = self.sim.get_agent(self.aihabitat['default_agent'])._sensors['audio_sensor']
        audio_sensor.setAudioSourceTransform(source_position + np.array([0, self.aihabitat["sensor_height"], 0]))  # add 1.5m to the height calculation

    def render_ir(self,
                  source_id: int
                  ) -> torch.Tensor:
        """
        Render IR given the source ID
        """

        source = self.source_list[source_id]
        self.update_source(source)
        ir = torch.tensor(self.sim.get_sensor_observations()['audio_sensor'], device=self.device)

        return ir

    def render_ir_simple(self,
                         source_position: T.Tuple[float, float, float],
                         receiver_position: T.Tuple[float, float, float],
                         ) -> torch.Tensor:
        """
        Render IR given the source ID
        """

        # source
        self.update_source_position(source_position)

        # receiver
        self.update_receiver_position(receiver_position)

        # render ir
        ir = torch.tensor(self.sim.get_sensor_observations()['audio_sensor'], device=self.device)

        return ir

    def render_ir_all(self) -> T.List[torch.Tensor]:
        """
        Render IR for all sources
        """

        ir_list = []
        for source_id in range(self.n_sources):
            print(f'Rendering IR {source_id}/{self.n_sources}...')
            ir = self.render_ir(source_id)
            ir_list.append(ir)

        return ir_list

    def render_image(self,
                     is_instance=False
                     ):
        """
        Render image including rgb, depth, and semantic
        """

        observation = self.sim.get_sensor_observations()
        rgb = observation["color_sensor"]
        depth = observation["depth_sensor"]

        # Semantic
        # semantic = sim.get_sensor_observations()["semantic_sensor"]
        # is_valid = (depth != 0)
        # semantic[~is_valid] = semantic.max() + 1

        # if is_instance:
        #     # Display instance id
        #     aihabitat_utils.display_sample(rgb, semantic, depth, filename=f'{dir_results}/view/view_instance.png')
        # else:
        #     # Display category id
        #     category = aihabitat_utils.semantic_id_to_category_id(semantic, sim.semantic_scene.objects)
        #     void_id = 0
        #     category[~is_valid] = void_id
        #     aihabitat_utils.display_sample(rgb, category, depth, filename=f'{dir_results}/view/view_category.png')

        return rgb, depth

    def render_envmap(self):
        """
        Render environment map in *** format
        """

        with suppress_stdout_and_stderr():
            angles = [0, 270, 180, 90]
            rgb_panorama = []
            depth_panorama = []

            for angle_offset in angles:
                angle = self.receiver.rotation + angle_offset
                agent = self.sim.get_agent(self.aihabitat["default_agent"])
                new_state = self.sim.get_agent(self.aihabitat["default_agent"]).get_state()
                new_state.rotation = quat_from_angle_axis(
                    math.radians(angle), np.array([0, 1.0, 0])
                ) * quat_from_angle_axis(math.radians(0), np.array([1.0, 0, 0]))
                new_state.sensor_states = {}
                agent.set_state(new_state, True)

                observation = self.sim.get_sensor_observations()
                rgb_panorama.append(observation["color_sensor"])
                depth_panorama.append((observation['depth_sensor']))
            envmap_rgb = np.concatenate(rgb_panorama, axis=1)
            envmap_depth = np.concatenate(depth_panorama, axis=1)

            # rotate receiver to original angle
            self.update_receiver(self.receiver)

        return envmap_rgb, envmap_depth

    def generate_xy_grid_points(self,
                                grid_distance: float,
                                height: float = None,
                                filename_png: str = None,
                                meters_per_pixel: float = 0.005
                                ) -> torch.Tensor:
        """
        Generate the 3D positions of grid points at the given height
        """

        pathfinder = self.sim.pathfinder
        assert pathfinder.is_loaded
        # agent_height = pathfinder.nav_mesh_settings.agent_height  # to be navigable, full body of the agent should be inside
        if height is None:  # height of the agent foot
            height = 0
            # height = pathfinder.get_bounds()[0][1]  # floor height

        # Sample grid
        bounds = pathfinder.get_bounds()
        x_points = torch.arange(bounds[0][0], bounds[1][0] + grid_distance, grid_distance)
        z_points = torch.arange(bounds[0][2], bounds[1][2] + grid_distance, grid_distance)
        x_grid, z_grid = torch.meshgrid(x_points, z_points)
        y_value = height * torch.ones_like(x_grid.reshape(-1))

        # Combine x, y, and z coordinates into a single tensor of points
        points = torch.stack([x_grid.reshape(-1), y_value.reshape(-1), z_grid.reshape(-1)], dim=-1)
        is_points_navigable = []
        for point in points:
            is_points_navigable.append(pathfinder.is_navigable(point))  # navigable points
        torch.tensor(is_points_navigable).sum()

        # Flatten the tensor of points into a list
        grid_points = points[is_points_navigable]

        # assert len(grid_points) > 0
        # save image
        if filename_png is not None:
            aihabitat_utils.save_town_map_grid(filename_png, pathfinder, grid_points, meters_per_pixel=meters_per_pixel)

        return grid_points

    def generate_data(self, use_dry_sound: bool = False):
        """
        Generate all data including IR, envmap, audio, image
        """

        # env map
        if self.include_visual_sensor:
            envmap_rgb, envmap_depth = self.render_image()
        else:
            envmap_rgb, envmap_depth = None, None

        # IR
        self.add_audio_sensor()  # add audio_sensor after image rendering for faster image rendering
        ir_list = self.render_ir_all()
        # ir_total = sum_arrays_with_different_length(ir_list).detach().cpu()

        # audio_list
        dry_sound_list = []
        audio_list = []
        # audio_total = None
        if use_dry_sound:
            for source_id, source in enumerate(self.source_list):
                # load dry sound
                dry_sound = source.dry_sound
                if isinstance(dry_sound, str):
                    dry_sound, sample_rate = torchaudio.load(dry_sound)
                    self.dry_sound = dry_sound.to(self.device)
                    self.sample_rate = sample_rate

                ir = ir_list[source_id]
                audio = torch.stack([audio_utils.fft_conv(dry_sound[0], ir_channel, is_cpu=True) for ir_channel in ir])
                dry_sound_list.append(dry_sound.detach().cpu())
                audio_list.append(audio.detach().cpu())

            # audio_total
            # audio_total = sum_arrays_with_different_length(audio_list)

        # cpu
        ir_list = [tensor.detach().cpu() for tensor in ir_list]

        # dirname = '.'
        # with open(f'{dirname}/debug.txt', 'w') as f:
        #     f.write(f'NavMesh area: {self.sim.pathfinder.navigable_area}\n')
        #     f.write(f'NavMesh bounds: {self.sim.pathfinder.get_bounds()}\n')
        #     f.write(f'Receiver position: {self.receiver.position}\n')
        #     for s, source in enumerate(self.source_list):
        #         f.write(f'Source {s} position: {source.position}\n')
        #     f.write(f'\n')

        return dict(
            ir_list=ir_list,
            sample_rate=self.receiver.sample_rate,
            envmap=[envmap_rgb, envmap_depth],
            audio_list=audio_list,
            dry_sound_list=dry_sound_list,
        )


def sample_dry_sound_and_mesh(source_name: str):
    """
    Given the source class, sample dry sound and mesh accordingly
    """

    dry_sound = 'data/source/singing.wav'
    mesh = None  # TODO

    return dry_sound, mesh


def sum_arrays_with_different_length(arrays: T.List[torch.Tensor]) -> torch.Tensor:
    """
    Sum up the elements of a list of PyTorch tensors with different lengths along the second dimension.

    Args:
        arrays: A list of PyTorch tensors with shape (ch, num_columns).

    Returns:
        A PyTorch tensor of shape (ch, max_columns) containing the summed elements of the input tensors.
    """

    if len(arrays) == 0:
        return None

    # Get the maximum number of columns in the arrays
    max_cols = max([array.shape[1] for array in arrays])

    # Pad the arrays with zeros along the second dimension
    padded_arrays = [torch.nn.functional.pad(array, pad=(0, max_cols - array.shape[1]), mode='constant', value=0) for array in arrays]

    # Sum up the padded arrays
    summed_array = torch.sum(torch.stack(padded_arrays), dim=0)

    return summed_array


def save_data(data: dict,
              dirname: str,
              use_dry_sound: bool,
              save_files: bool = False
              ):

    # Save dictionary
    torch.save(data, f'{dirname}/data.pt')

    if save_files:
        # Save each file
        sample_rate = data['sample_rate']
        os.makedirs(f'{dirname}/files')

        # IR
        # audio_utils.save_audio(f'{dirname}/files/ir_total.flac', data['ir_total'], sample_rate)
        # audio_utils.plot_waveform(f'{dirname}/files/ir_total.png', data['ir_total'], sample_rate)

        for s, ir in enumerate(data['ir_list']):
            audio_utils.save_audio(f'{dirname}/files/ir_{s}.flac', ir, sample_rate)
            # audio_utils.plot_waveform(f'{dirname}/files/ir_{s}.png', ir, sample_rate)
            audio_utils.print_stats(ir)

        # envmap
        if data['envmap'][0] is not None:
            plt.imsave(f'{dirname}/files/envmap_rgb.png', data['envmap'][0])
            plt.imsave(f'{dirname}/files/envmap_depth.png', data['envmap'][1])

        if use_dry_sound:
            # dry sound
            for s, dry_sound in enumerate(data['dry_sound_list']):
                audio_utils.save_audio(f'{dirname}/files/dry_sound_{s}.flac', dry_sound, sample_rate)
                # audio_utils.plot_waveform(f'{dirname}/files/dry_sound_{s}.png', dry_sound, sample_rate)

            # audio
            # audio_utils.save_audio(f'{dirname}/files/audio_total.flac', data['audio_total'], sample_rate)
            # audio_utils.plot_waveform(f'{dirname}/files/audio_total.png', data['audio_total'], sample_rate)
            for s, audio in enumerate(data['audio_list']):
                audio_utils.save_audio(f'{dirname}/files/audio_{s}.flac', audio, sample_rate)
                # audio_utils.plot_waveform(f'{dirname}/files/audio_{s}.png', audio, sample_rate)


if __name__ == "__main__":
    print("Import successful!")
