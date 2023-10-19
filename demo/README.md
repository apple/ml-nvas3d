# Novel-View Acoustic Synthesis from 3D Reconstructed Rooms: Demo

This guide is for the demonstration for NVAS from 3D reconstructed rooms. It provides instructions for data generation, dry sound estimation using our model, and novel-view acoustic rendering.

## Quick Start

### Download the Pretrained Model
Download [our pretrained model](https://docs-assets.developer.apple.com/ml-research/models/nvas/checkpoint_200.pt) and place it in the `nvas3d/assets/saved_models/default/checkpoints/` directory.
To get started with the full pipeline quickly:
```bash
bash demo/run_demo.sh
```

### Launch the Demo
To get started with the full pipeline quickly:
```bash
bash demo/run_demo.sh
```

## Detailed Workflow
For a more detailed approach, you can explore each segment of the workflow individually:

### 1. Data Generation
Generate and save the demo data specific to a room. Please ensure you've installed SoundSpaces and downloaded the necessary data (sample Matterport3D room of 17DRP5sb8fy and material config) by following the instructions in [soundspaces_nvas3d/README.md](../soundspaces_nvas3d/README.md).

```bash
python demo/generate_demo_data.py --room 17DRP5sb8fy
```

The generated data will be structured as:
```yaml
data/nvas3d_demo/demo/{room}/0
├── receiver/          # Receiver audio.
├── wiener/            # Deconvolved audio to accelerate tests.
├── source/            # Ground truth dry audio.
├── reverb1/           # Ground truth reverberant audio for source 1.
├── reverb2/           # Ground truth reverberant audio for source 2.
├── ir_receiver/       # Ground truth RIRs from source to receiver.
├── config.png         # Visualization of room configuration.
└── metadata.pt        # Metadata (source indices, classes, grid points, and room info).
```

Additionally, visualizations of room indices will be located at `data/nvas3d_demo/{room}/index_{grid_distance}.png`. If not, please ensure that [headless rendering in Habitat-Sim](https://github.com/facebookresearch/habitat-sim/issues?q=headless) is configured correctly.

> [!Note]
> If you want to modify the scene configuration (e.g.,, source locations, receiver locations), edit `demo/nvas/config_demo/scene_config.json`.

### 2. Dry Sound Estimation Using Our Model
Run the NVAS3D model on the your dataset:
```bash
python demo/test_demo.py
```

The results will be structured as:
```yaml
{results_demo} = results/nvas3d_demo/default/demo/{room}/0
│
├── results_drysound/
│   ├── dry1_estimated.wav      # Estimated dry sound for source 1 location.
│   ├── dry2_estimated.wav      # Estimated dry sound for source 2 location.
│   ├── dry1_gt.wav             # Ground-truth dry sound for source 1.
│   ├── dry2_gt.wav             # Ground-truth dry sound for source 2.
│   └── detected/
│       └── dry_{query_idx}.wav # Estimated dry sound for query idx if positive.
│
├── baseline_dsp/
│   ├── deconv_and_sum1.wav     # Baseline result of dry sound for source 1.
│   └── deconv_and_sum2.wav     # Baseline result of dry sound for source 2.
│
├── results_detection/
│   ├── detection_heatmap.png   # Heatmap visualizing source detection results.
│   └── metadata.pt             # Detection results and metadata.
│
└── metrics.json                # Quantitative metrics.

```

### 3. Novel-view Acoustic Rendering
Render the novel-view video integrated with the sound:

```bash
python demo/generate_demo_video.py
```

The video results will be structured as:
```yaml
results/nvas3d_demo/default/demo/{room}/0
└── video/
    ├── moving_audio.wav       # Audio only: Interpolated sound for the moving receiver.
    ├── moving_audio_1.wav     # Audio only: Interpolated sound from separated Source 1 for the moving receiver.
    ├── moving_audio_2.wav     # Audio only: Interpolated sound from separated Source 2 for the moving receiver.
    ├── moving_video.mp4       # Video only: Interpolated video for the moving receiver.
    ├── nvas.mp4               # Video with Audio: NVAS video results with combined audio from all sources.
    ├── nvas_source1.mp4       # Video with Audio: NVAS video results for separated Source 1 audio.
    ├── nvas_source2.mp4       # Video with Audio: NVAS video results for separated Source 2 audio.
    └── rgb_receiver.png       # Image: A static view rendered from the receiver's perspective for reference.
```

> [!Note]
> If you want to modify the novel receiver's path, edit `demo/nvas/config_demo/path1_novel_receiver.json`.
