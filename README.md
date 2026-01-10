# 1. Generate dataset once (fast for debugging)
python generate_data.py --n_samples 5000 --output_dir data/generated

# 2. Train with generated data (first run slow, subsequent fast)
python train.py  # mode='generate'

# 3. Train with saved files (fastest)
python train.py  # mode='load', data_dir='data/generated'

# 4. Mixed mode for augmentation
python train.py  # mode='mixed'



docker run --gpus all --rm change-detect python3 -c "
from data_generators.dataset import ChangeDetectionDataset
from models.pointnet_mlp import PointNetMLP
print('✅ All imports work!')
"



# 1. Build (works now!)
docker build -t change-detect .

# 2. Interactive session
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  -p 6006:6006 \
  change-detect bash


# On HOST (enable X11 forwarding)
xhost +local:docker

# Run with display
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  change-detect bash


# Inside:
ls data_generators/  # see dataset.py, scene.py, shapes.py
python -c "from data_generators.dataset import ChangeDetectionDataset; print('OK')"

# Generate + train
python generate_data.py --n_samples 1000 --output_dir data/generated
python train.py --n_train 800 --n_val 200 --epochs 20 --output_dir outputs/test



# Inside Docker:

# 1. Generate HDF5 dataset
python utils/file_io.py --n_samples 5000 --batch_size 1000 --output_dir data/generated

# 2. Quick dataset preview
python -c "from utils.viz import quick_sample_preview; quick_sample_preview(4)"

# 3. Train
python train.py --n_train 4000 --data_mode load --output_dir outputs/run1

# 4. Visualize predictions
python -c "from utils.viz import plot_predictions; plot_predictions('outputs/run1/best_model.pth', 6)"

# 5. Single sample plot
python -c "
from data_generators.dataset import ChangeDetectionDataset
from utils.viz import plot_sample
ds = ChangeDetectionDataset(1)
plot_sample(ds[0], 'single_sample.png')
"


# 1. Single sample (first file, sample 0)
python viz_hdf5.py data/generated

# 2. Specific sample
python viz_hdf5.py data/generated --sample 5

# 3. Batch overview (12 samples)
python viz_hdf5.py data/generated --overview

# 4. Save everything
python viz_hdf5.py data/generated --overview --save

export PYTHONPATH=$PYTHONPATH:/app


python utils/file_io.py --n_samples 10 --batch_size 10 --output_dir data/generated



✅ Your Final Problem Definition (Restated Precisely)

You want the model to learn:

Given two unordered point clouds P and Q, predict which spatial regions correspond to a geometric change.

Where “change” includes:

rigidly transformed shapes

partially removed shapes

fully removed shapes

newly added shapes

And you do NOT require:

point-to-point correspondence

identity tracking

change typing (add/remove/move)

You will:

randomly permute point order

treat point clouds as sets

supervise only where change occurred

This is a pure geometric change detection problem.

Given:

P ∈ ℝ^{N×2}
Q ∈ ℝ^{N×2}
change ∈ {0,1}^N


The network must infer:

regions that do not align geometrically

regions that exist in one but not the other

regions that moved

This is exactly what modern change detection networks (PCD, Flow-based, attention-based) do.

🧩 Architectural Compatibility (Good News)

This formulation works well with:

PointNet / PointNet++

Set Transformers

Cross-attention models

Flow-based matching (without supervision)

Siamese encoders + difference heads

add randome transformations to each shapes as well


Partial remove
Let's first skip this for now



You have two point clouds, P (original scene) and Q (changed scene).

The point clouds are permuted and of different sizes; there’s no direct correspondence.

You want the model to detect which points in the new scene Q correspond to “changed regions”, without assuming index-wise matching.

This is fundamentally a set-to-set change detection problem, not per-point classification. So yes, the model has to implicitly reason about correspondences between P and Q and then infer which points in Q are “changed”.

1️⃣ Conceptual Approaches

Here are some possible ways to model this:

A. Point-wise attention / set matching

Encode P and Q separately into per-point features.

Compute cross-attention between points of Q and points of P.

This lets each Q point “look at” all points in P to see if it has a matching point.

Predict change for Q based on how well it matches something in P.

This is similar to transformer-based point cloud matching or soft correspondence networks.

B. Set-level encoding → region prediction

Encode P and Q globally (e.g., PointNet-style max pooling).

Combine global features: e.g., diff = f(Q) - f(P).

Decode diff to per-point change likelihood in Q.

No explicit correspondences are required; the model must infer them from context.

Works well if the changes are large enough spatially, but may struggle with tiny local changes.

C. Graph-based / bipartite matching

Treat P and Q as nodes in a graph.

Learn a soft matching matrix M between P and Q.

For each Q point, compute a weighted sum of features from P based on M.

Predict change as the difference between matched features.

This is more explicit about correspondences, but heavier computationally.

2️⃣ Simplified Idea for Your Current Scenario

Since you’re using PointNet-style features, the simplest working approach is:

Encode P and Q separately using an MLP (or PointNet).

Max-pool each to get a global context: g_P, g_Q.

For each point in Q, concatenate its local feature with the global difference: feat_Q_i || (g_Q - g_P).

Feed this into a small decoder MLP → predict whether Q_i is in a changed region.

Intuition:

If a point in Q has no corresponding point in P, its feature plus global difference will make the model predict “changed”.

No per-point matching is needed, but the model must reason globally.



python train.py --data_dir data/generated --epochs 100 --batch_size 16


the mask vector is to pad the pointcloud of different lengths.


When to use it: Once your code is stable and you want to run a "sweep" (e.g., testing 5 different model designs or 3 different geometric change thresholds).