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
