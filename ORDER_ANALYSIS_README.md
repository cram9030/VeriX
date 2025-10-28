# Order Analysis Script

This document explains how to use the `order_analysis.py` script to analyze and compare the efficiency of different traversal orders (heuristic vs random) in VeriX.

## Overview

The `order_analysis.py` script provides:

1. **Comparative Analysis**: Runs VeriX on multiple images with both heuristic and random traversal orders
2. **Histogram Visualization**: Generates an interactive Plotly histogram showing when SAT pixels are discovered during the traversal
3. **Animation Visualization**: Creates frame-by-frame animations showing the pixel classification process

## What Changed in VeriX.py

The `get_explanation()` method in `VeriX.py` has been enhanced to track detailed history at each iteration:

### New History Format

Previously, history only recorded information when a SAT pixel was found:
```python
{
    'pixel': int,
    'sat_size': int,
    'timeout_size': int,
    'unsat_size': int
}
```

Now, history records information at **every iteration** with the complete state:
```python
{
    'step': int,                    # Iteration number (0-indexed)
    'pixel': int,                   # Current pixel being tested
    'result': str,                  # 'sat', 'unsat', or 'timeout'
    'unsat_size': int,             # Size of unsat_set BEFORE processing this pixel
    'sat_set': list,               # Copy of current sat_set
    'timeout_set': list,           # Copy of current timeout_set
    'unsat_set': list              # Copy of current unsat_set
}
```

This allows reconstruction of the exact state at any point during the analysis.

## Installation

### Using Docker (Recommended)

1. Navigate to the environment directory:
   ```bash
   cd environment
   ```

2. Build the Docker image (this includes plotly):
   ```bash
   docker-compose build verix-gpu  # For GPU support
   # OR
   docker-compose build verix-cpu  # For CPU only
   ```

3. Start the container:
   ```bash
   docker-compose run verix-gpu
   # OR
   docker-compose run verix-cpu
   ```

4. Inside the container, navigate to the VeriX directory:
   ```bash
   cd /workspace/verix
   ```

### Manual Installation

If not using Docker, ensure you have the following dependencies:
```bash
pip install numpy plotly tensorflow keras onnx onnxruntime scikit-image matplotlib
```

## Usage

### Basic Usage

```bash
python order_analysis.py \
    --dataset MNIST \
    --num_images 5 \
    --animation_image_idx 0
```

### All Options

```bash
python order_analysis.py \
    --dataset {MNIST,GTSRB} \
    --num_images NUM \
    --epsilon EPSILON \
    --image_indices IDX1 IDX2 ... \
    --animation_image_idx IDX \
    --random_seed SEED \
    --output_dir PATH
```

### Parameters

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | Yes | Dataset to use: 'MNIST' or 'GTSRB' | - |
| `--num_images` | Yes | Number of images to analyze | - |
| `--epsilon` | No | Perturbation magnitude | 0.05 (MNIST), 0.01 (GTSRB) |
| `--image_indices` | No | Specific image indices to use | Random selection |
| `--animation_image_idx` | Yes | Which image from the set to animate (0-indexed) | - |
| `--random_seed` | No | Random seed for reproducibility | 42 |
| `--output_dir` | No | Output directory for results | `outputs/order_analysis` |

## Examples

### Example 1: Analyze 5 random MNIST images

```bash
python3 order_analysis.py \
    --dataset MNIST \
    --num_images 5 \
    --animation_image_idx 0
```

This will:
- Select 5 random MNIST images
- Run VeriX with both heuristic and random traversal on each image
- Generate a histogram comparing the traversal orders
- Create animations for the first image (index 0) showing both traversals

### Example 2: Analyze specific GTSRB images

```bash
python3 order_analysis.py \
    --dataset GTSRB \
    --num_images 3 \
    --image_indices 10 25 50 \
    --animation_image_idx 1 \
    --epsilon 0.02 \
    --output_dir outputs/my_analysis
```

This will:
- Analyze GTSRB images at indices 10, 25, and 50
- Use epsilon=0.02 for perturbation
- Create animations for the image at index 25 (the second in the list)
- Save all outputs to `outputs/my_analysis`

### Example 3: Reproducible analysis

```bash
python3 order_analysis.py \
    --dataset MNIST \
    --num_images 10 \
    --animation_image_idx 0 \
    --random_seed 12345
```

Using `--random_seed` ensures the same random images are selected each time.

### Example 4: Plot only mode

```bash
python3 order_analysis.py \
      --load_results outputs/order_analysis/results_summary.json \
      --animation_image_idx 0 \
      --output_dir outputs/new_plots
```
Using '--load_results' will use the provided json to generate the plots.

## Output Files

The script generates the following files in the output directory:

### 1. `histogram_comparison.html`

An interactive Plotly histogram comparing heuristic vs random traversal orders.

**Axes:**
- **X-axis**: Size of the unsat_set when a SAT pixel was discovered
- **Y-axis**: Count of SAT pixels discovered at that unsat_set size

**Interpretation:**
- A histogram skewed **left** (toward smaller unsat_set sizes) means the traversal finds relevant pixels early with fewer irrelevant pixels identified
- **Heuristic traversal** typically shows a left-skewed distribution, indicating it efficiently identifies relevant pixels early
- **Random traversal** typically shows a more uniform or right-skewed distribution

### 2. `animation_heuristic_image_<idx>.html`

Frame-by-frame animation showing the heuristic traversal process.

**Color coding:**
- **Transparent Green**: SAT pixels (adversarial examples found)
- **Transparent Light Blue**: UNSAT pixels (irrelevant pixels)
- **Transparent Orange**: TIMEOUT pixels (solver timed out)

### 3. `animation_random_image_<idx>.html`

Frame-by-frame animation showing the random traversal process (same color coding as above).

### 4. `results_summary.json`

JSON file containing:
- Dataset and parameters used
- Selected image indices
- Final counts of sat_set, timeout_set, and unsat_set for each image and traversal method
- History length for each analysis

## Understanding the Results

### Histogram Analysis

The histogram shows the **efficiency** of each traversal order:

- **Lower unsat_set sizes at discovery** = More efficient (fewer irrelevant pixels checked before finding relevant ones)
- **Higher SAT counts at low unsat_sizes** = Better traversal order

**Example interpretation:**
```
If heuristic shows:
- 20 SAT pixels found when unsat_size = 0-5
- 10 SAT pixels found when unsat_size = 6-10

And random shows:
- 5 SAT pixels found when unsat_size = 0-5
- 15 SAT pixels found when unsat_size = 6-10
- 10 SAT pixels found when unsat_size = 11-15

Then heuristic is more efficient - it finds most relevant pixels earlier
```

### Animation Analysis

The animations help visualize:
1. **Order of pixel checking**: You can see which pixels are tested first
2. **Clustering of relevant pixels**: SAT pixels often cluster together
3. **Efficiency**: How quickly the relevant region is identified
4. **Comparison**: Side-by-side viewing of heuristic vs random shows the difference clearly

## Performance Considerations

- **Runtime**: Each image requires running VeriX twice (heuristic + random), which involves solving multiple Marabou queries. Expect several minutes per image depending on image complexity and timeout settings.
- **Memory**: Large numbers of images may require significant memory for storing history
- **GPU**: If using GPU-accelerated ONNX runtime, ensure your Docker container has GPU access

## Tips for Jekyll Blog Integration

The generated HTML files can be embedded in Jekyll blog posts:

1. Copy the HTML files to your Jekyll `assets` directory
2. Embed in a blog post using an iframe:

```html
<iframe src="/assets/histogram_comparison.html"
        width="100%"
        height="600"
        frameborder="0">
</iframe>
```

Or link directly:
```markdown
[View Interactive Histogram](/assets/histogram_comparison.html)
```

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure you're running inside the Docker container or have all dependencies installed.

### Issue: CUDA/GPU errors

**Solution**: Use the CPU Docker variant (`verix-cpu`) or ensure your GPU drivers and NVIDIA Docker runtime are properly configured.

### Issue: Gurobi license errors

**Solution**: Mount your Gurobi license file in the docker-compose.yml (uncomment the license volume mount line).

### Issue: Long runtime

**Solution**:
- Reduce `--num_images`
- Adjust timeout in VeriX.py options (line 33-36)
- Use simpler images (MNIST is faster than GTSRB)

## Advanced Usage

### Custom epsilon values per dataset

```bash
# Higher epsilon for MNIST (allows larger perturbations)
python order_analysis.py --dataset MNIST --num_images 3 --epsilon 0.1 --animation_image_idx 0

# Lower epsilon for GTSRB (more subtle perturbations)
python order_analysis.py --dataset GTSRB --num_images 3 --epsilon 0.005 --animation_image_idx 0
```

### Analyzing specific challenging images

If you've identified particularly interesting images through prior analysis:

```bash
python order_analysis.py \
    --dataset MNIST \
    --num_images 3 \
    --image_indices 1234 5678 9012 \
    --animation_image_idx 0
```

## Contributing

If you find bugs or have suggestions for improvements, please open an issue on the VeriX repository.

## References

- VeriX Paper: [Add citation]
- Marabou: https://github.com/NeuralNetworkVerification/Marabou
- Plotly: https://plotly.com/python/
