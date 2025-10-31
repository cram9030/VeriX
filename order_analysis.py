"""
Order Analysis Script for VeriX

This script analyzes the efficiency of different traversal orders (heuristic vs random)
by running VeriX on multiple images and generating:
1. Histogram comparing the traversal orders
2. Animations showing the pixel classification process

Usage:
    python order_analysis.py --dataset MNIST --num_images 5 --animation_image_idx 0
"""

import argparse
import os
import json
import random
import base64
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from VeriX import VeriX


def encode_image_to_base64(image_array):
    """
    Encode a numpy image array to base64 string for JSON storage.

    :param image_array: numpy array of image
    :return: base64-encoded string
    """
    # Convert to bytes
    buffer = BytesIO()
    np.save(buffer, image_array)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded


def decode_image_from_base64(encoded_string):
    """
    Decode a base64 string back to numpy image array.

    :param encoded_string: base64-encoded string
    :return: numpy array of image
    """
    decoded = base64.b64decode(encoded_string.encode('utf-8'))
    buffer = BytesIO(decoded)
    buffer.seek(0)
    image_array = np.load(buffer)
    return image_array


def load_dataset(dataset_name):
    """
    Load the specified dataset (MNIST or GTSRB).

    :param dataset_name: 'MNIST' or 'GTSRB'
    :return: tuple of (x_test, y_test, model_path)
    """
    if dataset_name == 'MNIST':
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32') / 255
        model_path = 'models/mnist-10x2.onnx'
        return x_test, y_test, model_path

    elif dataset_name == 'GTSRB':
        import pickle
        with open('models/gtsrb.pickle', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        x_test = data[b'x_test']
        y_test = data[b'y_test']
        x_test = x_test.astype('float32') / 255
        model_path = 'models/gtsrb-10x2.onnx'
        return x_test, y_test, model_path

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'MNIST' or 'GTSRB'.")


def select_images(x_test, num_images, image_indices=None, seed=None):
    """
    Select images from the test set.

    :param x_test: test images
    :param num_images: number of images to select
    :param image_indices: specific indices to use (optional)
    :param seed: random seed for reproducibility
    :return: list of selected image indices
    """
    if image_indices is not None:
        return image_indices

    if seed is not None:
        random.seed(seed)

    return random.sample(range(len(x_test)), num_images)


def run_verix_analysis(dataset, image, model_path, traverse, epsilon, output_dir):
    """
    Run VeriX analysis on a single image with specified traversal order.

    :param dataset: dataset name
    :param image: image array
    :param model_path: path to ONNX model
    :param traverse: 'heuristic' or 'random'
    :param epsilon: perturbation magnitude
    :param output_dir: directory for outputs
    :return: dictionary with 'result' (from get_explanation) and 'sensitivity' (numpy array or None)
    """
    verix = VeriX(
        dataset=dataset,
        image=image,
        model_path=model_path,
        plot_original=False,
        output_dir=output_dir
    )

    verix.traversal_order(traverse=traverse, plot_sensitivity=False)

    result = verix.get_explanation(
        epsilon=epsilon,
        plot_explanation=False,
        plot_counterfactual=False,
        plot_timeout=False
    )

    # Capture sensitivity (only available for heuristic traversal)
    return {
        'result': result,
        'sensitivity': verix.sensitivity
    }


def process_single_image(img_idx, x_test, dataset, model_path, epsilon, output_dir):
    """
    Process a single image with both heuristic and random traversal orders.
    This function is designed to be called by multiprocessing.Pool.

    :param img_idx: index of image in x_test
    :param x_test: test dataset array
    :param dataset: dataset name
    :param model_path: path to ONNX model
    :param epsilon: perturbation magnitude
    :param output_dir: directory for outputs
    :return: dictionary containing results for both traversal orders
    """
    image = x_test[img_idx]

    # Run heuristic traversal
    heuristic_analysis = run_verix_analysis(
        dataset=dataset,
        image=image,
        model_path=model_path,
        traverse='heuristic',
        epsilon=epsilon,
        output_dir=output_dir
    )

    # Run random traversal
    random_analysis = run_verix_analysis(
        dataset=dataset,
        image=image,
        model_path=model_path,
        traverse='random',
        epsilon=epsilon,
        output_dir=output_dir
    )

    return {
        'image_idx': img_idx,
        'image': image,
        'heuristic': heuristic_analysis['result'],
        'heuristic_sensitivity': heuristic_analysis['sensitivity'],
        'random': random_analysis['result'],
        'random_sensitivity': random_analysis['sensitivity']
    }


def load_results_from_json(json_path):
    """
    Load complete results from a JSON file.

    :param json_path: path to the results JSON file
    :return: tuple of (config dict, all_results list with decoded images)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    json_results = data['results']

    # Decode images and reconstruct results
    all_results = []
    for json_result in json_results:
        # Decode image from base64
        image = decode_image_from_base64(json_result['image_data'])

        # Convert sensitivity lists back to numpy arrays (handle None and backward compatibility)
        heuristic_sensitivity = json_result['heuristic'].get('sensitivity')
        if heuristic_sensitivity is not None:
            heuristic_sensitivity = np.array(heuristic_sensitivity)

        random_sensitivity = json_result['random'].get('sensitivity')
        if random_sensitivity is not None:
            random_sensitivity = np.array(random_sensitivity)

        # Reconstruct result structure
        result = {
            'image_idx': json_result['image_idx'],
            'image': image,
            'heuristic': {
                'sat_set': json_result['heuristic']['sat_set'],
                'timeout_set': json_result['heuristic']['timeout_set'],
                'unsat_set': json_result['heuristic']['unsat_set'],
                'history': json_result['heuristic']['history']
            },
            'heuristic_sensitivity': heuristic_sensitivity,
            'random': {
                'sat_set': json_result['random']['sat_set'],
                'timeout_set': json_result['random']['timeout_set'],
                'unsat_set': json_result['random']['unsat_set'],
                'history': json_result['random']['history']
            },
            'random_sensitivity': random_sensitivity
        }
        all_results.append(result)

    return config, all_results


def generate_histogram(all_results, dataset, output_dir):
    """
    Generate histogram comparing heuristic vs random traversal orders.

    X-axis: unsat_set size at the time a SAT pixel was found
    Y-axis: count of SAT pixels found at that unsat_set size

    :param all_results: dictionary containing results from all images
    :param dataset: dataset name for title
    :param output_dir: directory to save the HTML file
    """
    # Aggregate data for histogram
    heuristic_bins = {}
    random_bins = {}

    for result in all_results:
        history = result['heuristic']['history']
        for entry in history:
            if entry['result'] == 'sat':
                unsat_size = entry['unsat_size']
                heuristic_bins[unsat_size] = heuristic_bins.get(unsat_size, 0) + 1

        history = result['random']['history']
        for entry in history:
            if entry['result'] == 'sat':
                unsat_size = entry['unsat_size']
                random_bins[unsat_size] = random_bins.get(unsat_size, 0) + 1

    # Prepare data for plotting
    max_unsat = max(max(heuristic_bins.keys(), default=0), max(random_bins.keys(), default=0))
    x_values = list(range(max_unsat + 1))

    heuristic_counts = [heuristic_bins.get(x, 0) for x in x_values]
    random_counts = [random_bins.get(x, 0) for x in x_values]

    # Calculate zoom range (first and last non-zero bins)
    all_bins = set(heuristic_bins.keys()).union(set(random_bins.keys()))
    if all_bins:
        min_nonzero = min(all_bins)
        max_nonzero = max(all_bins)
    else:
        min_nonzero, max_nonzero = 0, 0

    # Create plotly figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_values,
        y=heuristic_counts,
        name='Heuristic',
        marker_color='green',
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        x=x_values,
        y=random_counts,
        name='Random',
        marker_color='red',
        opacity=0.7
    ))

    fig.update_layout(
        title=dict(
            text=f'{dataset} - SAT Pixels Found vs Unsat Set Size',
            font=dict(size=20)
        ),
        xaxis_title=dict(
            text='Unsat Set Size (number of irrelevant pixels identified)',
            font=dict(size=16)
        ),
        yaxis_title=dict(
            text='Count of SAT Pixels Found',
            font=dict(size=16)
        ),
        xaxis=dict(
            range=[min_nonzero - 0.5, max_nonzero + 0.5],
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            tickfont=dict(size=14)
        ),
        barmode='group',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=14)
        )
    )

    # Add summary statistics as annotations
    heuristic_total = sum(heuristic_counts)
    random_total = sum(random_counts)
    heuristic_mean = sum(k * v for k, v in heuristic_bins.items()) / heuristic_total if heuristic_total > 0 else 0
    random_mean = sum(k * v for k, v in random_bins.items()) / random_total if random_total > 0 else 0

    annotation_text = (
        f"<b>Summary:</b><br>"
        f"Heuristic - Total SAT: {heuristic_total}, Mean Unsat Size: {heuristic_mean:.2f}<br>"
        f"Random - Total SAT: {random_total}, Mean Unsat Size: {random_mean:.2f}"
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left",
        font=dict(size=12)
    )

    # Save as HTML
    output_path = os.path.join(output_dir, 'histogram_comparison.html')
    fig.write_html(output_path)
    print(f"Histogram saved to: {output_path}")

    return fig


def create_composited_frame(base_image, history_entry, dataset):
    """
    Create a single composited frame with colored pixels overlaid on base image.

    :param base_image: original image array
    :param history_entry: single entry from history with sat_set, timeout_set, unsat_set
    :param dataset: dataset name ('MNIST' or 'GTSRB')
    :return: RGB image as uint8 array (0-255 range) for Plotly
    """
    width, height = base_image.shape[0], base_image.shape[1]

    # Convert to RGB if grayscale
    if dataset == 'MNIST':
        base_gray = base_image[:, :, 0]
        frame_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)
    else:
        frame_rgb = base_image.copy()

    # Define colors (RGB format, values 0-1)
    colors = {
        'sat': np.array([0, 1, 0]),            # Green
        'timeout': np.array([1, 0.647, 0]),    # Orange
        'unsat': np.array([0.678, 0.847, 0.902])  # Light blue
    }
    alpha = 0.6  # Transparency factor

    # Apply colors with alpha blending for SAT pixels
    for pixel_idx in history_entry['sat_set']:
        row, col = pixel_idx // height, pixel_idx % height
        frame_rgb[row, col] = alpha * colors['sat'] + (1 - alpha) * frame_rgb[row, col]

    # Apply colors for TIMEOUT pixels
    for pixel_idx in history_entry['timeout_set']:
        row, col = pixel_idx // height, pixel_idx % height
        frame_rgb[row, col] = alpha * colors['timeout'] + (1 - alpha) * frame_rgb[row, col]

    # Apply colors for UNSAT pixels
    for pixel_idx in history_entry['unsat_set']:
        row, col = pixel_idx // height, pixel_idx % height
        frame_rgb[row, col] = alpha * colors['unsat'] + (1 - alpha) * frame_rgb[row, col]

    # Convert to uint8 (0-255 range) for Plotly
    frame_uint8 = (frame_rgb * 255).astype(np.uint8)

    return frame_uint8


def generate_sensitivity_overlay(image, sensitivity, pixel_set, dataset,
                                 set_name, image_idx, output_dir,
                                 global_sens_range=None):
    """
    Generate interactive visualization of sensitivity values overlaid on image.
    Only pixels in pixel_set will show sensitivity values as a heatmap overlay.

    :param image: original image array
    :param sensitivity: 2D sensitivity array (width, height)
    :param pixel_set: set of pixel indices to visualize (sat_set/unsat_set)
    :param dataset: dataset name ('MNIST' or 'GTSRB')
    :param set_name: name of the set ('SAT' or 'UNSAT')
    :param image_idx: index of the image
    :param output_dir: directory to save the HTML file
    :param global_sens_range: tuple (min, max) for consistent color scaling
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    height = image.shape[1]

    # Convert base image to RGB for display
    if dataset == 'MNIST':
        base_gray = image[:, :, 0]
        base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)
    else:
        base_rgb = image.copy()

    # Convert to uint8 for base image
    base_uint8 = (base_rgb * 255).astype(np.uint8)

    # Extract sensitivity values for selected pixels
    sensitivity_values = []
    for pixel_idx in pixel_set:
        row, col = pixel_idx // height, pixel_idx % height
        sensitivity_values.append(sensitivity[row, col])

    if len(sensitivity_values) > 0:
        # Normalize sensitivity for color mapping
        sens_array = np.array(sensitivity_values)

        # Use global range if provided, otherwise use local range
        if global_sens_range is not None:
            vmin, vmax = global_sens_range
        else:
            vmin, vmax = sens_array.min(), sens_array.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap('viridis')

        alpha = 0.7  # Overlay transparency
        result_rgb = base_rgb.copy()

        # Overlay sensitivity colors on selected pixels
        for pixel_idx in pixel_set:
            row, col = pixel_idx // height, pixel_idx % height
            sens_val = sensitivity[row, col]
            color_rgba = colormap(norm(sens_val))
            color_rgb = np.array(color_rgba[:3])
            result_rgb[row, col] = alpha * color_rgb + (1 - alpha) * result_rgb[row, col]

        # Convert to uint8
        result_uint8 = (result_rgb * 255).astype(np.uint8)

        # Create Plotly figure with overlaid image
        fig = go.Figure(data=go.Image(z=result_uint8))

        # Add a dummy scatter trace for colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=1,
                color=[vmin, vmax],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(text='Sensitivity', font=dict(size=14)),
                    len=0.7,
                    thickness=15,
                    x=1.02
                )
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        # No pixels in the set
        fig = go.Figure(data=go.Image(z=base_uint8))
        fig.add_annotation(
            text=f"No pixels in {set_name} set",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{dataset} - {set_name} Set Sensitivity Overlay - Image {image_idx}',
            font=dict(size=18)
        ),
        xaxis=dict(showticklabels=False, showgrid=False, constrain='domain'),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x', constrain='domain'),
        template='plotly_white',
        height=600,
        width=700
    )

    # Save as HTML
    output_filename = f'sensitivity_overlay_{set_name.lower()}_image_{image_idx}.html'
    output_path = os.path.join(output_dir, output_filename)
    fig.write_html(output_path)
    print(f"  Sensitivity overlay ({set_name}) saved to: {output_path}")

    return fig


def generate_animation(image, result, dataset, traverse, image_idx, output_dir):
    """
    Generate animation showing the pixel classification process.

    :param image: original image array
    :param result: result dictionary from get_explanation
    :param dataset: dataset name
    :param traverse: traversal method used
    :param image_idx: index of the image
    :param output_dir: directory to save the HTML file
    """
    history = result['history']

    # Create composited frames for animation
    frames = []
    for entry in history:
        composited_frame = create_composited_frame(image, entry, dataset)
        frames.append(composited_frame)

    # Create the animation figure with first frame
    fig = go.Figure(data=[go.Image(z=frames[0])])

    # Add legend using invisible scatter traces
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgb(0, 255, 0)'),
        showlegend=True,
        name='SAT (Relevant)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgb(255, 165, 0)'),
        showlegend=True,
        name='TIMEOUT'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgb(173, 216, 230)'),
        showlegend=True,
        name='UNSAT (Irrelevant)'
    ))

    # Create animation frames
    plotly_frames = []
    for i, composited_frame in enumerate(frames):
        entry = history[i]

        frame_title = (
            f"{dataset} - {traverse.capitalize()} Traversal - Image {image_idx} | "
            f"Step: {entry['step']} | "
            f"SAT: {len(entry['sat_set'])} | "
            f"TIMEOUT: {len(entry['timeout_set'])} | "
            f"UNSAT: {len(entry['unsat_set'])}"
        )

        # Frame data only includes the image - legend traces are static in initial figure
        frame_data = [go.Image(z=composited_frame)]

        plotly_frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            traces=[0],  # Explicitly update only trace 0 (the image)
            layout=go.Layout(title_text=frame_title)
        ))

    fig.frames = plotly_frames

    # Update layout with animation controls
    fig.update_layout(
        title=f"{dataset} - {traverse.capitalize()} Traversal - Image {image_idx}",
        xaxis=dict(showticklabels=False, showgrid=False, constrain='domain'),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x', constrain='domain'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 10, "redraw": True},
                                     "fromcurrent": True,
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}])
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                y=-0.1,
                xanchor="left",
                currentvalue=dict(
                    prefix="Step: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                steps=[
                    dict(
                        args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        label=str(k),
                        method="animate"
                    )
                    for k, f in enumerate(fig.frames)
                ]
            )
        ]
    )

    # Save as HTML
    output_filename = f'animation_{traverse}_image_{image_idx}.html'
    output_path = os.path.join(output_dir, output_filename)
    fig.write_html(output_path, auto_play=False)
    print(f"Animation saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='VeriX Order Analysis')

    # Load mode: regenerate plots from existing results
    parser.add_argument('--load_results', type=str, default=None,
                        help='Path to results JSON file (for plot-only mode)')

    # Analysis mode parameters (not required if loading from JSON)
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'GTSRB'],
                        help='Dataset to use (MNIST or GTSRB)')
    parser.add_argument('--num_images', type=int,
                        help='Number of images to analyze')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Perturbation magnitude (default: 0.05 for MNIST, 0.01 for GTSRB)')
    parser.add_argument('--image_indices', type=int, nargs='+', default=None,
                        help='Specific image indices to use (optional)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect as cpu_count // 4)')

    # Common parameters
    parser.add_argument('--animation_image_idx', type=int, required=True,
                        help='Index (0-based) from the selected set to animate')
    parser.add_argument('--output_dir', type=str, default='outputs/order_analysis',
                        help='Output directory (default: outputs/order_analysis)')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.load_results is None:
        # Full analysis mode: require dataset and num_images
        if args.dataset is None or args.num_images is None:
            parser.error("--dataset and --num_images are required when not using --load_results")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Branch based on mode
    if args.load_results is not None:
        # ===== PLOT-ONLY MODE: Load from JSON and regenerate plots =====
        print(f"Loading results from: {args.load_results}")
        config, all_results = load_results_from_json(args.load_results)

        dataset = config['dataset']
        print(f"Loaded {len(all_results)} results for dataset: {dataset}")

        # Generate histogram
        print("\n--- Generating histogram from loaded results ---")
        generate_histogram(all_results, dataset, args.output_dir)

        # Generate animations for specified image
        if args.animation_image_idx < len(all_results):
            print(f"\n--- Generating animations for image index {args.animation_image_idx} ---")
            animation_result = all_results[args.animation_image_idx]
            animation_image = animation_result['image']
            animation_img_idx = animation_result['image_idx']

            print("  Generating heuristic animation...")
            generate_animation(
                image=animation_image,
                result=animation_result['heuristic'],
                dataset=dataset,
                traverse='heuristic',
                image_idx=animation_img_idx,
                output_dir=args.output_dir
            )

            print("  Generating random animation...")
            generate_animation(
                image=animation_image,
                result=animation_result['random'],
                dataset=dataset,
                traverse='random',
                image_idx=animation_img_idx,
                output_dir=args.output_dir
            )

            # Generate sensitivity overlays for heuristic traversal
            if animation_result['heuristic_sensitivity'] is not None:
                print("\n--- Generating sensitivity overlays ---")

                # Calculate global sensitivity range for consistent coloring
                sensitivity = animation_result['heuristic_sensitivity']
                global_sens_range = (
                    float(sensitivity.min()),
                    float(sensitivity.max())
                )

                print("  Generating SAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=sensitivity,
                    pixel_set=animation_result['heuristic']['sat_set'],
                    dataset=dataset,
                    set_name='SAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=global_sens_range
                )

                print("  Generating UNSAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=sensitivity,
                    pixel_set=animation_result['heuristic']['unsat_set'],
                    dataset=dataset,
                    set_name='UNSAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=global_sens_range
                )

                print("  Generating Random SAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=sensitivity,
                    pixel_set=animation_result['random']['sat_set'],
                    dataset=dataset,
                    set_name='Random_SAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=global_sens_range
                )

                print("  Generating Random UNSAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=sensitivity,
                    pixel_set=animation_result['random']['unsat_set'],
                    dataset=dataset,
                    set_name='Random_UNSAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=global_sens_range
                )
            else:
                print("\n--- Note: No sensitivity data available ---")
        else:
            print(f"\n--- ERROR: animation_image_idx {args.animation_image_idx} is out of range (0-{len(all_results)-1}) ---")

        print("\n=== Plot generation complete! ===")
        print(f"All outputs saved to: {args.output_dir}")

    else:
        # ===== FULL ANALYSIS MODE: Run VeriX and generate everything =====

        # Set default epsilon based on dataset
        if args.epsilon is None:
            args.epsilon = 0.05 if args.dataset == 'MNIST' else 0.01

        print(f"Loading {args.dataset} dataset...")
        x_test, y_test, model_path = load_dataset(args.dataset)

        print(f"Selecting {args.num_images} images...")
        selected_indices = select_images(x_test, args.num_images, args.image_indices, args.random_seed)
        print(f"Selected image indices: {selected_indices}")

        # Calculate number of parallel workers
        if args.workers is not None:
            num_workers = args.workers
        else:
            # Auto-detect: use cpu_count // 4 as smart default
            num_workers = max(1, cpu_count() // 4)
        print(f"Using {num_workers} parallel workers")

        # Run analysis on each image in parallel
        print(f"\n--- Processing {args.num_images} images in parallel ---")
        process_func = partial(
            process_single_image,
            x_test=x_test,
            dataset=args.dataset,
            model_path=model_path,
            epsilon=args.epsilon,
            output_dir=args.output_dir
        )

        with Pool(processes=num_workers) as pool:
            all_results = list(tqdm(
                pool.imap(process_func, selected_indices),
                total=len(selected_indices),
                desc="Analyzing images"
            ))

        # Print summary for each image
        for i, result in enumerate(all_results):
            img_idx = result['image_idx']
            print(f"\n--- Image {i+1}/{args.num_images} (index: {img_idx}) Results ---")
            print(f"  Heuristic - SAT: {len(result['heuristic']['sat_set'])}, "
                  f"TIMEOUT: {len(result['heuristic']['timeout_set'])}, "
                  f"UNSAT: {len(result['heuristic']['unsat_set'])}")
            print(f"  Random - SAT: {len(result['random']['sat_set'])}, "
                  f"TIMEOUT: {len(result['random']['timeout_set'])}, "
                  f"UNSAT: {len(result['random']['unsat_set'])}")

        # Generate histogram
        print("\n--- Generating histogram ---")
        generate_histogram(all_results, args.dataset, args.output_dir)

        # Generate animations for specified image
        if args.animation_image_idx < len(selected_indices):
            print(f"\n--- Generating animations for image index {args.animation_image_idx} ---")
            animation_img_idx = selected_indices[args.animation_image_idx]
            animation_image = x_test[animation_img_idx]
            animation_results = all_results[args.animation_image_idx]

            print("  Generating heuristic animation...")
            generate_animation(
                image=animation_image,
                result=animation_results['heuristic'],
                dataset=args.dataset,
                traverse='heuristic',
                image_idx=animation_img_idx,
                output_dir=args.output_dir
            )

            print("  Generating random animation...")
            generate_animation(
                image=animation_image,
                result=animation_results['random'],
                dataset=args.dataset,
                traverse='random',
                image_idx=animation_img_idx,
                output_dir=args.output_dir
            )

            # Generate sensitivity overlays for heuristic traversal
            if animation_results['heuristic_sensitivity'] is not None:
                print("\n--- Generating sensitivity overlays ---")

                # Calculate global sensitivity range for consistent coloring
                sensitivity = animation_results['heuristic_sensitivity']
                global_sens_range = (
                    float(sensitivity.min()),
                    float(sensitivity.max())
                )

                print("  Generating SAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=sensitivity,
                    pixel_set=animation_results['heuristic']['sat_set'],
                    dataset=args.dataset,
                    set_name='SAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=global_sens_range
                )

                print("  Generating UNSAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=sensitivity,
                    pixel_set=animation_results['heuristic']['unsat_set'],
                    dataset=args.dataset,
                    set_name='UNSAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=global_sens_range
                )

            # Generate sensitivity overlays for random traversal
            if animation_results['random_sensitivity'] is not None:
                print("\n--- Generating random traversal sensitivity overlays ---")

                # Calculate global sensitivity range for consistent coloring
                random_sensitivity = animation_results['random_sensitivity']
                random_sens_range = (
                    float(random_sensitivity.min()),
                    float(random_sensitivity.max())
                )

                print("  Generating Random SAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=random_sensitivity,
                    pixel_set=animation_results['random']['sat_set'],
                    dataset=args.dataset,
                    set_name='Random_SAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=random_sens_range
                )

                print("  Generating Random UNSAT set sensitivity overlay...")
                generate_sensitivity_overlay(
                    image=animation_image,
                    sensitivity=random_sensitivity,
                    pixel_set=animation_results['random']['unsat_set'],
                    dataset=args.dataset,
                    set_name='Random_UNSAT',
                    image_idx=animation_img_idx,
                    output_dir=args.output_dir,
                    global_sens_range=random_sens_range
                )
            else:
                print("\n--- Note: No sensitivity data available ---")
        else:
            print(f"\n--- ERROR: animation_image_idx {args.animation_image_idx} is out of range (0-{len(selected_indices)-1}) ---")

        # Save complete results as JSON (with full history and image data)
        print("\n--- Saving complete results ---")
        # Convert numpy arrays and images to JSON-serializable format
        json_results = []
        for result in all_results:
            # Encode image to base64
            image_encoded = encode_image_to_base64(result['image'])

            # Convert history entries (convert sets to lists)
            def convert_history(history):
                return [{
                    'step': int(entry['step']),
                    'pixel': int(entry['pixel']),
                    'result': entry['result'],
                    'unsat_size': int(entry['unsat_size']),
                    'sat_set': [int(x) for x in entry['sat_set']],
                    'timeout_set': [int(x) for x in entry['timeout_set']],
                    'unsat_set': [int(x) for x in entry['unsat_set']]
                } for entry in history]

            # Convert sensitivity arrays to lists (handle None values)
            heuristic_sensitivity_list = (
                result['heuristic_sensitivity'].tolist()
                if result['heuristic_sensitivity'] is not None
                else None
            )
            random_sensitivity_list = (
                result['random_sensitivity'].tolist()
                if result['random_sensitivity'] is not None
                else None
            )

            json_result = {
                'image_idx': int(result['image_idx']),
                'image_data': image_encoded,
                'image_shape': list(result['image'].shape),
                'heuristic': {
                    'sat_set': [int(x) for x in result['heuristic']['sat_set']],
                    'timeout_set': [int(x) for x in result['heuristic']['timeout_set']],
                    'unsat_set': [int(x) for x in result['heuristic']['unsat_set']],
                    'history': convert_history(result['heuristic']['history']),
                    'sensitivity': heuristic_sensitivity_list
                },
                'random': {
                    'sat_set': [int(x) for x in result['random']['sat_set']],
                    'timeout_set': [int(x) for x in result['random']['timeout_set']],
                    'unsat_set': [int(x) for x in result['random']['unsat_set']],
                    'history': convert_history(result['random']['history']),
                    'sensitivity': random_sensitivity_list
                }
            }
            json_results.append(json_result)

        # Create complete JSON with metadata
        complete_json = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'script_version': '3.0',
                'description': 'Complete VeriX order analysis results with full history, image data, and sensitivity maps'
            },
            'config': {
                'dataset': args.dataset,
                'num_images': args.num_images,
                'epsilon': float(args.epsilon),
                'random_seed': args.random_seed,
                'selected_indices': selected_indices
            },
            'results': json_results
        }

        summary_path = os.path.join(args.output_dir, 'results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(complete_json, f, indent=2)

        print(f"Results summary saved to: {summary_path}")
        print("\n=== Analysis complete! ===")
        print(f"All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
