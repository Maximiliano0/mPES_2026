"""
PES - Pandemic Experiment Scenario

result_formatter.py

Module for formatting and visualizing experiment results.
Generates JSON summary and PNG plots for experiment analysis.
"""

import json
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def generate_results_report(subject_id, outputs_path, performances, all_performances, resources_data=None):
    """
    Generate JSON and PNG report files for experiment results.
    
    Args:
        subject_id (str): Subject identifier (e.g., "001_TEST")
        outputs_path (str): Path to outputs directory
        performances (list): List of performance metrics per sequence
        all_performances (list): 2D array of performances per block/session
        resources_data (dict, optional): Additional resource allocation data
    
    Returns:
        tuple: (json_filepath, png_filepath)
    """
    
    # Calculate statistics
    stats = _calculate_statistics(performances, all_performances)
    
    # Generate JSON report
    json_filepath = _save_json_report(subject_id, outputs_path, stats, resources_data)
    
    # Generate PNG plots
    png_filepath = _save_png_plots(subject_id, outputs_path, performances, all_performances, stats)
    
    return json_filepath, png_filepath


def _calculate_statistics(performances, all_performances):
    """Calculate statistical summaries of performances."""
    
    performances = numpy.array(performances)
    all_performances = numpy.array(all_performances)
    
    stats = {
        'overall_mean': float(numpy.mean(performances)),
        'overall_median': float(numpy.median(performances)),
        'overall_std': float(numpy.std(performances)),
        'overall_min': float(numpy.min(performances)),
        'overall_max': float(numpy.max(performances)),
        'total_sequences': len(performances),
        'first_block_mean': float(numpy.mean(performances[:8])) if len(performances) >= 8 else None,
        'last_block_mean': float(numpy.mean(performances[-8:])) if len(performances) >= 8 else None,
        'improvement': float(numpy.mean(performances[-8:]) - numpy.mean(performances[:8])) if len(performances) >= 16 else None,
        'percentile_25': float(numpy.percentile(performances, 25)),
        'percentile_75': float(numpy.percentile(performances, 75)),
    }
    
    # Per-block statistics
    block_stats = []
    for block_idx, block_perf in enumerate(all_performances):
        block_perf = numpy.array(block_perf)
        block_stats.append({
            'block_number': block_idx + 1,
            'mean': float(numpy.mean(block_perf)),
            'median': float(numpy.median(block_perf)),
            'std': float(numpy.std(block_perf)),
            'min': float(numpy.min(block_perf)),
            'max': float(numpy.max(block_perf)),
            'num_sequences': len(block_perf),
        })
    
    stats['per_block_statistics'] = block_stats
    
    return stats


def _save_json_report(subject_id, outputs_path, stats, resources_data):
    """Save statistics to JSON file."""
    
    report = {
        'subject_id': subject_id,
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'resources': resources_data if resources_data else {},
        'report_type': 'PES_Experiment_Results'
    }
    
    json_filename = f'PES_results_{subject_id}.json'
    json_filepath = os.path.join(outputs_path, json_filename)
    
    with open(json_filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return json_filepath


def _save_png_plots(subject_id, outputs_path, performances, all_performances, stats):
    """Generate and save PNG plots for performance visualization."""
    
    try:
        performances = numpy.array(performances, dtype=float)
        
        # Normalize all_performances structure - ensure it's a list of lists
        normalized_all_perf = []
        if isinstance(all_performances, (list, numpy.ndarray)):
            for item in all_performances:
                if isinstance(item, (list, numpy.ndarray)):
                    item_list = [float(x) for x in item]
                    if item_list:  # Only add non-empty blocks
                        normalized_all_perf.append(item_list)
        
        if not normalized_all_perf:
            normalized_all_perf = [[float(x) for x in performances]]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Overall Performance Trend
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(range(1, len(performances) + 1), performances, 'b-o', linewidth=2, markersize=4, alpha=0.7)
        ax1.axhline(y=stats['overall_mean'], color='r', linestyle='--', linewidth=2, label=f"Mean: {stats['overall_mean']:.3f}")
        ax1.fill_between(range(1, len(performances) + 1), 
                         stats['overall_mean'] - stats['overall_std'],
                         stats['overall_mean'] + stats['overall_std'],
                         alpha=0.2, color='red')
        ax1.set_xlabel('Sequence Number', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Performance', fontsize=11, fontweight='bold')
        ax1.set_title('Performance Over All Sequences', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim([0, 1.05])
        
        # 2. Distribution Histogram
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(performances, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=stats['overall_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(x=stats['overall_median'], color='green', linestyle='--', linewidth=2, label='Median')
        ax2.set_xlabel('Performance', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Distribution', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        
        # 3. Box Plot by Block
        ax3 = fig.add_subplot(gs[1, 0])
        if len(normalized_all_perf) > 0:
            # Create proper boxplot data
            bp = ax3.boxplot(normalized_all_perf, labels=[f'B{i+1}' for i in range(len(normalized_all_perf))],
                              patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
        ax3.set_ylabel('Performance', fontsize=10, fontweight='bold')
        ax3.set_title('Performance by Block', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 1.05])
        
        # 4. Cumulative Mean
        ax4 = fig.add_subplot(gs[1, 1])
        cumulative_mean = numpy.cumsum(performances) / numpy.arange(1, len(performances) + 1)
        ax4.plot(range(1, len(cumulative_mean) + 1), cumulative_mean, 'g-o', linewidth=2, markersize=4)
        ax4.set_xlabel('Sequence Number', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Cumulative Mean', fontsize=10, fontweight='bold')
        ax4.set_title('Cumulative Mean Performance', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1.05])
        
        # 5. Block-wise Mean Comparison
        ax5 = fig.add_subplot(gs[1, 2])
        block_means = [numpy.mean(block) for block in normalized_all_perf]
        if len(block_means) > 0:
            ax5.bar(range(1, len(block_means) + 1), block_means, color='steelblue', alpha=0.7)
        ax5.set_xlabel('Block Number', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Mean Performance', fontsize=10, fontweight='bold')
        ax5.set_title('Block-wise Mean Performance', fontsize=11, fontweight='bold')
        ax5.set_ylim([0, 1.05])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Statistics Summary Table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('tight')
        ax6.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['Overall Mean', f"{stats['overall_mean']:.4f}"],
            ['Overall Median', f"{stats['overall_median']:.4f}"],
            ['Std Deviation', f"{stats['overall_std']:.4f}"],
            ['Min Performance', f"{stats['overall_min']:.4f}"],
            ['Max Performance', f"{stats['overall_max']:.4f}"],
            ['25th Percentile', f"{stats['percentile_25']:.4f}"],
            ['75th Percentile', f"{stats['percentile_75']:.4f}"],
            ['Total Sequences', f"{stats['total_sequences']}"],
        ]
        
        if stats['first_block_mean'] is not None:
            summary_data.append(['First Block Mean', f"{stats['first_block_mean']:.4f}"])
            summary_data.append(['Last Block Mean', f"{stats['last_block_mean']:.4f}"])
            summary_data.append(['Improvement', f"{stats['improvement']:.4f}"])
        
        table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        # Main title
        fig.suptitle(f'PES Experiment Results - Subject {subject_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        png_filename = f'PES_results_{subject_id}.png'
        png_filepath = os.path.join(outputs_path, png_filename)
        
        plt.savefig(png_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return png_filepath
        
    except Exception as e:
        print(f"Error generating PNG plots: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        raise


def print_results_summary(json_filepath, png_filepath):
    """Print summary of generated report files."""
    
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS REPORT GENERATED")
    print("="*70)
    print(f"✓ JSON Report: {json_filepath}")
    print(f"✓ PNG Plots:   {png_filepath}")
    print("="*70 + "\n")
    return json_filepath, png_filepath
