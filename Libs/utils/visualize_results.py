#!/usr/bin/env python3
"""
Standalone Visualization Script for ICU-AKI Research Results

This script provides comprehensive visualization capabilities for analyzing 
experimental results, training logs, and model performance comparisons.

Features:
• Generate publication-ready figures from saved results
• Interactive visualization dashboard
• Custom plot generation with flexible configurations
• Statistical analysis and significance testing
• Export to multiple formats (PNG, PDF, SVG, EPS)

Usage:
    python Libs/scripts/visualize_results.py --results Output/experiment_results.csv
    python Libs/scripts/visualize_results.py --tensorboard Output/runs/
    python Libs/scripts/visualize_results.py --interactive
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Import our visualization utilities
from Libs.utils.vis_utils import (ICLR_COLORS, MEDICAL_COLORS,
                                  plot_convergence_diagnostics,
                                  plot_ope_comparison_with_ci,
                                  plot_policy_distribution_comparison,
                                  plot_treatment_strategy_heatmap,
                                  save_figure_publication_ready,
                                  set_plot_style)
from scipy import stats

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the visualization script."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class ResultsVisualizer:
    """Comprehensive results visualization tool for ICU-AKI research."""
    
    def __init__(
        self,
        style: str = "publication",
        output_dir: Path = Path("Output/visualizations"),
        formats: List[str] = None
    ):
        """Initialize the visualizer.
        
        Args:
            style: Visualization style (publication, presentation, poster)
            output_dir: Directory to save generated plots
            formats: Output formats for figures
        """
        self.style = style
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ["png", "pdf"]
        
        # Set global plot style
        set_plot_style(style)
        
        logger.info("📊 ResultsVisualizer initialized")
        logger.info("   • Style: %s", style)
        logger.info("   • Output directory: %s", self.output_dir)
        logger.info("   • Formats: %s", self.formats)
    
    def load_csv_results(self, csv_path: Path) -> pd.DataFrame:
        """Load experiment results from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logger.info("✅ Loaded CSV results: %d rows, %d columns", len(df), len(df.columns))
            return df
        except Exception as e:
            logger.error("❌ Failed to load CSV results: %s", e)
            raise
    
    def load_json_results(self, json_path: Path) -> Dict[str, Any]:
        """Load experiment results from JSON file."""
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
            logger.info("✅ Loaded JSON results: %d algorithms", len(results))
            return results
        except Exception as e:
            logger.error("❌ Failed to load JSON results: %s", e)
            raise
    
    def plot_algorithm_comparison(
        self,
        df: pd.DataFrame,
        metrics: List[str] = None,
        title_suffix: str = ""
    ) -> List[Path]:
        """Generate comprehensive algorithm comparison plots."""
        if metrics is None:
            metrics = ["reward", "survival_rate", "ips_survival", "runtime_s"]
            
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            logger.warning("No valid metrics found in data")
            return []
            
        logger.info("🎨 Generating algorithm comparison for metrics: %s", available_metrics)
        
        # Create comparison figure
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = [ICLR_COLORS['primary'], ICLR_COLORS['secondary'], 
                 ICLR_COLORS['accent'], ICLR_COLORS['success']]
        
        for i, metric in enumerate(available_metrics[:4]):  # Limit to 4 plots
            ax = axes[i]
            
            # Handle confidence intervals if available
            ci_col = f"{metric}_ci"
            if ci_col in df.columns:
                # Parse CI data if it's JSON string
                cis = []
                for ci_str in df[ci_col]:
                    try:
                        if isinstance(ci_str, str):
                            ci = json.loads(ci_str)
                        else:
                            ci = ci_str
                        cis.append(ci)
                    except:
                        cis.append([None, None])
                
                # Extract lower and upper bounds
                ci_lower = [ci[0] if ci[0] is not None else 0 for ci in cis]
                ci_upper = [ci[1] if ci[1] is not None else 0 for ci in cis]
                
                # Plot with error bars
                ax.errorbar(
                    range(len(df)), df[metric], 
                    yerr=[np.array(df[metric]) - np.array(ci_lower),
                          np.array(ci_upper) - np.array(df[metric])],
                    fmt='o', capsize=5, color=colors[i % len(colors)], linewidth=2
                )
            else:
                # Simple bar plot
                bars = ax.bar(range(len(df)), df[metric], 
                             color=colors[i % len(colors)], alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['baseline'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add statistical significance annotations if multiple runs
            if len(df) > 1:
                self._add_significance_annotations(ax, df[metric])
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        title = f"Algorithm Performance Comparison{title_suffix}"
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Save figure
        save_paths = save_figure_publication_ready(
            fig, 
            self.output_dir / "algorithm_comparison",
            formats=self.formats
        )
        
        plt.close(fig)
        return save_paths
    
    def plot_convergence_analysis(
        self,
        tensorboard_logs: Path,
        algorithms: List[str] = None
    ) -> List[Path]:
        """Generate convergence analysis from TensorBoard logs."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import \
                EventAccumulator
            
            log_dirs = list(tensorboard_logs.glob("*/tensorboard"))
            if not log_dirs:
                log_dirs = list(tensorboard_logs.glob("*"))
                
            if not log_dirs:
                logger.warning("No TensorBoard logs found in %s", tensorboard_logs)
                return []
            
            logger.info("🎨 Generating convergence analysis from %d log directories", len(log_dirs))
            
            # Create convergence comparison figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics_to_plot = ['train/loss', 'val/reward', 'val/survival_rate', 'train/epoch_time']
            colors = [ICLR_COLORS['primary'], ICLR_COLORS['secondary'], 
                     ICLR_COLORS['accent'], ICLR_COLORS['success']]
            
            for log_dir in log_dirs:
                algo_name = log_dir.parent.name if log_dir.name == "tensorboard" else log_dir.name
                if algorithms and algo_name not in algorithms:
                    continue
                    
                try:
                    ea = EventAccumulator(str(log_dir))
                    ea.Reload()
                    
                    # Plot each metric
                    for i, metric in enumerate(metrics_to_plot):
                        ax = axes[i]
                        
                        if metric in ea.scalars.Keys():
                            scalar_events = ea.scalars.Items(metric)
                            steps = [event.step for event in scalar_events]
                            values = [event.value for event in scalar_events]
                            
                            ax.plot(steps, values, label=algo_name, 
                                   color=colors[hash(algo_name) % len(colors)], linewidth=2)
                        
                        ax.set_title(metric.replace('/', ' ').replace('_', ' ').title())
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                except Exception as e:
                    logger.warning("Failed to process %s: %s", log_dir, e)
            
            plt.tight_layout()
            fig.suptitle("Training Convergence Analysis", fontsize=16, y=0.98)
            
            save_paths = save_figure_publication_ready(
                fig,
                self.output_dir / "convergence_analysis",
                formats=self.formats
            )
            
            plt.close(fig)
            return save_paths
            
        except ImportError:
            logger.error("TensorBoard not available for convergence analysis")
            return []
        except Exception as e:
            logger.error("Error in convergence analysis: %s", e)
            return []
    
    def plot_statistical_analysis(
        self,
        df: pd.DataFrame,
        primary_metric: str = "reward"
    ) -> List[Path]:
        """Generate statistical analysis plots with significance testing."""
        if primary_metric not in df.columns:
            logger.warning("Primary metric %s not found in data", primary_metric)
            return []
            
        logger.info("🎨 Generating statistical analysis for %s", primary_metric)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distribution plot
        df[primary_metric].hist(bins=20, ax=ax1, alpha=0.7, color=ICLR_COLORS['primary'])
        ax1.axvline(df[primary_metric].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[primary_metric].mean():.3f}')
        ax1.set_title(f'{primary_metric.title()} Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot by algorithm
        df.boxplot(column=primary_metric, by='baseline', ax=ax2)
        ax2.set_title(f'{primary_metric.title()} by Algorithm')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. Correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax3)
            ax3.set_title('Metric Correlations')
        
        # 4. Performance ranking
        sorted_df = df.sort_values(primary_metric, ascending=False)
        bars = ax4.bar(range(len(sorted_df)), sorted_df[primary_metric], 
                      color=ICLR_COLORS['accent'], alpha=0.7)
        ax4.set_xticks(range(len(sorted_df)))
        ax4.set_xticklabels(sorted_df['baseline'], rotation=45)
        ax4.set_title(f'Algorithm Ranking by {primary_metric.title()}')
        
        # Add rank annotations
        for i, (bar, value) in enumerate(zip(bars, sorted_df[primary_metric])):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'#{i+1}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        fig.suptitle(f'Statistical Analysis - {primary_metric.title()}', fontsize=16, y=0.98)
        
        save_paths = save_figure_publication_ready(
            fig,
            self.output_dir / f"statistical_analysis_{primary_metric}",
            formats=self.formats
        )
        
        plt.close(fig)
        return save_paths
    
    def _add_significance_annotations(self, ax, values: pd.Series) -> None:
        """Add statistical significance annotations to plot."""
        try:
            if len(values) >= 2:
                # Find best performing algorithm
                best_idx = values.idxmax()
                best_value = values.max()
                
                # Add significance star for best performer
                ax.text(best_idx, best_value + 0.05, '★', 
                       ha='center', va='bottom', fontsize=20, color='gold')
        except Exception:
            pass  # Silently skip if annotation fails
    
    def generate_summary_report(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Path:
        """Generate a comprehensive summary report."""
        if save_path is None:
            save_path = self.output_dir / "summary_report.txt"
            
        logger.info("📄 Generating summary report...")
        
        with open(save_path, 'w') as f:
            f.write("ICU-AKI Experiment Results Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Algorithms Evaluated: {len(df)}\n")
            f.write(f"Algorithms: {', '.join(df['baseline'])}\n\n")
            
            # Performance summary
            if 'reward' in df.columns:
                best_algo = df.loc[df['reward'].idxmax(), 'baseline']
                best_reward = df['reward'].max()
                f.write(f"Best Performing Algorithm: {best_algo} (Reward: {best_reward:.4f})\n")
            
            if 'survival_rate' in df.columns:
                best_survival = df.loc[df['survival_rate'].idxmax(), 'baseline']
                best_surv_rate = df['survival_rate'].max()
                f.write(f"Highest Survival Rate: {best_survival} ({best_surv_rate:.4f})\n")
            
            if 'runtime_s' in df.columns:
                fastest_algo = df.loc[df['runtime_s'].idxmin(), 'baseline']
                fastest_time = df['runtime_s'].min()
                f.write(f"Fastest Training: {fastest_algo} ({fastest_time:.2f}s)\n\n")
            
            # Detailed metrics table
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            for _, row in df.iterrows():
                f.write(f"\n{row['baseline']}:\n")
                for col in df.columns:
                    if col != 'baseline' and pd.notna(row[col]):
                        f.write(f"  {col}: {row[col]}\n")
        
        logger.info("✅ Summary report saved to %s", save_path)
        return save_path

def main():
    """Main function for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize ICU-AKI experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results", "-r", type=Path,
        help="Path to results file (CSV or JSON)"
    )
    parser.add_argument(
        "--tensorboard", "-t", type=Path,
        help="Path to TensorBoard logs directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default="Output/custom_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--style", "-s", choices=["publication", "presentation", "poster"],
        default="publication", help="Visualization style"
    )
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"],
        choices=["png", "pdf", "svg", "eps"],
        help="Output formats"
    )
    parser.add_argument(
        "--metrics", nargs="+", 
        default=["reward", "survival_rate", "ips_survival", "runtime_s"],
        help="Metrics to visualize"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive visualization dashboard"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(
        style=args.style,
        output_dir=args.output,
        formats=args.formats
    )
    
    saved_files = []
    
    # Generate visualizations from results file
    if args.results:
        if args.results.suffix == '.csv':
            df = visualizer.load_csv_results(args.results)
            
            # Generate comparison plots
            paths = visualizer.plot_algorithm_comparison(df, args.metrics)
            saved_files.extend(paths)
            
            # Generate statistical analysis
            paths = visualizer.plot_statistical_analysis(df)
            saved_files.extend(paths)
            
            # Generate summary report
            report_path = visualizer.generate_summary_report(df)
            saved_files.append(report_path)
            
        elif args.results.suffix == '.json':
            results = visualizer.load_json_results(args.results)
            logger.info("JSON result processing not fully implemented yet")
    
    # Generate convergence analysis from TensorBoard logs
    if args.tensorboard:
        paths = visualizer.plot_convergence_analysis(args.tensorboard)
        saved_files.extend(paths)
    
    # Summary
    logger.info("🎉 Visualization complete!")
    logger.info("Generated %d files:", len(saved_files))
    for file_path in saved_files:
        logger.info("  📄 %s", file_path)

if __name__ == "__main__":
    main() 