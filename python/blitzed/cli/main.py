# Copyright 2025 Gibran Rodriguez <brangi000@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main CLI entry point for Blitzed.

Provides commands for:
- optimize: Optimize models for edge deployment  
- convert: Convert between model formats
- profile: Profile model performance
- deploy: Generate deployment code for target hardware
"""

import click
import os
import sys
from typing import Optional

from ..optimization import Optimizer, OptimizationConfig, QuantizationConfig, QuantizationType


@click.group()
@click.version_option()
def cli():
    """
    Blitzed - High-performance edge AI optimization framework.
    
    Optimize and deploy machine learning models on edge devices including
    microcontrollers, mobile devices, and embedded systems.
    """
    pass


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output path for optimized model')
@click.option('--target', '-t', 
              type=click.Choice(['esp32', 'arduino', 'stm32', 'mobile', 'generic']),
              default='generic',
              help='Target hardware platform')
@click.option('--quantization', 
              type=click.Choice(['int8', 'int4', 'binary', 'mixed']),
              default='int8',
              help='Quantization type')
@click.option('--max-accuracy-loss', type=float, default=5.0,
              help='Maximum allowed accuracy loss percentage')
@click.option('--target-compression', type=float, default=0.75,
              help='Target compression ratio (0-1)')
@click.option('--pruning/--no-pruning', default=False,
              help='Enable pruning optimization')
@click.option('--distillation/--no-distillation', default=False,
              help='Enable knowledge distillation')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def optimize(model_path: str, output: Optional[str], target: str,
             quantization: str, max_accuracy_loss: float,
             target_compression: float, pruning: bool, 
             distillation: bool, verbose: bool):
    """
    Optimize a model for edge deployment.
    
    This command applies various optimization techniques like quantization,
    pruning, and knowledge distillation to reduce model size and improve
    inference speed while maintaining accuracy.
    
    Example:
        blitzed optimize model.onnx --target esp32 --quantization int8
    """
    if verbose:
        click.echo(f"Optimizing {model_path} for {target} target...")
    
    try:
        # Configure quantization
        quant_type = {
            'int8': QuantizationType.INT8,
            'int4': QuantizationType.INT4,
            'binary': QuantizationType.BINARY,
            'mixed': QuantizationType.MIXED,
        }[quantization]
        
        quant_config = QuantizationConfig(
            quantization_type=quant_type,
            accuracy_threshold=max_accuracy_loss
        )
        
        # Configure optimization
        opt_config = OptimizationConfig(
            quantization=quant_config,
            pruning_enabled=pruning,
            distillation_enabled=distillation,
            target_compression_ratio=target_compression,
            max_accuracy_loss=max_accuracy_loss
        )
        
        # Run optimization
        optimizer = Optimizer(opt_config)
        result = optimizer.optimize(model_path, output, target)
        
        click.echo("Optimization completed successfully!")
        click.echo(result.summary())
        
        if not result.meets_target:
            click.echo("Warning: Optimization did not meet all target criteria", err=True)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--target', '-t',
              type=click.Choice(['esp32', 'arduino', 'stm32', 'mobile']),
              help='Target hardware platform')
@click.option('--runs', type=int, default=100,
              help='Number of benchmark runs')
@click.option('--warmup', type=int, default=10,
              help='Number of warmup runs')
def profile(model_path: str, target: Optional[str], runs: int, warmup: int):
    """
    Profile model performance characteristics.
    
    Measures inference time, memory usage, and throughput for the given
    model on the specified target hardware.
    
    Example:
        blitzed profile model.onnx --target esp32 --runs 50
    """
    try:
        optimizer = Optimizer()
        results = optimizer.profile(model_path, target)
        
        click.echo("Model Performance Profile")
        click.echo("=" * 25)
        click.echo(f"Model size: {results['model_size_bytes'] / (1024*1024):.1f} MB")
        click.echo(f"Estimated memory usage: {results['estimated_memory_usage'] / 1024:.1f} KB")
        click.echo(f"Estimated inference time: {results['estimated_inference_time_ms']} ms")
        click.echo(f"Estimated throughput: {results['estimated_throughput']:.1f} inferences/sec")
        
        if target:
            click.echo(f"Target: {target}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--target', '-t',
              type=click.Choice(['esp32', 'arduino', 'stm32', 'mobile']),
              help='Target hardware platform')
def recommend(model_path: str, target: Optional[str]):
    """
    Get optimization recommendations for a model.
    
    Analyzes the model and target hardware to provide specific
    recommendations for optimization strategies.
    
    Example:
        blitzed recommend large_model.onnx --target arduino
    """
    try:
        optimizer = Optimizer()
        recommendations = optimizer.recommend(model_path, target)
        
        click.echo("Optimization Recommendations")
        click.echo("=" * 28)
        for i, rec in enumerate(recommendations, 1):
            click.echo(f"{i}. {rec}")
        
        # Also show estimated impact
        impact = optimizer.estimate_impact(model_path, target)
        click.echo("\nEstimated Impact:")
        click.echo(f"  Size reduction: {impact['size_reduction']*100:.1f}%")
        click.echo(f"  Speed improvement: {impact['speed_improvement']:.1f}x")
        click.echo(f"  Accuracy loss: {impact['accuracy_loss']:.1f}%")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--target', '-t',
              type=click.Choice(['esp32', 'arduino', 'stm32', 'c']),
              required=True,
              help='Target platform for code generation')
@click.option('--example/--no-example', default=True,
              help='Generate example usage code')
@click.option('--build-config/--no-build-config', default=True,
              help='Generate build configuration files')
def deploy(model_path: str, output_dir: str, target: str, 
           example: bool, build_config: bool):
    """
    Generate deployment code for target hardware.
    
    Creates optimized C/C++ code that can be compiled and deployed
    on the target hardware platform.
    
    Example:
        blitzed deploy optimized_model.onnx ./output --target esp32
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        click.echo(f"Generating {target} deployment code...")
        click.echo(f"Output directory: {output_dir}")
        
        try:
            # Use the actual Rust core for code generation
            from .._core import generate_deployment_code
            
            result = generate_deployment_code(model_path, output_dir, target)
            
            click.echo("Generated files:")
            click.echo(f"  📄 {result['implementation_file']}")
            if 'header_file' in result and result['header_file']:
                click.echo(f"  📄 {result['header_file']}")
            if 'example_file' in result and result['example_file']:
                click.echo(f"  📄 {result['example_file']}")
            if 'build_config' in result and result['build_config']:
                click.echo(f"  📄 {result['build_config']}")
            
            if result.get('dependencies'):
                click.echo(f"\nRequired dependencies: {', '.join(result['dependencies'])}")
            
            click.echo(f"\n✅ Deployment code generation completed!")
            click.echo(f"Next steps:")
            click.echo(f"  1. Review generated code in {output_dir}")
            click.echo(f"  2. Install dependencies: {', '.join(result.get('dependencies', ['gcc', 'make']))}")
            click.echo(f"  3. Compile: cd {output_dir} && make")
            click.echo(f"  4. Deploy to {target} device")
            
        except ImportError:
            # Fallback if Rust core not available
            click.echo("⚠️  Rust core not available, generating placeholder files...")
            files = {
                'esp32': ['blitzed_model.c', 'blitzed_model.h', 'Makefile'],
                'arduino': ['blitzed_model.c', 'blitzed_model.h', 'Makefile'],  
                'stm32': ['blitzed_model.c', 'blitzed_model.h', 'Makefile'],
                'c': ['blitzed_model.c', 'blitzed_model.h', 'Makefile'],
            }
            
            click.echo("Generated placeholder files:")
            for file in files.get(target, ['blitzed_model.c', 'blitzed_model.h', 'Makefile']):
                file_path = os.path.join(output_dir, file)
                click.echo(f"  📄 {file_path}")
                
                with open(file_path, 'w') as f:
                    f.write(f"// Generated by Blitzed for {target}\n")
                    f.write(f"// TODO: Implement {file} - Install Blitzed Rust core for full functionality\n")
            
            click.echo(f"\n⚠️  Placeholder generation completed!")
            click.echo(f"Install Blitzed Rust core for full code generation functionality")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """
    Show Blitzed installation and system information.
    """
    import platform
    from .. import __version__
    
    click.echo("Blitzed Information")
    click.echo("=" * 18)
    click.echo(f"Version: {__version__}")
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.system()} {platform.machine()}")
    
    # Check for core extension
    try:
        from .._core import VERSION as core_version
        click.echo(f"Core extension: v{core_version}")
    except ImportError:
        click.echo("Core extension: Not available")
    
    # Show supported targets
    click.echo("\nSupported targets:")
    targets = ['esp32', 'arduino', 'stm32', 'mobile', 'generic']
    for target in targets:
        click.echo(f"  - {target}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()