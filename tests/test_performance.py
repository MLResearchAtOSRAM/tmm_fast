"""Opt-in solver benchmarks with CSV and graphical reports.

Run ``python -m pytest tests/test_performance.py --run-performance -s``. Torch measurements keep
the inputs and outputs on the selected device. NumPy measurements include conversion to Torch and,
for CUDA, transfer to and from the GPU. Reports are written to ``tests/render_output``.
"""

import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from tmm_fast import coh_tmm, inc_tmm


BATCH_SIZES = (1, 4, 16, 64)
BACKENDS = ('torch', 'numpy')
DEVICES = ('cpu',) + (('cuda',) if torch.cuda.is_available() else ())
MINIMUM_THROUGHPUT_GAIN = 1.05
WARMUPS = 2
ROUNDS = 5
RENDER_DIR = Path(__file__).resolve().parent / 'render_output'

pytestmark = pytest.mark.performance


@dataclass(frozen=True)
class Scenario:
    name: str
    solver: str
    layers: int
    wavelengths: int
    angles: int
    mask: tuple = ()


@dataclass(frozen=True)
class Measurement:
    scenario: str
    solver: str
    backend: str
    device: str
    layers: int
    wavelengths: int
    angles: int
    batch_size: int
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    stacks_per_second: float
    optical_points_per_second: float
    samples_seconds: tuple


SCENARIOS = (
    Scenario('coherent-balanced', 'coherent', layers=8, wavelengths=64, angles=16),
    Scenario('coherent-spectral', 'coherent', layers=8, wavelengths=256, angles=1),
    Scenario('coherent-angular', 'coherent', layers=8, wavelengths=1, angles=128),
    Scenario('coherent-deep', 'coherent', layers=20, wavelengths=32, angles=8),
    Scenario('incoherent', 'incoherent', layers=8, wavelengths=64, angles=16),
    Scenario('mixed-coherence', 'incoherent', layers=12, wavelengths=64, angles=16,
             mask=((1, 2), (4, 5), (8, 9))),
)

RESULTS = []


@pytest.fixture(scope='module', autouse=True)
def performance_session():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    RESULTS.clear()
    yield
    torch.set_num_threads(previous_threads)
    if RESULTS:
        write_reports(RESULTS)


@pytest.fixture(params=SCENARIOS, ids=lambda scenario: scenario.name)
def scenario(request):
    return request.param


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


def stack(scenario, batch_size, backend, device):
    build_device = device if backend == 'torch' else torch.device('cpu')
    wavelengths = torch.linspace(
        400e-9, 1200e-9, scenario.wavelengths, dtype=torch.float64, device=build_device
    )
    angles = torch.linspace(
        0, np.deg2rad(75), scenario.angles, dtype=torch.float64, device=build_device
    )

    inner_indices = (1.45 + .002j, 2.2 + .001j, 1.7 + .003j,
                     2.4 + .001j, 1.3 + .002j, 1.9 + .001j)
    indices = [1.0]
    indices.extend(inner_indices[i % len(inner_indices)] for i in range(scenario.layers - 2))
    indices.append(1.52)
    indices = torch.tensor(indices, dtype=torch.complex128, device=build_device)
    N = indices[None, :, None].repeat(batch_size, 1, scenario.wavelengths)

    inner_thickness = torch.linspace(
        80e-9, 180e-9, scenario.layers - 2, dtype=torch.float64, device=build_device
    )
    thickness = torch.cat([
        torch.tensor([np.inf], dtype=torch.float64, device=build_device),
        inner_thickness,
        torch.tensor([np.inf], dtype=torch.float64, device=build_device),
    ])
    D = thickness[None].repeat(batch_size, 1)
    scaling = (torch.ones(1, dtype=torch.float64, device=build_device) if batch_size == 1 else
               torch.linspace(.95, 1.05, batch_size, dtype=torch.float64,
                              device=build_device))
    D[:, 1:-1] *= scaling[:, None]

    if scenario.solver == 'incoherent':
        coherent = {layer for group in scenario.mask for layer in group}
        for layer in range(1, D.shape[1] - 1):
            if layer not in coherent:
                D[:, layer] *= 20

    values = (N, D, angles, wavelengths)
    if backend == 'numpy':
        return tuple(value.numpy() for value in values)
    return values


def solver_call(scenario, backend, device, inputs):
    N, D, angles, wavelengths = inputs
    device_argument = device if backend == 'numpy' else None
    if scenario.solver == 'coherent':
        return lambda: coh_tmm('s', N, D, angles, wavelengths, device=device_argument)
    mask = [list(group) for group in scenario.mask]
    return lambda: inc_tmm('s', N, D, mask, angles, wavelengths, device=device_argument)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def runtimes(call, device):
    for _ in range(WARMUPS):
        call()
    synchronize(device)

    samples = []
    for _ in range(ROUNDS):
        synchronize(device)
        started = time.perf_counter()
        call()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return samples


def check_result(result, scenario, batch_size, backend, device):
    expected_shape = (batch_size, scenario.angles, scenario.wavelengths)
    assert result['R'].shape == expected_shape
    if backend == 'numpy':
        assert isinstance(result['R'], np.ndarray)
    else:
        assert isinstance(result['R'], torch.Tensor)
        assert result['R'].device == device


def measure(scenario, backend, device, batch_size):
    inputs = stack(scenario, batch_size, backend, device)
    call = solver_call(scenario, backend, device, inputs)
    check_result(call(), scenario, batch_size, backend, device)
    samples = runtimes(call, device)
    median = statistics.median(samples)
    optical_points = batch_size * scenario.wavelengths * scenario.angles
    return Measurement(
        scenario=scenario.name,
        solver=scenario.solver,
        backend=backend,
        device=device.type,
        layers=scenario.layers,
        wavelengths=scenario.wavelengths,
        angles=scenario.angles,
        batch_size=batch_size,
        median_seconds=median,
        minimum_seconds=min(samples),
        maximum_seconds=max(samples),
        stacks_per_second=batch_size / median,
        optical_points_per_second=optical_points / median,
        samples_seconds=tuple(samples),
    )


def test_batch_throughput(scenario, backend, device):
    measurements = []
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            measurement = measure(scenario, backend, device, batch_size)
            RESULTS.append(measurement)
            measurements.append(measurement)

    print(f'\n{scenario.name}, {backend} inputs on {device}:')
    for result in measurements:
        print(f'  batch {result.batch_size:>2}: '
              f'{result.median_seconds * 1e3:>9.3f} ms '
              f'[{result.minimum_seconds * 1e3:.3f}, '
              f'{result.maximum_seconds * 1e3:.3f}], '
              f'{result.stacks_per_second:>9.1f} stacks/s')

    single = measurements[0].stacks_per_second
    batched = measurements[-1].stacks_per_second
    assert batched >= single * MINIMUM_THROUGHPUT_GAIN, (
        f'{scenario.name} with {backend} inputs on {device} did not improve throughput by at '
        f'least {MINIMUM_THROUGHPUT_GAIN - 1:.0%}: {single:.1f} stacks/s at batch 1, '
        f'{batched:.1f} stacks/s at batch {BATCH_SIZES[-1]}'
    )


def test_gpu_speedup_is_reported():
    if 'cuda' not in DEVICES:
        pytest.skip('CUDA is not available')

    indexed = {(result.scenario, result.backend, result.device, result.batch_size): result
               for result in RESULTS}
    speedups = []
    print('\nGPU speedup (CPU median / GPU median):')
    for scenario in SCENARIOS:
        for backend in BACKENDS:
            values = []
            for batch_size in BATCH_SIZES:
                cpu = indexed[(scenario.name, backend, 'cpu', batch_size)]
                gpu = indexed[(scenario.name, backend, 'cuda', batch_size)]
                values.append(cpu.median_seconds / gpu.median_seconds)
            speedups.extend(values)
            formatted = ', '.join(f'batch {batch}: {speedup:.2f}x'
                                  for batch, speedup in zip(BATCH_SIZES, values))
            print(f'  {scenario.name}, {backend}: {formatted}')
    assert np.isfinite(speedups).all() and np.all(np.asarray(speedups) > 0)


def write_reports(results):
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RENDER_DIR / 'performance_results.csv'
    summaries = []
    for result in results:
        summary = asdict(result)
        summary.pop('samples_seconds')
        summaries.append(summary)
    fieldnames = list(summaries[0])
    with csv_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    raw_path = RENDER_DIR / 'performance_samples.csv'
    raw_fieldnames = ['scenario', 'solver', 'backend', 'device', 'batch_size',
                      'round', 'seconds']
    with raw_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=raw_fieldnames)
        writer.writeheader()
        for result in results:
            for round_number, seconds in enumerate(result.samples_seconds, 1):
                writer.writerow({
                    'scenario': result.scenario,
                    'solver': result.solver,
                    'backend': result.backend,
                    'device': result.device,
                    'batch_size': result.batch_size,
                    'round': round_number,
                    'seconds': seconds,
                })

    plot_measurements(results, 'median_seconds', 'Median runtime [ms]',
                      'performance_runtime.png', scale=1e3)
    plot_measurements(results, 'stacks_per_second', 'Throughput [stacks s⁻¹]',
                      'performance_throughput.png')
    plot_ratio(results, numerator='numpy', denominator='torch', devices=DEVICES,
               ylabel='Runtime ratio, NumPy/Torch []',
               filename='performance_numpy_overhead.png')
    plot_gpu_speedup(results)
    print(f'\nPerformance reports written to {RENDER_DIR}')


def subplot_grid():
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    return figure, axes.flatten()


def finish_plot(figure, axes, ylabel, filename):
    for axis in axes:
        axis.set_xscale('log', base=2)
        axis.set_xticks(BATCH_SIZES)
        axis.set_xticklabels(BATCH_SIZES)
        axis.grid(True, which='both', alpha=.25)
        axis.set_xlabel('Batch size [stacks]')
        axis.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout(rect=(0, .08, 1, 1))
    if handles:
        figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .01),
                      ncol=max(1, len(labels)))
    figure.savefig(RENDER_DIR / filename, dpi=140)
    plt.close(figure)


def scenario_title(scenario):
    wavelength_range = '400 nm' if scenario.wavelengths == 1 else '400–1200 nm'
    angle_range = '0°' if scenario.angles == 1 else '0–75°'
    return (f'{scenario.name}\n{scenario.layers} layers · '
            f'{scenario.wavelengths} λ points [{wavelength_range}]\n'
            f'{scenario.angles} θ points [{angle_range}]')


def plot_measurements(results, field, ylabel, filename, scale=1):
    figure, axes = subplot_grid()
    for axis, scenario in zip(axes, SCENARIOS):
        selected = [result for result in results if result.scenario == scenario.name]
        for backend in BACKENDS:
            for device in DEVICES:
                line = sorted(
                    (result for result in selected
                     if result.backend == backend and result.device == device),
                    key=lambda result: result.batch_size,
                )
                axis.plot([result.batch_size for result in line],
                          [getattr(result, field) * scale for result in line], marker='o',
                          label=f'{backend}/{device}')
        axis.set_title(scenario_title(scenario), fontsize=10)
        axis.set_yscale('log')
    finish_plot(figure, axes, ylabel, filename)


def plot_ratio(results, numerator, denominator, devices, ylabel, filename):
    indexed = {(result.scenario, result.backend, result.device, result.batch_size): result
               for result in results}
    figure, axes = subplot_grid()
    for axis, scenario in zip(axes, SCENARIOS):
        for device in devices:
            ratios = []
            for batch_size in BATCH_SIZES:
                top = indexed[(scenario.name, numerator, device, batch_size)]
                bottom = indexed[(scenario.name, denominator, device, batch_size)]
                ratios.append(top.median_seconds / bottom.median_seconds)
            axis.plot(BATCH_SIZES, ratios, marker='o', label=device)
        axis.axhline(1, color='black', linewidth=1, linestyle='--')
        axis.set_title(scenario_title(scenario), fontsize=10)
    finish_plot(figure, axes, ylabel, filename)


def plot_gpu_speedup(results):
    figure, axes = subplot_grid()
    if 'cuda' not in DEVICES:
        for axis in axes:
            axis.axis('off')
        figure.text(.5, .5, 'CUDA was not available during this benchmark run',
                    ha='center', va='center', fontsize=16)
        figure.savefig(RENDER_DIR / 'performance_gpu_speedup.png', dpi=140)
        plt.close(figure)
        return

    indexed = {(result.scenario, result.backend, result.device, result.batch_size): result
               for result in results}
    for axis, scenario in zip(axes, SCENARIOS):
        for backend in BACKENDS:
            speedups = []
            for batch_size in BATCH_SIZES:
                cpu = indexed[(scenario.name, backend, 'cpu', batch_size)]
                gpu = indexed[(scenario.name, backend, 'cuda', batch_size)]
                speedups.append(cpu.median_seconds / gpu.median_seconds)
            axis.plot(BATCH_SIZES, speedups, marker='o', label=backend)
        axis.axhline(1, color='black', linewidth=1, linestyle='--')
        axis.set_title(scenario_title(scenario), fontsize=10)
    finish_plot(figure, axes, 'Speedup, CPU/GPU runtime [×]',
                'performance_gpu_speedup.png')
