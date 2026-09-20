# VoltaMagpie

**VoltaMagpie** is an experimental Windows upscaling project that combines the **Magpie** capture/presentation framework with a custom **CUDA + WMMA Tensor Core** reconstruction backend for NVIDIA Volta GPUs, especially the **Titan V (SM 7.0)**.

The goal is simple:

> **Put the Titan V's otherwise underused Tensor Cores to useful work for real-time game upscaling, while keeping Magpie's capture, presentation, effect-selection, and workflow intact.**

VoltaMagpie is an independent project and is **not NVIDIA DLSS**, nor is it an official NVIDIA implementation.

---

## Why VoltaMagpie exists

NVIDIA's Titan V is an unusual GPU for modern gaming workloads. It has dedicated Volta Tensor Cores, but many modern upscaling paths either target newer NVIDIA architectures, rely on APIs/features that do not map cleanly to Volta, or do not provide a practical way to make the Titan V's Tensor Cores do useful reconstruction work.

The earlier **VoltaDLSS** work proved that a custom CUDA/WMMA path could execute real Volta Tensor Core instructions on the Titan V. The challenge was turning that proof-of-concept into something practical and reusable for real games.

Magpie provides a strong shell for that job: it already handles window capture, scaling, effect configuration, output presentation, and a large collection of post-processing effects. Magpie also supports combining effects in scaling modes and loading effects from its `effects` directory.

Instead of replacing Magpie, VoltaMagpie adds a specialized Volta CUDA reconstruction path behind it.

---

## Architecture

```text
                         VoltaMagpie

Game / Window
     │
     ▼
Magpie Graphics Capture
     │
     │  D3D11 BGRA/RGBA texture
     ▼
VoltaMagpie renderer hook
     │
     ▼
VoltaDLSSWmma.dll
     │
     ├── CUDA 12.9
     ├── EASU-style reconstruction
     ├── FP16 WMMA input
     ├── FP32 accumulation
     └── Volta HMMA / Tensor Cores
     │
     ▼
CUDA surface output
     │
     ├── optional Magpie sharpening / AA effects
     │
     ▼
Magpie presentation
     │
     ▼
Display
```

The current direct CUDA path is designed around mapped D3D11 texture arrays and CUDA surface objects rather than copying every frame through intermediate linear CUDA buffers.

A representative real-time workload is:

```text
1024×768
   ↓
VoltaMagpie
   ↓
1920×1080
```

Magpie remains responsible for capture and presentation; VoltaMagpie supplies the Volta-specific reconstruction backend.

---

## What is already proven

### Titan V / Volta Tensor Core support

A standalone WMMA test was successfully run on a **NVIDIA TITAN V** at **compute capability 7.0**.

The test reported:

```text
GPU: NVIDIA TITAN V
Compute capability: 7.0
WMMA 16x16x16 test: PASS
Tensor Core path is executing successfully.
```

### Real Tensor Core instructions

The compiled `VoltaDLSSWmma.dll` was disassembled with `cuobjdump`, and the generated SASS contains Volta Tensor Core instructions including:

```text
HMMA.884.F32.F32
```

This confirms that the active CUDA reconstruction kernel is compiled to use Volta HMMA Tensor Core instructions rather than relying entirely on ordinary CUDA ALUs.

### Magpie integration

The VoltaDLSS path was integrated into Magpie's renderer and successfully executed at runtime.

A previous 1:1 smoke test reported:

```text
VoltaDLSS v2 active: 1920x1080 -> 1920x1080, CUDA kernel 5.162 ms
```

That test was used only to verify that Magpie was successfully calling the CUDA backend.

The current optimization benchmark uses:

```text
1024x768 -> 1920x1080
```

and therefore should be treated separately from the original 1:1 smoke test.

### Selectable Magpie mode

VoltaDLSS was added as a selectable Magpie scaling mode, allowing it to be used alongside existing Magpie scaling configurations and post-processing effects.

---

## Current implementation

The current Volta backend is derived from the earlier VoltaDLSS CUDA/WMMA work and uses an EASU-style reconstruction path with WMMA matrix operations.

The implementation uses:

* **NVIDIA Volta / SM 7.0** as the target architecture
* **CUDA 12.9** for offline compilation
* **Visual Studio 2022 / MSVC 14.44** as the CUDA host toolchain
* **Visual Studio 2026 / MSVC 14.51** for the Magpie C++ build
* FP16 data for Tensor Core input
* FP32 accumulation
* WMMA / HMMA Tensor Core execution
* D3D11/CUDA interop
* CUDA surface objects
* Magpie's existing capture and presentation path

The deliberate toolchain split is intentional: Magpie builds successfully with the newer Visual Studio toolchain, while the known-good offline CUDA build for the Volta backend uses CUDA 12.9 with the VS2022 host compiler.

---

## Design goals

VoltaMagpie is not trying to reproduce every feature of FSR 2 or NVIDIA DLSS.

The primary target is:

### Maximum useful Tensor Core work per frame

Use WMMA/HMMA for meaningful image reconstruction work instead of treating the Tensor Cores as a hardware curiosity.

### Maximum practical performance

Reduce avoidable costs around the kernel, especially:

* redundant global-memory reads
* intermediate copies
* unnecessary synchronization
* excessive register/shared-memory pressure
* inefficient matrix packing
* unnecessary initialization work

### Maximum image quality

Keep the underlying reconstruction mathematically sound while using higher-precision accumulation where useful and allowing controlled sharpening after reconstruction.

### Practical balance

The target is a **middle point between speed, image quality, and Tensor Core utilization**, rather than an extreme benchmark designed to maximize one metric at the expense of the others.

---

## Reconstruction quality

The current spatial reconstruction path intentionally preserves the EASU-style algorithm rather than replacing it with a lower-quality approximation.

The current implementation includes:

* Exact 12-tap EASU sampling positions.
* Original EASU direction and weighting calculations.
* FP16 WMMA inputs.
* FP32 accumulation.
* Source-pixel caching.
* EASU weight reuse across RGB processing.
* Output clamping.
* Direct RGBA8 surface output.

The current spatial path does **not** introduce a temporal approximation simply to obtain better benchmark numbers.

The goal is to keep reconstruction quality stable while optimizing the implementation underneath it.

---

## Optimization progress

The main spatial CUDA optimization pass has now been completed.

### Completed optimizations

#### FP16 input + FP32 accumulation

The reconstruction uses FP16 input data for Tensor Core throughput while accumulating into FP32.

The intended balance is:

```text
FP16 input
   ↓
Volta HMMA
   ↓
FP32 accumulation
   ↓
RGBA8 output
```

This preserves higher accumulation precision without abandoning Volta Tensor Core execution.

#### Source-pixel caching

Required EASU source pixels are cached in shared memory so the same source pixel is not repeatedly fetched for every RGB channel.

#### EASU weight reuse

EASU weights are generated once per output pixel and reused while processing the RGB channels.

#### Compiler and loop tuning

The fixed-size EASU loops use compiler-directed unrolling and optimization settings appropriate for the static workload.

#### CUDA interop optimization

The D3D11 resources use the appropriate CUDA graphics registration flags and mapping hints.

The implementation avoids per-frame resource registration when the underlying texture remains unchanged.

#### Removed unnecessary device-wide synchronization

Redundant device-wide synchronization was removed from the normal execution path where it was not required.

#### Direct D3D11 → CUDA mapped-array path

The current direct path maps the D3D11 input and output textures into CUDA as arrays and uses CUDA surface objects directly.

The target path is therefore:

```text
D3D11 input texture
       ↓
CUDA mapped array
       ↓
CUDA surface read
       ↓
EASU / HMMA
       ↓
CUDA surface write
       ↓
D3D11 output texture
```

rather than requiring a separate CUDA linear-buffer copy for every frame.

#### Reduced shared-memory initialization

The kernel no longer initializes the entire result tile before every operation when that storage is guaranteed to be overwritten before use.

Only required WMMA padding is initialized.

#### Warp-level synchronization

The direct kernel operates as one warp of 32 threads.

Required block-wide synchronization points were changed to:

```cpp
__syncwarp();
```

where the synchronization domain is limited to the active warp.

This produced the best observed kernel result during the optimization pass.

---

## Tested and rejected optimizations

Several changes were implemented and benchmarked rather than assumed to be beneficial.

### WMMA `half2` packing

An attempted paired `half2` packing implementation was tested.

Observed result:

```text
Before:
~1.0 ms class

Packed version:
1.315 ms
```

The additional packing/control overhead outweighed the intended memory benefit, so the change was reverted.

### Persistent asynchronous CUDA stream/event pipeline

A persistent non-blocking CUDA stream with reusable CUDA events was tested.

Observed result:

```text
Async version:
1.341 ms
```

against a roughly 1.0 ms class optimized kernel result.

The change was therefore reverted.

### Removing the first required warp synchronization

The first `__syncwarp()` in the direct kernel was experimentally removed.

Observed result:

```text
~0.8–1.0 ms class
        ↓
3.683 ms
```

The synchronization point was restored.

This demonstrates that the synchronization is required for the current shared-memory/WMMA data flow and should not be removed merely because the kernel uses one warp.

### Manual output-store rewrite

The final direct output path was inspected at the SASS level.

The compiler already emits a direct surface store for the final `uchar4` output. No additional manual output-store rewrite was retained.

---

## Current benchmark

Primary development benchmark:

```text
GPU:            NVIDIA TITAN V
Input:          1024 × 768
Output:         1920 × 1080
Backend:        VoltaDLSSWmma
Path:           Direct D3D11 → CUDA surface path
Effect:         VoltaDLSS only
```

Best observed CUDA kernel measurement during development:

```text
0.821 ms
```

This is a **best observed single measurement**, not a guaranteed average.

Repeated launches showed meaningful timing variation, so future performance comparisons should use repeated samples and a statistical summary such as median rather than comparing isolated fastest/slowest launches.

---

## Benchmarking philosophy

Performance numbers should be treated as measurements, not assumptions.

The most useful metrics are:

```text
Input resolution
Output resolution
CUDA kernel time
Total Magpie frame time
FPS / presentation rate
Frame pacing
GPU utilization
VRAM usage
Temperature
```

Tensor Core usage should be verified using multiple forms of evidence:

1. The CUDA backend executes successfully at runtime.
2. The compiled SASS contains Volta HMMA instructions.
3. Profiling/telemetry can later be used to measure actual Tensor Core pipeline activity.

Generic GPU utilization alone is **not** a Tensor Core utilization measurement.

---

## Sharpening and post-processing

VoltaMagpie keeps sharpening and anti-aliasing separate from the CUDA reconstruction kernel.

Typical chains include:

```text
VoltaDLSS
    ↓
CAS
```

or:

```text
VoltaDLSS
    ↓
FXAA
    ↓
CAS
```

Magpie's existing post-processing effects can therefore be used without embedding sharpening logic into the CUDA reconstruction kernel.

This separation makes it easier to benchmark reconstruction independently from sharpening and anti-aliasing.

---

## Current status

### Working

* Magpie builds successfully with the current development toolchain.
* Volta CUDA backend builds successfully with CUDA 12.9 + VS2022.
* Titan V SM 7.0 WMMA execution is proven.
* `VoltaDLSSWmma.dll` builds successfully.
* Magpie successfully loads and invokes the CUDA backend.
* VoltaDLSS is available as a selectable scaling mode.
* Direct D3D11 → CUDA mapped-array access is functional.
* CUDA surface objects are used for direct texture reads/writes.
* The direct reconstruction kernel contains real Volta HMMA instructions.
* The spatial EASU optimization pass is complete.

### Current optimized path

```text
D3D11 capture
    ↓
Mapped CUDA array
    ↓
CUDA surface reads
    ↓
EASU weight generation
    ↓
Source-pixel cache
    ↓
FP16 WMMA
    ↓
FP32 HMMA accumulation
    ↓
Clamp / convert
    ↓
CUDA surface write
    ↓
Magpie presentation
```

### Development status

The current CUDA kernel is intentionally frozen after the spatial optimization pass.

Further changes should only be accepted when they demonstrate a reproducible improvement while preserving reconstruction quality.

---

## Remaining work

The main low-level spatial optimization pass is complete.

Remaining work is now primarily validation and higher-level reconstruction research.

### Real-world validation

* Real game testing.
* Different game engines and rendering APIs.
* Image-quality comparison against existing Magpie scaling methods.
* Frame-time and frame-pacing measurements.
* Long-duration stability testing.
* Different input/output resolutions.

### Tensor Core telemetry

Future diagnostics can expose:

```text
CUDA kernel time
HMMA activity
GPU utilization
VRAM usage
Frame time
```

This is useful for measuring how effectively the Titan V's Tensor Cores are being used in real workloads.

### Temporal reconstruction

A temporal mode may be explored as a separate future architecture.

A true FSR 2-style implementation is substantially more complicated because temporal reconstruction benefits from:

* motion vectors
* depth
* jitter information
* game-specific render targets
* history rejection and accumulation

A generic Magpie capture path does not normally expose those native game resources.

The long-term goal is therefore not to claim FSR 2 equivalence, but to investigate whether a lightweight temporal layer can improve perceived detail while remaining practical for a generic Magpie workflow.

---

## Known limitations

* The direct CUDA backend currently targets the single-effect `VoltaDLSS` profile.
* Chaining `VoltaDLSS` with additional effects currently falls back to the normal Magpie HLSL path.
* Temporal reconstruction is not implemented.
* Tensor Core utilization telemetry is not yet implemented.
* Broader compatibility testing is still required.
* The current benchmark is a CUDA kernel measurement and does not by itself represent total end-to-end frame latency.

---

## Build environment

### Magpie

```text
OS:            Windows 10
Visual Studio: 2026 Community
MSVC:          14.51.x
CMake:         Visual Studio bundled CMake
```

### Volta CUDA backend

```text
GPU:           NVIDIA TITAN V
Architecture:  SM 7.0 / Volta
CUDA:          12.9
Host compiler: VS2022 Community / MSVC 14.44.35207
```

### Main paths

```text
C:\Magpie-WMMA\
├── src\
│   ├── Magpie.Core\
│   └── Effects\
├── VoltaDLSSWmma\
└── bin\x64\Release\
```

Generated build outputs such as `bin`, `obj`, `.vs`, intermediate compiler files, and temporary optimization backups should normally remain outside version control.

---

## Relationship to Magpie

VoltaMagpie is built around **Blinue's Magpie** project and reuses Magpie's capture, presentation, renderer, effect-selection, and effect infrastructure.

Magpie provides the application shell that makes the project practical; VoltaDLSS supplies the specialized CUDA/WMMA reconstruction backend for Volta hardware.

Magpie's upstream project:

https://github.com/Blinue/Magpie

Magpie documentation:

* [Magpie built-in effects](https://github.com/Blinue/Magpie/wiki/Built-in-effects)
* [Magpie custom scaling configurations](https://github.com/Blinue/Magpie/wiki/Customizing_Scaling_Configurations/66a3d8c9eaeda01119a4bf5a1cb05c2f9badefe5)

When redistributing modified Magpie source, retain the applicable upstream license and attribution notices and comply with the GPLv3 terms.

---

## Disclaimer

VoltaMagpie is an experimental community project for NVIDIA Volta hardware.

It is not affiliated with, endorsed by, or sponsored by NVIDIA.

The name **VoltaDLSS** refers to this project's custom reconstruction path and should not be interpreted as NVIDIA DLSS technology.

---

## Long-term goal

The long-term goal of VoltaMagpie is straightforward:

> **Make the Titan V's Volta Tensor Cores genuinely useful for real-time game upscaling, with the best practical combination of image quality, latency, and performance that this architecture can deliver.**

The project favors measured improvements, reproducible benchmarks, and quality-preserving optimizations over simply maximizing a utilization percentage.
