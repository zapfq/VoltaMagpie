# VoltaMagpie

**VoltaMagpie** is an experimental Windows upscaling project that combines the **Magpie** capture/presentation framework with a custom **CUDA + WMMA Tensor Core** reconstruction backend for NVIDIA Volta GPUs, especially the **Titan V (SM 7.0)**.

The goal is simple:

> **Put the Titan V's otherwise underused Tensor Cores to useful work for real-time game upscaling, while keeping Magpie's capture, presentation, effect-selection, and workflow intact.**

VoltaMagpie is an independent project and is **not NVIDIA DLSS**, nor is it an official NVIDIA implementation.

---

## Why VoltaMagpie exists

NVIDIA's Titan V is an unusual GPU for modern gaming workloads. It has dedicated Volta Tensor Cores, but many modern upscaling paths either target newer NVIDIA architectures, rely on APIs/features that do not map cleanly to Volta, or do not provide a practical way to make the Titan V's Tensor Cores do useful reconstruction work.

The earlier **VoltaDLSS** work proved that a custom CUDA/WMMA path could execute real Volta Tensor Core instructions on the Titan V. The challenge was turning that proof-of-concept into something practical and reusable for real games.

Magpie provides a strong shell for that job: it already handles window capture, scaling, effect configuration, output presentation, and a large collection of post-processing effects. Magpie also supports combining effects in scaling modes and loading effects from its `effects` directory. [Blinue/Magpie](https://github.com/Blinue/Magpie) [Magpie built-in effects](https://github.com/Blinue/Magpie/wiki/Built-in-effects)

Instead of replacing Magpie, VoltaMagpie adds a specialized Volta CUDA path behind it.

---

## Architecture

```text
                         VoltaMagpie

Game / Window
     │
     ▼
Magpie Graphics Capture
     │
     │  BGRA8 frame
     ▼
VoltaMagpie renderer hook
     │
     ▼
VoltaDLSSWmma.dll
     │
     ├── CUDA 12.9
     ├── EASU-style reconstruction
     ├── WMMA
     └── Volta HMMA / Tensor Cores
     │
     ▼
RGBA8 output texture
     │
     ├── optional Magpie sharpening / AA effects
     │
     ▼
Magpie presentation
     │
     ▼
Display
```

The intended workload is a normal real-time scaling path such as:

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

The compiled `VoltaDLSSWmma.dll` was disassembled with `cuobjdump`, and the generated SASS contained Volta Tensor Core instructions including:

```text
HMMA.884.F16.F16
```

This confirms that the CUDA path is not merely running on ordinary CUDA ALUs: the compiled kernel contains the Volta HMMA Tensor Core instruction path.

### Magpie integration

The VoltaDLSS path was integrated into Magpie's renderer and successfully executed at runtime. The Magpie log reported:

```text
VoltaDLSS v2 active: 1920x1080 -> 1920x1080, CUDA kernel 5.162 ms
```

That particular smoke test used a 1:1 1920×1080 input/output surface, so it is **not** the final 1024×768 → 1080p benchmark. It was used to verify that the CUDA DLL was actually being called by Magpie.

### Selectable Magpie mode

VoltaDLSS was added as a normal entry in Magpie's v4 `scalingModes` configuration, making it selectable alongside existing Magpie modes such as FSR and Lanczos.

---

## Current implementation

The current Volta backend is derived from the earlier VoltaDLSS CUDA/WMMA work and uses an EASU-style reconstruction path with WMMA matrix operations.

The implementation uses:

- **NVIDIA Volta / SM 7.0** as the target architecture
- **CUDA 12.9** for offline compilation
- **Visual Studio 2022 / MSVC 14.44** as the CUDA host toolchain
- **Visual Studio 2026 / MSVC 14.51** for the Magpie C++ build
- **FP16 source data** for high-throughput Tensor Core operation
- WMMA/HMMA Tensor Core execution
- D3D11/CUDA interop
- Magpie's existing capture and presentation path

The deliberate toolchain split is intentional: Magpie builds successfully with VS2026, while the known-good offline CUDA path for Volta uses CUDA 12.9 with VS2022.

---

## Design goals

VoltaMagpie is not trying to reproduce every feature of FSR 2 or NVIDIA DLSS.

The primary target is:

### Maximum useful Tensor Core work per frame

Use WMMA/HMMA for meaningful image reconstruction work instead of treating the Tensor Cores as a hardware curiosity.

### Maximum performance

Reduce avoidable costs around the kernel, especially:

- redundant global-memory reads
- intermediate copies
- unnecessary synchronization
- excessive register/shared-memory pressure
- inefficient WMMA packing

### Maximum image quality

Keep the underlying reconstruction mathematically sound while improving numerical precision where useful and allowing a controlled sharpening stage after reconstruction.

### Practical balance

The target is a **middle point between speed, image quality, and Tensor Core utilization**, rather than an extreme benchmark designed to maximize one metric at the expense of the others.

---

## Optimization roadmap

### Phase 1 — safe kernel optimizations

The first optimization pass focuses on changes that should not alter the reconstruction algorithm:

- cache reusable source pixels
- reuse EASU weights across RGB channels
- unroll fixed-size loops
- use `__restrict__` where valid
- use D3D11/CUDA read-only and write-discard registration hints
- remove redundant device-wide synchronization
- compile the kernel with aggressive optimization settings

### Phase 2 — numerical quality

The preferred balanced profile is:

```text
FP16 input
   ↓
FP32 Tensor Core accumulation
   ↓
RGBA8 output
```

Volta Tensor Cores support mixed-precision FP16 input with FP32 accumulation. The purpose here is to improve accumulation precision without abandoning Tensor Core execution.

This is expected to be a **quality/performance trade-off**, so it will be benchmarked rather than assumed to be faster.

### Phase 3 — remove unnecessary texture copies

The current architecture contains an intermediate CUDA-buffer path around the D3D11 resources.

The longer-term target is:

```text
Current
D3D11 texture
    ↓
CUDA copy
    ↓
CUDA buffer
    ↓
WMMA/EASU
    ↓
CUDA copy
    ↓
D3D11 texture
```

Toward:

```text
Target
D3D11 texture
    ↓
CUDA mapped resource
    ↓
WMMA/EASU
    ↓
D3D11 mapped output
```

This is expected to be one of the most important end-to-end latency optimizations because it attacks memory movement rather than only arithmetic throughput.

### Phase 4 — better WMMA packing

The existing approach demonstrates real HMMA execution, but the matrix formulation is not yet a perfect representation of the underlying EASU dot products.

Future work will investigate more efficient WMMA layouts that reduce wasted matrix work while preserving the reconstruction result.

This is the main research area for maximizing useful Tensor Core utilization.

### Phase 5 — temporal reconstruction research

A lightweight temporal mode may be explored later, using previous reconstructed frames and history rejection/accumulation.

A true FSR 2-style integration is a different class of problem because temporal reconstruction benefits from game-provided motion vectors, depth, and jitter information. A generic Magpie capture path does not normally expose those native render resources.

The goal therefore is **not** to claim FSR 2 equivalence, but to investigate whether a lightweight temporal layer can improve perceived detail without making the generic Magpie path impractical.

---

## Sharpening and post-processing

VoltaMagpie can use Magpie's existing post-processing effects after reconstruction.

Typical test chains include:

```text
VoltaDLSS
    ↓
CAS
```

or, when anti-aliasing is needed:

```text
VoltaDLSS
    ↓
FXAA Medium
    ↓
CAS
```

Magpie documents CAS, AdaptiveSharpen, FXAA, SMAA, FSR_RCAS, and other effects as composable effects with configurable parameters. [Magpie built-in effects](https://github.com/Blinue/Magpie/wiki/Built-in-effects)

The project intentionally keeps sharpening separate from the CUDA reconstruction kernel so that reconstruction performance can be benchmarked independently.

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
GPU utilization
VRAM usage
Temperature
```

Tensor Core usage should be verified using multiple forms of evidence rather than ordinary GPU utilization alone:

1. The CUDA kernel is executing at runtime.
2. The compiled SASS contains Volta HMMA instructions.
3. Profiling/telemetry can be added later for actual HMMA pipeline activity.

A generic `nvidia-smi` GPU utilization percentage is **not** by itself a Tensor Core utilization measurement.

---

## Current status

**Working:**

- Magpie builds successfully with VS2026.
- Volta CUDA backend builds successfully with CUDA 12.9 + VS2022.
- Titan V SM70 WMMA path is proven.
- `VoltaDLSSWmma.dll` builds and loads.
- Magpie can call the CUDA backend at runtime.
- VoltaDLSS can appear as a selectable Magpie scaling mode.
- HMMA Tensor Core instructions are present in the compiled backend.

**Still under development:**

- final kernel optimization
- direct D3D11/CUDA resource path
- improved WMMA packing
- final FP16/FP32 balanced configuration
- integrated Volta telemetry overlay
- systematic quality/performance benchmarking against Magpie FSR/Lanczos and other effects

---

## Build environment

### Magpie

```text
OS:            Windows 10
Visual Studio: 2026 Community
MSVC:          14.51.x
CMake:         Visual Studio 2026 bundled CMake
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

Generated build outputs such as `bin`, `obj`, `.vs`, and restored NuGet packages should normally remain outside version control.

---

## Relationship to Magpie

VoltaMagpie is built around **Blinue's Magpie** project and reuses Magpie's capture, presentation, renderer, effect-selection, and effect infrastructure.

Magpie provides the application shell that makes the project practical; VoltaDLSS supplies the specialized CUDA/WMMA reconstruction backend for Volta hardware.

Magpie's upstream project describes itself as a general-purpose Windows 10/11 window upscaler and is licensed under **GPLv3**. citehttps://github.com/Blinue/Magpie|Magpie repository

When redistributing modified Magpie source, retain the applicable upstream license and attribution notices and comply with the GPLv3 terms.

### Upstream project

**Blinue/Magpie**

https://github.com/Blinue/Magpie

Magpie documentation also covers custom scaling configurations and composable effects, which are central to VoltaMagpie's integration approach. [Magpie scaling configuration documentation](https://github.com/Blinue/Magpie/wiki/Customizing_Scaling_Configurations/66a3d8c9eaeda01119a4bf5a1cb05c2f9badefe5)

---

## Disclaimer

VoltaMagpie is an experimental community project for NVIDIA Volta hardware.

It is not affiliated with, endorsed by, or sponsored by NVIDIA.

The name **VoltaDLSS** refers to this project's custom reconstruction path and should not be interpreted as NVIDIA DLSS technology.

---

## Long-term goal

The long-term goal of VoltaMagpie is straightforward:

> **Make the Titan V's Volta Tensor Cores genuinely useful for real-time game upscaling, with the best practical combination of image quality, latency, and performance that this architecture can deliver.**

The project will favor measured improvements, reproducible benchmarks, and quality-preserving optimizations over simply maximizing a utilization percentage.
