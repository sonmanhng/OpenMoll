# OpenMoll

A modern, fast, and feature-rich 3D molecular visualization and drug discovery workspace built in Rust and Bevy. OpenMoll enables researchers to visualize protein structures, explore molecular dynamics trajectories, and perform structure-based drug discovery analysis directly on their local machines.

[![GitHub repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sonmanhng/open-smiles)

## Features

- **High-Performance 3D Rendering**: Built on top of the Bevy game engine, utilizing GPU acceleration for smooth visualization of large macromolecules.
- **Molecular Dynamics**: Load and playback GROMACS `.xtc` trajectory files. Seamlessly sync 3D visualizations with MD frames in real-time.
- **Multiple Visualization Styles**: Switch between Spacefill, Wireframe, Sticks, Ribbons, and Surface (EPS) representations.
- **Drug Discovery Workspace**: Identify and analyze ligand-binding pockets, extract interaction fingerprints, and visualize molecular contacts.
- **Coordinate Transformation**: Adjust and translate XYZ coordinates of specific ligands or the entire protein structure. Export modified structures with full topological bond preservation.
- **Structure Alignment**: Perform sequence and structure alignments using Needleman-Wunsch, Smith-Waterman, and TM-Align algorithms.

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (edition 2021)
- Cargo (comes with Rust)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sonmanhng/open-smiles.git
cd open-smiles
```

2. Run the application using Cargo:
```bash
cargo run --release
```

## Usage

- **Import Structure**: Click "Import Structure" on the left panel to load a `.pdb`, `.cif`, or `.mmcif` file.
- **MD Trajectory**: Once a structure is loaded, click "Import Trajectory (XTC)" to load a `.xtc` file. The trajectory will map onto the currently loaded topology.
- **Camera Controls**:
  - **Left Click + Drag**: Orbit / Rotate
  - **Right Click + Drag**: Pan
  - **Scroll**: Zoom in/out
- **Transform & Export**: Navigate to the "Transform" tab on the right panel to translate selected objects (Protein or Ligand) in 3D space. Click "Export Selected" to save the modified structure as `.pdb` or `.sdf` (V2000).

## License

This project is open-source and available under the MIT License.

