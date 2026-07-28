use crate::core::protein::Protein;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use mcubes::{MarchingCubes, MeshSide};
use std::collections::HashMap;

use crate::core::electrostatics::{calculate_coulombic_potential, get_residue_charge, PointCharge};

pub fn generate_surface_mesh(
    protein: &Protein,
    viz_state: &crate::graphics::VizState,
) -> Option<Mesh> {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    let mut active_atoms_colors: Vec<(Vec3, [f32; 4])> = Vec::new();
    let mut has_atoms = false;
    let mut any_eps = false;

    for chain in &protein.chains {
        if let Some(state) = viz_state.chain_states.get(&chain.id) {
            if !state.surface && !state.eps_surface {
                continue;
            }
            if state.eps_surface {
                any_eps = true;
            }
            let chain_color = if state.use_custom_color {
                [
                    state.custom_color[0],
                    state.custom_color[1],
                    state.custom_color[2],
                    1.0,
                ]
            } else {
                [0.8, 0.8, 0.8, 1.0] // Default fallback if needed
            };

            for residue in &chain.residues {
                for atom in &residue.atoms {
                    has_atoms = true;
                    min_x = min_x.min(atom.x as f32);
                    max_x = max_x.max(atom.x as f32);
                    min_y = min_y.min(atom.y as f32);
                    max_y = max_y.max(atom.y as f32);
                    min_z = min_z.min(atom.z as f32);
                    max_z = max_z.max(atom.z as f32);

                    let atom_color = if !state.use_custom_color {
                        let el = atom
                            .element
                            .chars()
                            .next()
                            .unwrap_or('C')
                            .to_ascii_uppercase();
                        match el {
                            'C' => [0.5, 0.5, 0.5, 1.0],
                            'O' => [1.0, 0.0, 0.0, 1.0],
                            'N' => [0.0, 0.0, 1.0, 1.0],
                            'S' => [1.0, 1.0, 0.0, 1.0],
                            'P' => [1.0, 0.5, 0.0, 1.0],
                            _ => [0.8, 0.8, 0.8, 1.0],
                        }
                    } else {
                        chain_color
                    };

                    active_atoms_colors.push((
                        Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32),
                        atom_color,
                    ));
                }
            }
        }
    }

    if !has_atoms {
        return None;
    }

    let padding = 4.0;
    min_x -= padding;
    max_x += padding;
    min_y -= padding;
    max_y += padding;
    min_z -= padding;
    max_z += padding;

    let resolution = viz_state.surface_resolution.max(0.1);
    let nx = ((max_x - min_x) / resolution).ceil() as usize;
    let ny = ((max_y - min_y) / resolution).ceil() as usize;
    let nz = ((max_z - min_z) / resolution).ceil() as usize;

    let mut grid = vec![0.0f32; nx * ny * nz];

    let radius = 1.8;
    let radius_sq = radius * radius;

    for chain in &protein.chains {
        if let Some(state) = viz_state.chain_states.get(&chain.id) {
            if !state.surface && !state.eps_surface {
                continue;
            }
        } else {
            continue;
        }

        for residue in &chain.residues {
            for atom in &residue.atoms {
                let ax = atom.x as f32;
                let ay = atom.y as f32;
                let az = atom.z as f32;

                let vx_min = ((ax - radius * 2.0 - min_x) / resolution).max(0.0) as usize;
                let vx_max =
                    ((ax + radius * 2.0 - min_x) / resolution).min((nx - 1) as f32) as usize;

                let vy_min = ((ay - radius * 2.0 - min_y) / resolution).max(0.0) as usize;
                let vy_max =
                    ((ay + radius * 2.0 - min_y) / resolution).min((ny - 1) as f32) as usize;

                let vz_min = ((az - radius * 2.0 - min_z) / resolution).max(0.0) as usize;
                let vz_max =
                    ((az + radius * 2.0 - min_z) / resolution).min((nz - 1) as f32) as usize;

                for x in vx_min..=vx_max {
                    for y in vy_min..=vy_max {
                        for z in vz_min..=vz_max {
                            let px = min_x + (x as f32) * resolution;
                            let py = min_y + (y as f32) * resolution;
                            let pz = min_z + (z as f32) * resolution;

                            let dx = px - ax;
                            let dy = py - ay;
                            let dz = pz - az;
                            let dist_sq = dx * dx + dy * dy + dz * dz;

                            let density = (-dist_sq / radius_sq).exp();
                            let idx = x + y * nx + z * nx * ny;
                            if idx < grid.len() {
                                grid[idx] += density;
                            }
                        }
                    }
                }
            }
        }
    }

    let mc = MarchingCubes::new(
        (nx, ny, nz),
        (resolution, resolution, resolution),
        (1.0, 1.0, 1.0),
        lin_alg::f32::Vec3::new(min_x, min_y, min_z),
        grid,
        viz_state.surface_iso_level, // iso_level
    )
    .unwrap();

    let mc_mesh = mc.generate(MeshSide::OutsideOnly);

    // Build spatial grid for coloring
    let cell_size = 4.0_f32;
    let mut spatial_grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (idx, (apos, _)) in active_atoms_colors.iter().enumerate() {
        let cx = (apos.x / cell_size).floor() as i32;
        let cy = (apos.y / cell_size).floor() as i32;
        let cz = (apos.z / cell_size).floor() as i32;
        spatial_grid.entry((cx, cy, cz)).or_default().push(idx);
    }

    let mut positions = Vec::with_capacity(mc_mesh.vertices.len());
    let mut normals = Vec::with_capacity(mc_mesh.vertices.len());
    let mut colors = Vec::with_capacity(mc_mesh.vertices.len());

    let mut charges = Vec::new();
    if any_eps {
        for chain in &protein.chains {
            if let Some(state) = viz_state.chain_states.get(&chain.id) {
                if !state.eps_surface {
                    continue;
                }
            } else {
                continue;
            }

            for residue in &chain.residues {
                let q = get_residue_charge(&residue.name);
                if q != 0.0 {
                    if let Some(atom) = residue.atoms.first() {
                        charges.push(PointCharge {
                            x: atom.x as f32,
                            y: atom.y as f32,
                            z: atom.z as f32,
                            q,
                        });
                    }
                }
            }
        }
    }

    for v in &mc_mesh.vertices {
        let px = v.posit.x;
        let py = v.posit.y;
        let pz = v.posit.z;

        positions.push([px, py, pz]);
        normals.push([v.normal.x, v.normal.y, v.normal.z]);

        // Very basic handling: if any eps chain exists, we just compute it for the whole surface mesh
        // This avoids discontinuities in color.
        if any_eps {
            let pot = calculate_coulombic_potential(px, py, pz, &charges);
            let pot_clamped = pot.clamp(-1.0, 1.0);
            if pot_clamped < 0.0 {
                let intensity = 1.0 + pot_clamped;
                colors.push([1.0, intensity, intensity, 1.0]);
            } else {
                let intensity = 1.0 - pot_clamped;
                colors.push([intensity, intensity, 1.0, 1.0]);
            }
        } else {
            let mut min_dist_sq = f32::MAX;
            let mut closest_color = [0.8, 0.8, 0.8, 1.0];
            let v_pos = Vec3::new(px, py, pz);

            let cx = (px / cell_size).floor() as i32;
            let cy = (py / cell_size).floor() as i32;
            let cz = (pz / cell_size).floor() as i32;

            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(indices) = spatial_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &idx in indices {
                                let (apos, acol) = &active_atoms_colors[idx];
                                let d2 = v_pos.distance_squared(*apos);
                                if d2 < min_dist_sq {
                                    min_dist_sq = d2;
                                    closest_color = *acol;
                                }
                            }
                        }
                    }
                }
            }

            if min_dist_sq == f32::MAX {
                for (apos, acol) in &active_atoms_colors {
                    let d2 = v_pos.distance_squared(*apos);
                    if d2 < min_dist_sq {
                        min_dist_sq = d2;
                        closest_color = *acol;
                    }
                }
            }

            colors.push(closest_color);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(mc_mesh.indices.len());
    for chunk in mc_mesh.indices.chunks(3) {
        if chunk.len() == 3 {
            indices.push(chunk[0] as u32);
            indices.push(chunk[2] as u32);
            indices.push(chunk[1] as u32);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();

    Some(mesh)
}
