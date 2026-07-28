use crate::core::protein::{Atom, Protein, Residue};
use bevy::prelude::Vec3;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum InteractionType {
    HydrogenBond,
    SaltBridge,
    Hydrophobic,
    PiPiStacking,
}

#[derive(Debug, Clone)]
pub struct Interaction {
    pub p1: Vec3,
    pub p2: Vec3,
    pub i_type: InteractionType,
    pub dist: f32,
    pub atom1_name: String,
    pub atom2_name: String,
    pub res_name: String,
    pub res_seq: i32,
    pub chain_id: String,
    pub ligand_index: usize,
}

/// Detects non-covalent interactions between Ligands and Protein chains.
pub fn detect_interactions(
    protein: &Protein,
    hbond_threshold: f32,
    hydro_threshold: f32,
) -> Vec<Interaction> {
    let mut interactions = Vec::new();

    // 1. Collect all protein atoms that could be involved
    let mut protein_atoms = Vec::new();
    for chain in &protein.chains {
        for res in &chain.residues {
            for atom in &res.atoms {
                protein_atoms.push((atom, res, chain.id.clone()));
            }
        }
    }

    // 2. Iterate through all ligand atoms
    for (ligand_index, ligand) in protein.ligands.iter().enumerate() {
        for l_atom in &ligand.atoms {
            let l_pos = Vec3::new(l_atom.x as f32, l_atom.y as f32, l_atom.z as f32);
            let l_elem = l_atom.element.trim().to_uppercase();
            let is_l_donor_acceptor = l_elem == "O" || l_elem == "N" || l_elem == "F";
            let is_l_carbon = l_elem == "C";

            for (p_atom, p_res, chain_id) in &protein_atoms {
                let p_pos = Vec3::new(p_atom.x as f32, p_atom.y as f32, p_atom.z as f32);
                let p_elem = p_atom.element.trim().to_uppercase();
                let dist = l_pos.distance(p_pos);

                if dist > hbond_threshold.max(hydro_threshold) || dist < 1.0 {
                    continue; // Skip atoms too far or covalently bonded
                }

                // Hydrogen Bonds (Donor/Acceptor)
                if dist < hbond_threshold {
                    let is_p_donor_acceptor = p_elem == "O" || p_elem == "N" || p_elem == "F";
                    if is_l_donor_acceptor && is_p_donor_acceptor {
                        // Very crude heuristic for Salt Bridges vs H-Bonds
                        let p_res_name = p_res.name.as_str();
                        let is_acidic = p_res_name == "ASP" || p_res_name == "GLU";
                        let is_basic = p_res_name == "ARG" || p_res_name == "LYS";

                        // If one is O and other is N and it involves charged residues
                        if (is_acidic && p_elem == "O" && l_elem == "N")
                            || (is_basic && p_elem == "N" && l_elem == "O")
                        {
                            interactions.push(Interaction {
                                p1: l_pos,
                                p2: p_pos,
                                i_type: InteractionType::SaltBridge,
                                dist,
                                atom1_name: l_atom.name.clone(),
                                atom2_name: p_atom.name.clone(),
                                res_name: p_res.name.clone(),
                                res_seq: p_res.seq_num,
                                chain_id: chain_id.clone(),
                                ligand_index,
                            });
                            continue;
                        }

                        interactions.push(Interaction {
                            p1: l_pos,
                            p2: p_pos,
                            i_type: InteractionType::HydrogenBond,
                            dist,
                            atom1_name: l_atom.name.clone(),
                            atom2_name: p_atom.name.clone(),
                            res_name: p_res.name.clone(),
                            res_seq: p_res.seq_num,
                            chain_id: chain_id.clone(),
                            ligand_index,
                        });
                    }
                }

                // Hydrophobic Interactions
                if dist < hydro_threshold {
                    if is_l_carbon && p_elem == "C" {
                        // Check if they are part of hydrophobic residues
                        let p_res_name = p_res.name.as_str();
                        let is_hydrophobic =
                            ["ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TYR", "TRP"]
                                .contains(&p_res_name);
                        if is_hydrophobic {
                            interactions.push(Interaction {
                                p1: l_pos,
                                p2: p_pos,
                                i_type: InteractionType::Hydrophobic,
                                dist,
                                atom1_name: l_atom.name.clone(),
                                atom2_name: p_atom.name.clone(),
                                res_name: p_res.name.clone(),
                                res_seq: p_res.seq_num,
                                chain_id: chain_id.clone(),
                                ligand_index,
                            });
                        }
                    }
                }
            }
        }
    }

    detect_pi_stacking(protein, &mut interactions);

    interactions
}

fn detect_pi_stacking(protein: &Protein, interactions: &mut Vec<Interaction>) {
    let mut seen = HashSet::new();

    for (ligand_index, ligand) in protein.ligands.iter().enumerate() {
        let Some(ligand_ring) = ligand_aromatic_centroid(&ligand.atoms) else {
            continue;
        };

        for chain in &protein.chains {
            for residue in &chain.residues {
                if !is_aromatic_residue(&residue.name) {
                    continue;
                }

                let Some(residue_ring) = residue_aromatic_centroid(residue) else {
                    continue;
                };

                let dist = ligand_ring.distance(residue_ring);
                if !(3.3..=6.0).contains(&dist) {
                    continue;
                }

                let key = (
                    ligand_index,
                    chain.id.clone(),
                    residue.seq_num,
                    residue.name.clone(),
                );
                if !seen.insert(key) {
                    continue;
                }

                interactions.push(Interaction {
                    p1: ligand_ring,
                    p2: residue_ring,
                    i_type: InteractionType::PiPiStacking,
                    dist,
                    atom1_name: "aromatic-ring".into(),
                    atom2_name: "aromatic-ring".into(),
                    res_name: residue.name.clone(),
                    res_seq: residue.seq_num,
                    chain_id: chain.id.clone(),
                    ligand_index,
                });
            }
        }
    }
}

fn ligand_aromatic_centroid(atoms: &[Atom]) -> Option<Vec3> {
    let candidates: Vec<Vec3> = atoms
        .iter()
        .filter(|atom| {
            let element = atom.element.trim().to_uppercase();
            matches!(element.as_str(), "C" | "N")
        })
        .map(atom_pos)
        .collect();

    if candidates.len() < 5 {
        return None;
    }

    let center = centroid(candidates.iter().copied());
    let close_atoms: Vec<Vec3> = candidates
        .into_iter()
        .filter(|pos| pos.distance(center) <= 3.2)
        .collect();

    if close_atoms.len() < 5 {
        return None;
    }

    Some(centroid(close_atoms.into_iter()))
}

fn residue_aromatic_centroid(residue: &Residue) -> Option<Vec3> {
    let names: &[&str] = match residue.name.as_str() {
        "PHE" | "TYR" => &["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP" => &["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS" => &["CG", "ND1", "CD2", "CE1", "NE2"],
        _ => return None,
    };

    let mut points = Vec::new();
    for name in names {
        if let Some(atom) = residue.atoms.iter().find(|atom| atom.name.trim() == *name) {
            points.push(atom_pos(atom));
        }
    }

    if points.len() < 4 {
        None
    } else {
        Some(centroid(points.into_iter()))
    }
}

fn is_aromatic_residue(name: &str) -> bool {
    matches!(name, "PHE" | "TYR" | "TRP" | "HIS")
}

fn centroid(points: impl Iterator<Item = Vec3>) -> Vec3 {
    let mut center = Vec3::ZERO;
    let mut count = 0usize;
    for point in points {
        center += point;
        count += 1;
    }
    if count == 0 {
        Vec3::ZERO
    } else {
        center / count as f32
    }
}

fn atom_pos(atom: &Atom) -> Vec3 {
    Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32)
}
