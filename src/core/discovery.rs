use std::collections::{HashMap, HashSet};

use bevy::prelude::Vec3;

use crate::core::interactions::{Interaction, InteractionType};
use crate::core::protein::{Atom, Protein};

#[derive(Debug, Clone, Default)]
pub struct InteractionFingerprint {
    pub hydrogen_bonds: usize,
    pub salt_bridges: usize,
    pub hydrophobic_contacts: usize,
    pub pi_stacking: usize,
}

impl InteractionFingerprint {
    pub fn total(&self) -> usize {
        self.hydrogen_bonds + self.salt_bridges + self.hydrophobic_contacts + self.pi_stacking
    }
}

#[derive(Debug, Clone)]
pub struct BindingResidue {
    pub chain_id: String,
    pub res_name: String,
    pub res_seq: i32,
    pub min_distance: f32,
    pub atom_contacts: usize,
    pub fingerprint: InteractionFingerprint,
}

#[derive(Debug, Clone)]
pub struct BindingSiteSummary {
    pub ligand_index: usize,
    pub ligand_name: String,
    pub ligand_chain_id: String,
    pub ligand_seq_num: i32,
    pub ligand_atom_count: usize,
    pub residues: Vec<BindingResidue>,
    pub center: Vec3,
    pub radius: f32,
    pub hydrophobic_residues: usize,
    pub polar_residues: usize,
    pub charged_residues: usize,
    pub fingerprint: InteractionFingerprint,
}

impl BindingSiteSummary {
    pub fn residue_count(&self) -> usize {
        self.residues.len()
    }

    pub fn contact_count(&self) -> usize {
        self.residues.iter().map(|r| r.atom_contacts).sum()
    }

    pub fn ligand_label(&self) -> String {
        format!(
            "{} {}:{}",
            self.ligand_name, self.ligand_chain_id, self.ligand_seq_num
        )
    }
}

#[derive(Debug, Clone)]
struct ResidueAccumulator {
    chain_id: String,
    res_name: String,
    res_seq: i32,
    min_distance: f32,
    atom_contacts: usize,
    fingerprint: InteractionFingerprint,
    residue_center: Vec3,
}

pub fn analyze_binding_sites(
    protein: &Protein,
    interactions: &[Interaction],
    cutoff: f32,
) -> Vec<BindingSiteSummary> {
    let mut summaries = Vec::new();

    for (ligand_index, ligand) in protein.ligands.iter().enumerate() {
        if ligand.atoms.is_empty() {
            continue;
        }

        let ligand_center = centroid(ligand.atoms.iter());
        let mut residues: HashMap<(String, i32, String), ResidueAccumulator> = HashMap::new();

        for chain in &protein.chains {
            for residue in &chain.residues {
                let residue_center = centroid(residue.atoms.iter());
                for res_atom in &residue.atoms {
                    let res_pos = atom_pos(res_atom);
                    for lig_atom in &ligand.atoms {
                        let dist = res_pos.distance(atom_pos(lig_atom));
                        if dist <= cutoff {
                            let key = (chain.id.clone(), residue.seq_num, residue.name.clone());
                            let entry = residues.entry(key).or_insert_with(|| ResidueAccumulator {
                                chain_id: chain.id.clone(),
                                res_name: residue.name.clone(),
                                res_seq: residue.seq_num,
                                min_distance: dist,
                                atom_contacts: 0,
                                fingerprint: InteractionFingerprint::default(),
                                residue_center,
                            });
                            entry.min_distance = entry.min_distance.min(dist);
                            entry.atom_contacts += 1;
                        }
                    }
                }
            }
        }

        for inter in interactions
            .iter()
            .filter(|inter| inter.ligand_index == ligand_index)
        {
            let key = (
                inter.chain_id.clone(),
                inter.res_seq,
                inter.res_name.clone(),
            );
            if let Some(entry) = residues.get_mut(&key) {
                add_interaction(&mut entry.fingerprint, &inter.i_type);
            }
        }

        let mut residue_rows: Vec<BindingResidue> = residues
            .values()
            .map(|entry| BindingResidue {
                chain_id: entry.chain_id.clone(),
                res_name: entry.res_name.clone(),
                res_seq: entry.res_seq,
                min_distance: entry.min_distance,
                atom_contacts: entry.atom_contacts,
                fingerprint: entry.fingerprint.clone(),
            })
            .collect();
        residue_rows.sort_by(|a, b| {
            a.min_distance
                .partial_cmp(&b.min_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut radius = 0.0_f32;
        for entry in residues.values() {
            radius = radius.max(entry.residue_center.distance(ligand_center));
        }

        let residue_names: HashSet<_> = residues.values().map(|r| r.res_name.as_str()).collect();
        let hydrophobic_residues = residue_names
            .iter()
            .filter(|name| is_hydrophobic_residue(name))
            .count();
        let polar_residues = residue_names
            .iter()
            .filter(|name| is_polar_residue(name))
            .count();
        let charged_residues = residue_names
            .iter()
            .filter(|name| is_charged_residue(name))
            .count();

        let mut fingerprint = InteractionFingerprint::default();
        for inter in interactions
            .iter()
            .filter(|inter| inter.ligand_index == ligand_index)
        {
            add_interaction(&mut fingerprint, &inter.i_type);
        }

        summaries.push(BindingSiteSummary {
            ligand_index,
            ligand_name: ligand.name.clone(),
            ligand_chain_id: ligand.chain_id.clone(),
            ligand_seq_num: ligand.seq_num,
            ligand_atom_count: ligand.atoms.len(),
            residues: residue_rows,
            center: ligand_center,
            radius,
            hydrophobic_residues,
            polar_residues,
            charged_residues,
            fingerprint,
        });
    }

    summaries.sort_by(|a, b| b.residue_count().cmp(&a.residue_count()));
    summaries
}

pub fn residue_druggability_note(site: &BindingSiteSummary) -> &'static str {
    if site.residue_count() >= 12
        && site.fingerprint.total() >= 4
        && site.hydrophobic_residues >= 4
        && site.polar_residues >= 2
    {
        "Balanced pocket: good starting point for medicinal chemistry review"
    } else if site.residue_count() >= 8 && site.fingerprint.total() >= 2 {
        "Moderate pocket: inspect geometry and conserved residues"
    } else if site.residue_count() > 0 {
        "Shallow or weakly annotated pocket: verify ligand pose and contacts"
    } else {
        "No pocket residues found with the current cutoff"
    }
}

fn add_interaction(fingerprint: &mut InteractionFingerprint, i_type: &InteractionType) {
    match i_type {
        InteractionType::HydrogenBond => fingerprint.hydrogen_bonds += 1,
        InteractionType::SaltBridge => fingerprint.salt_bridges += 1,
        InteractionType::Hydrophobic => fingerprint.hydrophobic_contacts += 1,
        InteractionType::PiPiStacking => fingerprint.pi_stacking += 1,
    }
}

fn centroid<'a>(atoms: impl Iterator<Item = &'a Atom>) -> Vec3 {
    let mut center = Vec3::ZERO;
    let mut count = 0usize;
    for atom in atoms {
        center += atom_pos(atom);
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

fn is_hydrophobic_residue(name: &str) -> bool {
    matches!(
        name,
        "ALA" | "VAL" | "ILE" | "LEU" | "MET" | "PHE" | "TYR" | "TRP" | "PRO"
    )
}

fn is_polar_residue(name: &str) -> bool {
    matches!(
        name,
        "SER" | "THR" | "ASN" | "GLN" | "CYS" | "TYR" | "HIS" | "TRP"
    )
}

fn is_charged_residue(name: &str) -> bool {
    matches!(name, "ASP" | "GLU" | "ARG" | "LYS" | "HIS")
}
