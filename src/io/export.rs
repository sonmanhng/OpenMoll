use crate::core::protein::{Ligand, Protein};
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformTarget {
    Protein,
    Ligand(usize),
}

/// Helper to format a single atom into PDB string
fn format_pdb_atom(
    out: &mut String,
    record_type: &str,
    serial: i32,
    name: &str,
    res_name: &str,
    chain_id: &str,
    res_seq: i32,
    x: f64,
    y: f64,
    z: f64,
    b_factor: f64,
    element: &str,
) {
    let _ = writeln!(
        out,
        "{:<6}{:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}",
        record_type,
        serial.clamp(1, 99999),
        name,
        res_name,
        chain_id,
        res_seq.clamp(-999, 9999),
        x,
        y,
        z,
        b_factor,
        element
    );
}

/// Export the given protein (or part of it) to a PDB string
pub fn export_pdb(protein: &Protein, target: Option<TransformTarget>) -> String {
    let mut out = String::new();
    
    let export_protein = match target {
        None | Some(TransformTarget::Protein) => true,
        _ => false,
    };
    
    if export_protein {
        for chain in &protein.chains {
            for residue in &chain.residues {
                for atom in &residue.atoms {
                    // Quick formatting for atom name to align correctly
                    let mut fmt_name = format!("{:<4}", atom.name);
                    if atom.name.len() < 4 {
                        fmt_name = format!(" {:<3}", atom.name);
                    }
                    
                    let _ = writeln!(
                        out,
                        "{:<6}{:5} {:4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}",
                        "ATOM  ",
                        atom.serial.clamp(1, 99999),
                        fmt_name,
                        residue.name,
                        chain.id,
                        residue.seq_num.clamp(-999, 9999),
                        atom.x,
                        atom.y,
                        atom.z,
                        atom.b_factor,
                        atom.element
                    );
                }
            }
        }
    }
    
    for (idx, ligand) in protein.ligands.iter().enumerate() {
        let export_this_ligand = match target {
            None => true,
            Some(TransformTarget::Ligand(i)) if i == idx => true,
            _ => false,
        };
        
        if export_this_ligand {
            for atom in &ligand.atoms {
                let mut fmt_name = format!("{:<4}", atom.name);
                if atom.name.len() < 4 {
                    fmt_name = format!(" {:<3}", atom.name);
                }
                let _ = writeln!(
                    out,
                    "{:<6}{:5} {:4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}",
                    "HETATM",
                    atom.serial.clamp(1, 99999),
                    fmt_name,
                    ligand.name,
                    ligand.chain_id,
                    ligand.seq_num.clamp(-999, 9999),
                    atom.x,
                    atom.y,
                    atom.z,
                    atom.b_factor,
                    atom.element
                );
            }
            
            // Add CONECT records for ligand
            for &(a, b) in &ligand.bonds {
                if a < ligand.atoms.len() && b < ligand.atoms.len() {
                    let atom_a = &ligand.atoms[a];
                    let atom_b = &ligand.atoms[b];
                    let _ = writeln!(
                        out,
                        "CONECT{:5}{:5}",
                        atom_a.serial.clamp(1, 99999),
                        atom_b.serial.clamp(1, 99999)
                    );
                }
            }
        }
    }
    
    out.push_str("END\n");
    out
}

/// Export a ligand to V2000 SDF string
pub fn export_sdf(ligand: &Ligand) -> String {
    let mut out = String::new();
    
    // Header
    let _ = writeln!(out, "{}", ligand.name);
    let _ = writeln!(out, "  OpenMoll 3D");
    let _ = writeln!(out, "");
    
    // Counts line: atoms, bonds
    let _ = writeln!(
        out,
        "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000",
        ligand.atoms.len(),
        ligand.bonds.len()
    );
    
    // Atom block
    for atom in &ligand.atoms {
        let _ = writeln!(
            out,
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0",
            atom.x, atom.y, atom.z, atom.element
        );
    }
    
    // Bond block
    for &(a, b) in &ligand.bonds {
        // SDF is 1-indexed, bond type 1 (single)
        let _ = writeln!(out, "{:>3}{:>3}  1  0  0  0  0", a + 1, b + 1);
    }
    
    let _ = writeln!(out, "M  END");
    let _ = writeln!(out, "$$$$");
    
    out
}
