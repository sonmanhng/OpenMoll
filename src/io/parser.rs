use crate::core::protein::{
    Atom, Chain, Ligand, LigandType, MoleculeType, Protein, Residue, SecondaryStructure,
    COMMON_IONS, DNA_RESIDUES, RNA_RESIDUES, WATER_NAMES,
};
use crate::core::secondary::{
    assign_from_cif_file, assign_from_pdb_file, infer_secondary_structure,
};
use anyhow::Result;

/// Decode a hybrid-36 encoded field (GROMACS ≥100 000 atoms).
/// Returns None for plain decimal strings; caller may then parse as integer.
fn hybrid36_decode(s: &str) -> Option<i32> {
    let s = s.trim();
    if s.is_empty() { return None; }
    let first = s.chars().next()?;
    if first.is_ascii_digit() { return None; } // plain decimal
    let mut value: i64 = 0;
    for ch in s.chars() {
        let d = match ch {
            '0'..='9' => ch as i64 - '0' as i64,
            'A'..='Z' => ch as i64 - 'A' as i64 + 10,
            'a'..='z' => ch as i64 - 'a' as i64 + 36,
            _ => return None,
        };
        value = value * 36 + d;
    }
    Some(value as i32)
}

/// Infer element symbol from PDB atom name when the ELEMENT column is absent.
fn infer_element(atom_name: &str) -> String {
    // Strip leading digits (e.g. "1HB " → "H"), then take first alpha char.
    atom_name
        .chars()
        .find(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_string())
        .unwrap_or_else(|| "C".to_string())
}

/// Fully custom PDB parser — works for any number of atoms and any
/// serial/residue-number encoding (decimal, hybrid-36, OpenMM hex, etc.).
/// Does NOT depend on pdbtbx; groups atoms into chains/residues by
/// consecutive (chain, raw_res_key) runs, matching how MD engines write files.
fn load_pdb_native(path: &str) -> Result<Protein> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path, e))?;

    // ── Atom accumulator ──────────────────────────────────────────────────
    struct RawGroup {
        chain_id: String,
        res_name: String,
        seq_num: i32,   // sequential, per-chain
        atoms: Vec<Atom>,
        is_hetero: bool,
    }
    let mut groups: Vec<RawGroup> = Vec::new();

    // Running state
    let mut seq_serial: i32 = 0;
    let mut cur_chain = String::new();
    let mut cur_res_key = String::new(); // raw cols 21-26 (chain+resnum+icode)
    let mut cur_res_name = String::new();
    let mut cur_is_hetero = true;
    let mut cur_atoms: Vec<Atom> = Vec::new();
    let mut chain_res_counter: std::collections::HashMap<String, i32> =
        std::collections::HashMap::new();

    // Maps ORIGINAL PDB serial → our sequential seq_serial.
    // Critical so CONECT record bonds still match after renumbering.
    let mut orig_to_seq: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();

    // Flush current residue group into `groups`
    macro_rules! flush {
        () => {
            if !cur_atoms.is_empty() {
                let seq = *chain_res_counter.get(&cur_chain).unwrap_or(&0);
                groups.push(RawGroup {
                    chain_id: cur_chain.clone(),
                    res_name: cur_res_name.clone(),
                    seq_num: seq,
                    atoms: std::mem::take(&mut cur_atoms),
                    is_hetero: cur_is_hetero,
                });
            }
        };
    }

    for line in content.lines() {
        let is_atom   = line.starts_with("ATOM  ");
        let is_hetatm = line.starts_with("HETATM");
        if !is_atom && !is_hetatm { continue; }
        if line.len() < 54 { continue; }

        // Skip alternate conformations B/C/D/…
        let alt = line.as_bytes().get(16).copied().unwrap_or(b' ');
        if alt != b' ' && alt != b'A' { continue; }

        seq_serial += 1;

        // PDB fixed-format columns (0-indexed):
        //   6-10  atom serial  (original, may be hybrid-36/hex)
        //  12-15  atom name
        //  17-19  res name
        //  21     chain ID
        //  22-25  res seq num  (may be hybrid-36 / hex)
        //  26     iCode
        //  30-37  x
        //  38-45  y
        //  46-53  z
        //  60-65  b_factor
        //  76-77  element symbol

        // Decode original serial for CONECT mapping
        let orig_serial_str = &line[6..11];
        let orig_serial: i32 = hybrid36_decode(orig_serial_str)
            .or_else(|| orig_serial_str.trim().parse::<i32>().ok())
            .unwrap_or(seq_serial);
        // Store mapping (last occurrence wins for wrapping serials; ligands
        // are at the start of MD files so their serials are unique).
        orig_to_seq.insert(orig_serial, seq_serial);

        let atom_name  = line[12..16].trim().to_string();
        let res_name   = line[17..20].trim().to_string();
        let chain_id   = line[21..22].to_string();
        let res_key    = line[21..27.min(line.len())].to_string(); // chain+resnum+icode

        let x = line[30..38].trim().parse::<f64>().unwrap_or(0.0);
        let y = line[38..46].trim().parse::<f64>().unwrap_or(0.0);
        let z = line[46..54].trim().parse::<f64>().unwrap_or(0.0);

        let b_factor = line.get(60..66)
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0);

        let element = line.get(76..78)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| infer_element(&atom_name));

        let is_backbone = atom_name == "CA" || atom_name == "C4'";

        // Detect residue boundary (new res_key)
        if res_key != cur_res_key {
            flush!();
            let ctr = chain_res_counter.entry(chain_id.clone()).or_insert(0);
            *ctr += 1;
            cur_chain    = chain_id.clone();
            cur_res_name = res_name.clone();
            cur_res_key  = res_key;
            cur_is_hetero = is_hetatm;
        }
        if is_atom { cur_is_hetero = false; }

        cur_atoms.push(Atom {
            name: atom_name,
            element,
            serial: seq_serial,
            x, y, z,
            b_factor,
            is_backbone,
            is_hetero: is_hetatm,
        });
    }
    flush!();


    // ── Build Protein from groups ─────────────────────────────────────────
    use std::collections::HashMap;
    let mut chain_map: HashMap<String, Vec<usize>> = HashMap::new(); // chain_id → group indices
    for (i, g) in groups.iter().enumerate() {
        chain_map.entry(g.chain_id.clone()).or_default().push(i);
    }
    // Preserve chain order by first appearance
    let mut chain_order: Vec<String> = Vec::new();
    for g in &groups {
        if !chain_order.contains(&g.chain_id) {
            chain_order.push(g.chain_id.clone());
        }
    }

    let mut chains: Vec<Chain> = Vec::new();
    let mut ligands: Vec<Ligand> = Vec::new();

    for chain_id in &chain_order {
        let indices = &chain_map[chain_id];
        let mut residues: Vec<Residue> = Vec::new();

        for &gi in indices {
            let g = &groups[gi];
            if g.is_hetero {
                let non_h = g.atoms.iter().filter(|a| {
                    let el = a.element.trim().to_uppercase();
                    el != "H" && el != "D"
                }).count();
                let lig_type = if non_h <= 1 || COMMON_IONS.contains(&g.res_name.as_str()) {
                    LigandType::Ion
                } else {
                    LigandType::Ligand
                };
                ligands.push(Ligand {
                    name: g.res_name.clone(),
                    chain_id: chain_id.clone(),
                    seq_num: g.seq_num,
                    atoms: g.atoms.clone(),
                    ligand_type: lig_type,
                    bonds: Vec::new(),
                });
            } else {
                residues.push(Residue {
                    name: g.res_name.clone(),
                    seq_num: g.seq_num,
                    atoms: g.atoms.clone(),
                    secondary_structure: SecondaryStructure::Coil,
                });
            }
        }

        let mol_type = classify_chain_type(&residues);
        chains.push(Chain { id: chain_id.clone(), residues, molecule_type: mol_type });
    }

    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown")
        .to_string();

    let mut protein = Protein { name, chains, ligands, bonds: Vec::new() };

    // ── Parse CONECT records with serial remapping ─────────────────────────
    // CONECT records use the ORIGINAL PDB serials. We remap them to our
    // sequential seq_serials using orig_to_seq built during atom parsing.
    {
        let mut bond_set: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();

        for line in content.lines() {
            if !line.starts_with("CONECT") { continue; }
            // Source serial: cols 7-11 (0-indexed 6..11)
            let src_orig = match line.get(6..11).and_then(|s| s.trim().parse::<i32>().ok()) {
                Some(s) => s,
                None => continue,
            };
            let src = match orig_to_seq.get(&src_orig) {
                Some(&s) => s,
                None => continue, // atom wasn't loaded (e.g. alt conf skipped)
            };

            // Target serials at columns 11-16, 16-21, 21-26, 26-31
            for i in 0..4_usize {
                let start = 11 + i * 5;
                let end = start + 5;
                let tgt_str = if line.len() >= end {
                    &line[start..end]
                } else if line.len() > start {
                    &line[start..]
                } else {
                    break;
                };
                let tgt_orig = match tgt_str.trim().parse::<i32>() {
                    Ok(v) if v > 0 => v,
                    _ => continue,
                };
                let tgt = match orig_to_seq.get(&tgt_orig) {
                    Some(&t) => t,
                    None => continue,
                };
                let a = src.min(tgt);
                let b = src.max(tgt);
                bond_set.insert((a, b));
            }
        }

        protein.bonds = bond_set.into_iter().collect();
        protein.bonds.sort_unstable();

        // Also populate per-ligand bonds
        let mut serial_to_lig: std::collections::HashMap<i32, (usize, usize)> =
            std::collections::HashMap::new();
        for (li, lig) in protein.ligands.iter().enumerate() {
            for (ai, atom) in lig.atoms.iter().enumerate() {
                serial_to_lig.insert(atom.serial, (li, ai));
            }
        }
        let mut lig_bond_set: std::collections::HashSet<(usize, usize, usize)> =
            std::collections::HashSet::new();
        for &(a, b) in &protein.bonds {
            if let (Some(&(li1, ai1)), Some(&(li2, ai2))) =
                (serial_to_lig.get(&a), serial_to_lig.get(&b))
            {
                if li1 == li2 {
                    lig_bond_set.insert((li1, ai1.min(ai2), ai1.max(ai2)));
                }
            }
        }
        for (li, a, b) in lig_bond_set {
            if let Some(lig) = protein.ligands.get_mut(li) {
                lig.bonds.push((a, b));
            }
        }
        for lig in protein.ligands.iter_mut() {
            lig.bonds.sort();
        }
    }

    assign_from_pdb_file(&mut protein, path);
    infer_secondary_structure(&mut protein.chains);

    Ok(protein)
}


/// Load a protein structure from a PDB or mmCIF file.
/// For .pdb files: uses the fully custom native parser (handles large systems,
/// hybrid-36, hex residue numbers, etc.).
/// For .cif/.mmcif files: delegates to pdbtbx.
pub fn load_pdb(path: &str) -> Result<Protein> {
    // Route .pdb files to the custom native parser
    let lower = path.to_lowercase();
    if lower.ends_with(".pdb") {
        return load_pdb_native(path);
    }

    // ── mmCIF / CIF: use pdbtbx ───────────────────────────────────────────
    let (pdb, _errors) = pdbtbx::ReadOptions::new()
        .set_only_first_model(true)
        .read(path)
        .or_else(|_| {
            pdbtbx::ReadOptions::new()
                .set_level(pdbtbx::StrictnessLevel::Loose)
                .set_only_atomic_coords(true)
                .set_only_first_model(true)
                .read(path)
        })
        .map_err(|e| anyhow::anyhow!("Failed to open structure file: {:?}", e))?;

    let mut chains = Vec::new();
    let mut ligands: Vec<Ligand> = Vec::new();

    for chain in pdb.chains() {
        let mut residues = Vec::new();
        for residue in chain.residues() {
            let pdbtbx_atoms: Vec<_> = residue.atoms().collect();
            let res_name = residue.name().unwrap_or("UNK").trim().to_string();
            let all_hetero = pdbtbx_atoms.iter().all(|a| a.hetero());

            let atoms: Vec<Atom> = pdbtbx_atoms
                .iter()
                .map(|atom| Atom {
                    name: atom.name().to_string(),
                    element: atom
                        .element()
                        .map(|e| format!("{:?}", e))
                        .unwrap_or_default(),
                    serial: atom.serial_number() as i32,
                    x: atom.x(),
                    y: atom.y(),
                    z: atom.z(),
                    b_factor: atom.b_factor(),
                    is_backbone: atom.name() == "CA" || atom.name() == "C4'",
                    is_hetero: atom.hetero(),
                })
                .collect();

            if all_hetero {
                let non_h_count = atoms.iter().filter(|a| a.element.trim() != "H").count();
                let ligand_type = if non_h_count <= 1 || COMMON_IONS.contains(&res_name.as_str()) {
                    LigandType::Ion
                } else {
                    LigandType::Ligand
                };
                ligands.push(Ligand {
                    name: res_name,
                    chain_id: chain.id().to_string(),
                    seq_num: residue.serial_number() as i32,
                    atoms,
                    ligand_type,
                    bonds: Vec::new(),
                });
            } else {
                residues.push(Residue {
                    name: res_name,
                    seq_num: residue.serial_number() as i32,
                    atoms,
                    secondary_structure: SecondaryStructure::Coil,
                });
            }
        }
        let molecule_type = classify_chain_type(&residues);
        chains.push(Chain {
            id: chain.id().to_string(),
            residues,
            molecule_type,
        });
    }

    let name = pdb.identifier.as_deref().unwrap_or("Unknown").to_string();
    let mut protein = Protein { name, chains, ligands, bonds: Vec::new() };

    assign_from_cif_file(&mut protein, path);
    infer_secondary_structure(&mut protein.chains);

    Ok(protein)
}


/// Parse CONECT records from a PDB file to get explicit covalent bond topology for ligands.
/// This is the only reliable way to get correct bond connectivity for HETATMs like heme groups.
/// Parse CONECT records from a PDB file to get explicit covalent bond topology.
/// This will extract global bonds to protein.bonds, and also populate ligand-specific bonds.
fn parse_conect_records(path: &str, protein: &mut crate::core::protein::Protein) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return, // Not a plain text PDB (e.g., mmCIF), skip silently
    };

    let mut bond_set: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();

    for line in content.lines() {
        if !line.starts_with("CONECT") {
            continue;
        }

        let src_serial = match line[6..11.min(line.len())].trim().parse::<i32>() {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Serial numbers are in columns 12-16, 17-21, 22-26, 27-31 (1-indexed)
        let serials: Vec<i32> = (1..5)
            .filter_map(|i| {
                let start = 6 + i * 5;
                let end = start + 5;
                if line.len() >= end {
                    line[start..end].trim().parse::<i32>().ok()
                } else if line.len() > start {
                    line[start..].trim().parse::<i32>().ok()
                } else {
                    None
                }
            })
            .collect();

        for target_serial in serials {
            let a = src_serial.min(target_serial);
            let b = src_serial.max(target_serial);
            bond_set.insert((a, b));
        }
    }

    // Assign global bonds to protein
    protein.bonds = bond_set.into_iter().collect();
    protein.bonds.sort_unstable();

    // Also build a map from atom serial -> (ligand_index, atom_index_within_ligand)
    // to populate the local ligand.bonds (which the 2D interaction map relies on)
    let mut serial_to_ligand_atom: std::collections::HashMap<i32, (usize, usize)> =
        std::collections::HashMap::new();
    for (lig_idx, ligand) in protein.ligands.iter().enumerate() {
        for (atom_idx, atom) in ligand.atoms.iter().enumerate() {
            serial_to_ligand_atom.insert(atom.serial, (lig_idx, atom_idx));
        }
    }

    let mut lig_bonds: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    for &(a, b) in &protein.bonds {
        if let (Some(&(lig_idx1, a1)), Some(&(lig_idx2, b1))) =
            (serial_to_ligand_atom.get(&a), serial_to_ligand_atom.get(&b))
        {
            if lig_idx1 == lig_idx2 {
                lig_bonds.insert((lig_idx1, a1.min(b1), a1.max(b1)));
            }
        }
    }

    for (lig_idx, a, b) in lig_bonds {
        if let Some(ligand) = protein.ligands.get_mut(lig_idx) {
            ligand.bonds.push((a, b));
        }
    }

    // Sort each ligand's bond list for determinism
    for ligand in protein.ligands.iter_mut() {
        ligand.bonds.sort();
    }
}

/// Classify a chain's molecule type from its residue names.
///
/// Counts residues matching known RNA and DNA names. Whichever set has the
/// majority determines the type. If neither set has any matches (or there is
/// a tie), the chain defaults to `Protein`.
fn classify_chain_type(residues: &[Residue]) -> MoleculeType {
    let mut rna_count = 0usize;
    let mut dna_count = 0usize;

    for res in residues {
        let name = res.name.trim();
        if RNA_RESIDUES.contains(&name) {
            rna_count += 1;
        } else if DNA_RESIDUES.contains(&name) {
            dna_count += 1;
        }
    }

    if rna_count == 0 && dna_count == 0 {
        return MoleculeType::Protein;
    }
    if rna_count >= dna_count {
        MoleculeType::RNA
    } else {
        MoleculeType::DNA
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_chain_type_protein() {
        let residues = vec![
            Residue {
                name: "ALA".to_string(),
                seq_num: 1,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "GLY".to_string(),
                seq_num: 2,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
        ];
        assert_eq!(classify_chain_type(&residues), MoleculeType::Protein);
    }

    #[test]
    fn test_classify_chain_type_rna() {
        let residues = vec![
            Residue {
                name: "A".to_string(),
                seq_num: 1,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "U".to_string(),
                seq_num: 2,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "G".to_string(),
                seq_num: 3,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "C".to_string(),
                seq_num: 4,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
        ];
        assert_eq!(classify_chain_type(&residues), MoleculeType::RNA);
    }

    #[test]
    fn test_classify_chain_type_dna() {
        let residues = vec![
            Residue {
                name: "DA".to_string(),
                seq_num: 1,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "DT".to_string(),
                seq_num: 2,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "DG".to_string(),
                seq_num: 3,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "DC".to_string(),
                seq_num: 4,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
        ];
        assert_eq!(classify_chain_type(&residues), MoleculeType::DNA);
    }

    #[test]
    fn test_classify_chain_type_empty() {
        let residues: Vec<Residue> = vec![];
        assert_eq!(classify_chain_type(&residues), MoleculeType::Protein);
    }

    #[test]
    fn test_classify_chain_type_mixed_majority_rna() {
        // 3 RNA residues, 1 DNA residue -> RNA wins
        let residues = vec![
            Residue {
                name: "A".to_string(),
                seq_num: 1,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "U".to_string(),
                seq_num: 2,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "G".to_string(),
                seq_num: 3,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "DA".to_string(),
                seq_num: 4,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
        ];
        assert_eq!(classify_chain_type(&residues), MoleculeType::RNA);
    }

    #[test]
    fn test_backbone_detection_ca() {
        let atom = Atom {
            name: "CA".to_string(),
            element: "C".to_string(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            b_factor: 0.0,
            is_backbone: true,
            is_hetero: false,
        };
        assert!(atom.is_backbone);
    }

    #[test]
    fn test_backbone_detection_c4prime() {
        // C4' should be a backbone atom for nucleic acids
        let name = "C4'";
        let is_backbone = name == "CA" || name == "C4'";
        assert!(is_backbone);
    }

    #[test]
    fn test_non_backbone_atom() {
        let name = "CB";
        let is_backbone = name == "CA" || name == "C4'";
        assert!(!is_backbone);
    }

    #[test]
    fn test_classify_chain_type_dna_thymine_only() {
        // A chain with only "T" residues should be classified as DNA
        let residues = vec![
            Residue {
                name: "T".to_string(),
                seq_num: 1,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "T".to_string(),
                seq_num: 2,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
            Residue {
                name: "T".to_string(),
                seq_num: 3,
                atoms: vec![],
                secondary_structure: SecondaryStructure::Coil,
            },
        ];
        assert_eq!(classify_chain_type(&residues), MoleculeType::DNA);
    }

    #[test]
    fn test_nmr_multimodel_loads_single_model() {
        // 2KGP is an NMR RNA structure with 10 MODEL records.
        // The parser should only load the first model, not duplicate
        // chains/atoms across all 10 models.
        let protein = load_structure("examples/2KGP.pdb").expect("Failed to load 2KGP.pdb");

        // Should have exactly 1 chain (not 10 duplicated chains)
        assert_eq!(
            protein.chains.len(),
            1,
            "NMR multi-model file should produce 1 chain, got {}",
            protein.chains.len()
        );

        // The single chain should be classified as RNA
        assert_eq!(protein.chains[0].molecule_type, MoleculeType::RNA);

        // Atom count should be reasonable for a single model (~500-900),
        // not inflated by 10x from all models (~8590)
        let total_atoms: usize = protein.chains[0]
            .residues
            .iter()
            .map(|r| r.atoms.len())
            .sum();
        assert!(
            total_atoms < 1000,
            "Expected < 1000 atoms for single NMR model, got {} (multi-model duplication?)",
            total_atoms
        );
    }

    #[test]
    fn test_single_model_pdb_unaffected() {
        // 1UBQ is a single-model X-ray protein structure (ubiquitin).
        // Verify it still loads correctly after NMR multi-model handling.
        let protein = load_structure("examples/1UBQ.pdb").expect("Failed to load 1UBQ.pdb");

        // Ubiquitin has 1 chain (chain A)
        assert_eq!(
            protein.chains.len(),
            1,
            "1UBQ should have 1 chain, got {}",
            protein.chains.len()
        );

        // It should be classified as a protein
        assert_eq!(protein.chains[0].molecule_type, MoleculeType::Protein);

        // Ubiquitin has 76 amino acid residues; crystallographic water
        // molecules (HOH) are now filtered out during parsing.
        assert!(
            protein.chains[0].residues.len() >= 70 && protein.chains[0].residues.len() <= 80,
            "Expected ~76 residues for ubiquitin (waters filtered), got {}",
            protein.chains[0].residues.len()
        );
    }

    #[test]
    fn test_water_filtered_from_ubiquitin() {
        // 1UBQ has 76 amino acid residues and ~58 HOH waters.
        // After filtering, only the 76 AA residues should remain in chains,
        // and no ligands should be present (HOH is discarded, not a ligand).
        let protein = load_structure("examples/1UBQ.pdb").expect("Failed to load 1UBQ.pdb");

        let residue_count = protein.residue_count();
        assert!(
            residue_count >= 70 && residue_count <= 80,
            "Expected ~76 residues after water filtering, got {}",
            residue_count
        );

        // 1UBQ has no real ligands (only HOH waters as HETATM)
        assert_eq!(
            protein.ligand_count(),
            0,
            "Expected 0 ligands for 1UBQ (only waters), got {}",
            protein.ligand_count()
        );
    }

    #[test]
    fn test_4hhb_ligand_parsing() {
        // 4HHB is hemoglobin with 4 HEM (heme) ligands and ions
        let protein = load_structure("examples/4HHB.pdb").expect("Failed to load 4HHB.pdb");

        // Should have 4 protein chains
        assert_eq!(protein.chains.len(), 4);

        // Should have ligands (HEM groups and possibly PO4/ions)
        assert!(protein.ligand_count() > 0, "4HHB should have ligands");

        // At least the 4 HEM groups should be present
        let hem_count = protein.ligands.iter().filter(|l| l.name == "HEM").count();
        assert!(
            hem_count >= 4,
            "Expected at least 4 HEM ligands, got {}",
            hem_count
        );

        // HEM should be classified as Ligand (not Ion) since it's multi-atom
        for l in protein.ligands.iter().filter(|l| l.name == "HEM") {
            assert_eq!(
                l.ligand_type,
                LigandType::Ligand,
                "HEM should be Ligand type"
            );
        }
    }

    #[test]
    fn test_ion_classification() {
        // Verify COMMON_IONS contains expected ions
        assert!(COMMON_IONS.contains(&"ZN"), "ZN should be a common ion");
        assert!(COMMON_IONS.contains(&"MG"), "MG should be a common ion");
        assert!(COMMON_IONS.contains(&"CA"), "CA should be a common ion");
        // ATP is not a single-atom ion
        assert!(
            !COMMON_IONS.contains(&"ATP"),
            "ATP should not be a common ion"
        );
    }
}
