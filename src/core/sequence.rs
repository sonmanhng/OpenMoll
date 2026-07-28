use std::collections::BTreeMap;

use crate::core::protein::{Chain, MoleculeType, Protein, SecondaryStructure};

/// One-letter sequence for a polymer chain (gaps not included).
pub fn chain_sequence(chain: &Chain) -> String {
    chain
        .residues
        .iter()
        .map(|res| residue_code(&res.name, chain.molecule_type))
        .collect()
}

#[derive(Debug, Clone)]
pub struct ChainSequenceAnalysis {
    pub chain_id: String,
    pub molecule_type: MoleculeType,
    pub sequence: String,
    pub residue_count: usize,
    pub atom_count: usize,
    pub hydrophobic_fraction: f32,
    pub charged_fraction: f32,
    pub polar_fraction: f32,
    pub aromatic_fraction: f32,
    pub estimated_mw_da: f32,
    pub helix_fraction: f32,
    pub sheet_fraction: f32,
    pub coil_fraction: f32,
    pub composition: BTreeMap<char, usize>,
    pub motifs: Vec<SequenceMotif>,
}

#[derive(Debug, Clone)]
pub struct SequenceMotif {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub sequence: String,
}

pub fn analyze_sequences(protein: &Protein) -> Vec<ChainSequenceAnalysis> {
    protein
        .chains
        .iter()
        .map(|chain| {
            let sequence: String = chain
                .residues
                .iter()
                .map(|res| residue_code(&res.name, chain.molecule_type))
                .collect();
            let residue_count = chain.residues.len();
            let atom_count = chain.residues.iter().map(|res| res.atoms.len()).sum();
            let composition = composition(&sequence);

            let hydrophobic = count_matches(&sequence, is_hydrophobic_code);
            let charged = count_matches(&sequence, is_charged_code);
            let polar = count_matches(&sequence, is_polar_code);
            let aromatic = count_matches(&sequence, is_aromatic_code);

            let mut helix = 0usize;
            let mut sheet = 0usize;
            let mut coil = 0usize;
            for residue in &chain.residues {
                match residue.secondary_structure {
                    SecondaryStructure::Helix => helix += 1,
                    SecondaryStructure::Sheet => sheet += 1,
                    SecondaryStructure::Turn | SecondaryStructure::Coil => coil += 1,
                }
            }

            ChainSequenceAnalysis {
                chain_id: chain.id.clone(),
                molecule_type: chain.molecule_type,
                sequence: sequence.clone(),
                residue_count,
                atom_count,
                hydrophobic_fraction: fraction(hydrophobic, residue_count),
                charged_fraction: fraction(charged, residue_count),
                polar_fraction: fraction(polar, residue_count),
                aromatic_fraction: fraction(aromatic, residue_count),
                estimated_mw_da: estimate_molecular_weight(&sequence, chain.molecule_type),
                helix_fraction: fraction(helix, residue_count),
                sheet_fraction: fraction(sheet, residue_count),
                coil_fraction: fraction(coil, residue_count),
                composition,
                motifs: find_motifs(&sequence, chain.molecule_type),
            }
        })
        .collect()
}

pub fn format_fasta(analysis: &ChainSequenceAnalysis, structure_name: &str) -> String {
    let mut out = format!(
        ">{}|chain {}|{:?}|{} residues\n",
        structure_name, analysis.chain_id, analysis.molecule_type, analysis.residue_count
    );
    for chunk in analysis.sequence.as_bytes().chunks(80) {
        out.push_str(&String::from_utf8_lossy(chunk));
        out.push('\n');
    }
    out
}

fn composition(sequence: &str) -> BTreeMap<char, usize> {
    let mut map = BTreeMap::new();
    for code in sequence.chars() {
        *map.entry(code).or_insert(0) += 1;
    }
    map
}

fn find_motifs(sequence: &str, molecule_type: MoleculeType) -> Vec<SequenceMotif> {
    match molecule_type {
        MoleculeType::Protein => find_protein_motifs(sequence),
        MoleculeType::RNA | MoleculeType::DNA => find_nucleic_acid_motifs(sequence),
        MoleculeType::SmallMolecule => Vec::new(),
    }
}

fn find_protein_motifs(sequence: &str) -> Vec<SequenceMotif> {
    let chars: Vec<char> = sequence.chars().collect();
    let mut motifs = Vec::new();

    for i in 0..chars.len().saturating_sub(2) {
        if chars[i] == 'N' && chars[i + 1] != 'P' && matches!(chars[i + 2], 'S' | 'T') {
            motifs.push(motif("N-glycosylation sequon", i, i + 3, &chars));
        }
    }

    for i in 0..chars.len().saturating_sub(3) {
        if matches!(chars[i], 'S' | 'T') && chars[i + 2] == 'P' {
            motifs.push(motif("Proline-directed phosphorylation", i, i + 4, &chars));
        }
    }

    for i in 0..chars.len().saturating_sub(4) {
        let acidic_nearby = chars[i + 1..=i + 4]
            .iter()
            .filter(|&&c| matches!(c, 'D' | 'E'))
            .count();
        if matches!(chars[i], 'S' | 'T') && acidic_nearby >= 2 {
            motifs.push(motif("Acidic kinase site candidate", i, i + 5, &chars));
        }
    }

    motifs
}

fn find_nucleic_acid_motifs(sequence: &str) -> Vec<SequenceMotif> {
    let chars: Vec<char> = sequence.chars().collect();
    let mut motifs = Vec::new();
    for i in 0..chars.len().saturating_sub(5) {
        let window: String = chars[i..i + 6].iter().collect();
        if window == "AATAAA" || window == "AAUAAA" {
            motifs.push(SequenceMotif {
                name: "Polyadenylation signal candidate".into(),
                start: i + 1,
                end: i + 6,
                sequence: window,
            });
        }
    }
    motifs
}

fn motif(name: &str, start: usize, end: usize, chars: &[char]) -> SequenceMotif {
    SequenceMotif {
        name: name.into(),
        start: start + 1,
        end,
        sequence: chars[start..end].iter().collect(),
    }
}

fn residue_code(name: &str, molecule_type: MoleculeType) -> char {
    match molecule_type {
        MoleculeType::Protein | MoleculeType::SmallMolecule => amino_acid_code(name),
        MoleculeType::RNA | MoleculeType::DNA => nucleotide_code(name),
    }
}

fn amino_acid_code(name: &str) -> char {
    match name.trim() {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        "SEC" => 'U',
        "PYL" => 'O',
        _ => 'X',
    }
}

fn nucleotide_code(name: &str) -> char {
    match name.trim() {
        "A" | "DA" | "AMP" => 'A',
        "U" | "UMP" => 'U',
        "T" | "DT" => 'T',
        "G" | "DG" | "GMP" => 'G',
        "C" | "DC" | "CMP" => 'C',
        "I" | "DI" => 'I',
        _ => 'N',
    }
}

fn estimate_molecular_weight(sequence: &str, molecule_type: MoleculeType) -> f32 {
    sequence
        .chars()
        .map(|code| match molecule_type {
            MoleculeType::Protein | MoleculeType::SmallMolecule => amino_acid_mass(code),
            MoleculeType::RNA | MoleculeType::DNA => nucleotide_mass(code),
        })
        .sum()
}

fn amino_acid_mass(code: char) -> f32 {
    match code {
        'A' => 89.09,
        'R' => 174.20,
        'N' => 132.12,
        'D' => 133.10,
        'C' => 121.16,
        'Q' => 146.15,
        'E' => 147.13,
        'G' => 75.07,
        'H' => 155.16,
        'I' | 'L' => 131.17,
        'K' => 146.19,
        'M' => 149.21,
        'F' => 165.19,
        'P' => 115.13,
        'S' => 105.09,
        'T' => 119.12,
        'W' => 204.23,
        'Y' => 181.19,
        'V' => 117.15,
        _ => 110.0,
    }
}

fn nucleotide_mass(code: char) -> f32 {
    match code {
        'A' => 331.2,
        'C' => 307.2,
        'G' => 347.2,
        'T' => 322.2,
        'U' => 306.2,
        _ => 320.0,
    }
}

fn count_matches(sequence: &str, predicate: fn(char) -> bool) -> usize {
    sequence.chars().filter(|&c| predicate(c)).count()
}

fn fraction(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}

fn is_hydrophobic_code(code: char) -> bool {
    matches!(code, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y' | 'P')
}

fn is_charged_code(code: char) -> bool {
    matches!(code, 'D' | 'E' | 'R' | 'K' | 'H')
}

fn is_polar_code(code: char) -> bool {
    matches!(code, 'S' | 'T' | 'N' | 'Q' | 'C' | 'Y' | 'H' | 'W')
}

fn is_aromatic_code(code: char) -> bool {
    matches!(code, 'F' | 'W' | 'Y' | 'H')
}
