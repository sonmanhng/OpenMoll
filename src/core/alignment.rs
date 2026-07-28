use crate::core::protein::{Chain, Protein};
use crate::core::seq_align::{needleman_wunsch, smith_waterman, PairwiseAlignment};
use crate::core::sequence::chain_sequence;
use nalgebra::{Dyn, Matrix3, OMatrix, Vector3, SVD, U3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureAlignmentMode {
    /// Match Cα in file order (legacy behavior).
    OrderBased,
    /// Global Needleman–Wunsch, then superpose matched residues.
    SequenceGlobal,
    /// Local Smith–Waterman, then superpose matched residues.
    SequenceLocal,
}

#[derive(Debug, Clone)]
pub struct SequenceAwareAlignmentResult {
    pub rmsd: f64,
    pub matched_pairs: usize,
    pub sequence_identity: f64,
    pub sequence_similarity: f64,
    pub alignment: PairwiseAlignment,
}

/// Kabsch algorithm: aligns `mobile_points` onto `target_points`.
/// Returns (RMSD, Rotation matrix, Translation vector).
/// The transformation to apply to mobile is: `R * (mobile_point - centroid_m) + centroid_t`
pub fn calculate_kabsch_transform(
    target_points: &[Vector3<f64>],
    mobile_points: &[Vector3<f64>],
) -> Option<(f64, Matrix3<f64>, Vector3<f64>, Vector3<f64>)> {
    if target_points.is_empty() || target_points.len() != mobile_points.len() {
        return None;
    }

    let n = target_points.len() as f64;

    // Calculate centroids
    let mut centroid_t = Vector3::zeros();
    let mut centroid_m = Vector3::zeros();
    for i in 0..target_points.len() {
        centroid_t += target_points[i];
        centroid_m += mobile_points[i];
    }
    centroid_t /= n;
    centroid_m /= n;

    // Center the points
    // Convert to nalgebra dynamic matrices for SVD
    let mut t_centered = OMatrix::<f64, Dyn, U3>::zeros(target_points.len());
    let mut m_centered = OMatrix::<f64, Dyn, U3>::zeros(mobile_points.len());

    for i in 0..target_points.len() {
        let p_t = target_points[i] - centroid_t;
        let p_m = mobile_points[i] - centroid_m;

        t_centered.set_row(i, &p_t.transpose());
        m_centered.set_row(i, &p_m.transpose());
    }

    // Covariance matrix: H = m_centered^T * t_centered
    let h = m_centered.transpose() * t_centered;

    // SVD of H
    let svd = SVD::new(h, true, true);

    let u = svd.u?;
    let v_t = svd.v_t?;

    // R = V * U^T
    let mut r = v_t.transpose() * u.transpose();

    // Check for reflection (det < 0)
    if r.determinant() < 0.0 {
        // Multiply the third column of V by -1
        let mut v_t_corrected = v_t.transpose();
        for i in 0..3 {
            v_t_corrected[(i, 2)] *= -1.0;
        }
        r = v_t_corrected * u.transpose();
    }

    // Calculate RMSD
    let mut rmsd_sq = 0.0;
    for i in 0..target_points.len() {
        let p_m = mobile_points[i] - centroid_m;
        let p_t = target_points[i] - centroid_t;
        let p_m_rot = r * p_m;
        let diff = p_m_rot - p_t;
        rmsd_sq += diff.norm_squared();
    }
    let rmsd = (rmsd_sq / n).sqrt();

    Some((rmsd, r, centroid_t, centroid_m))
}

/// Helper function to extract CA backbone points for alignment (all chains, file order).
pub fn extract_backbone_points(protein: &Protein) -> Vec<Vector3<f64>> {
    let mut points = Vec::new();
    for chain in &protein.chains {
        points.extend(extract_chain_ca_points(chain));
    }
    points
}

/// Cα coordinates for one chain, one point per residue (in order).
pub fn extract_chain_ca_points(chain: &Chain) -> Vec<Vector3<f64>> {
    let mut points = Vec::new();
    for residue in &chain.residues {
        if let Some(atom) = residue.atoms.iter().find(|a| a.name.trim() == "CA") {
            points.push(Vector3::new(atom.x, atom.y, atom.z));
        }
    }
    points
}

pub fn find_chain<'a>(protein: &'a Protein, chain_id: &str) -> Option<&'a Chain> {
    protein
        .chains
        .iter()
        .find(|c| c.id == chain_id)
        .or_else(|| {
            protein
                .chains
                .iter()
                .find(|c| c.id.eq_ignore_ascii_case(chain_id))
        })
}

pub fn default_protein_chain_id(protein: &Protein) -> Option<String> {
    protein
        .chains
        .iter()
        .find(|c| {
            matches!(
                c.molecule_type,
                crate::core::protein::MoleculeType::Protein
            )
        })
        .map(|c| c.id.clone())
        .or_else(|| protein.chains.first().map(|c| c.id.clone()))
}

/// Superpose `mobile` onto `target` using sequence alignment between selected chains.
pub fn align_structures_sequence_aware(
    target: &Protein,
    mobile: &Protein,
    target_chain_id: &str,
    mobile_chain_id: &str,
    mode: StructureAlignmentMode,
) -> Option<(SequenceAwareAlignmentResult, Matrix3<f64>, Vector3<f64>, Vector3<f64>)> {
    let target_chain = find_chain(target, target_chain_id)?;
    let mobile_chain = find_chain(mobile, mobile_chain_id)?;

    let seq_t = chain_sequence(target_chain);
    let seq_m = chain_sequence(mobile_chain);
    if seq_t.is_empty() || seq_m.is_empty() {
        return None;
    }

    let alignment = match mode {
        StructureAlignmentMode::OrderBased => {
            let n = seq_t.len().min(seq_m.len());
            let matched: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
            PairwiseAlignment {
                aligned_a: seq_t.chars().take(n).collect(),
                aligned_b: seq_m.chars().take(n).collect(),
                score: 0,
                matched_columns: matched,
            }
        }
        StructureAlignmentMode::SequenceGlobal => needleman_wunsch(&seq_t, &seq_m),
        StructureAlignmentMode::SequenceLocal => smith_waterman(&seq_t, &seq_m),
    };

    let (target_pts, mobile_pts) =
        ca_points_from_alignment(target_chain, mobile_chain, &alignment.matched_columns);
    if target_pts.len() < 3 {
        return None;
    }

    let (rmsd, r, c_t, c_m) = calculate_kabsch_transform(&target_pts, &mobile_pts)?;
    let stats = crate::core::seq_align::pairwise_stats(&alignment);

    Some((
        SequenceAwareAlignmentResult {
            rmsd,
            matched_pairs: target_pts.len(),
            sequence_identity: stats.identity,
            sequence_similarity: stats.similarity,
            alignment,
        },
        r,
        c_t,
        c_m,
    ))
}

fn ca_points_from_alignment(
    target_chain: &Chain,
    mobile_chain: &Chain,
    matched: &[(usize, usize)],
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let ca_t = extract_chain_ca_points(target_chain);
    let ca_m = extract_chain_ca_points(mobile_chain);
    let mut target_pts = Vec::new();
    let mut mobile_pts = Vec::new();
    for &(i, j) in matched {
        if let (Some(p_t), Some(p_m)) = (ca_t.get(i), ca_m.get(j)) {
            target_pts.push(*p_t);
            mobile_pts.push(*p_m);
        }
    }
    (target_pts, mobile_pts)
}

/// Applies the calculated alignment transformation to a protein's coordinates.
pub fn apply_alignment(
    protein: &mut Protein,
    r: &Matrix3<f64>,
    centroid_t: &Vector3<f64>,
    centroid_m: &Vector3<f64>,
) {
    for chain in &mut protein.chains {
        for residue in &mut chain.residues {
            for atom in &mut residue.atoms {
                let p = Vector3::new(atom.x, atom.y, atom.z) - centroid_m;
                let p_rot = r * p;
                let final_p = p_rot + centroid_t;
                atom.x = final_p.x;
                atom.y = final_p.y;
                atom.z = final_p.z;
            }
        }
    }

    for ligand in &mut protein.ligands {
        for atom in &mut ligand.atoms {
            let p = Vector3::new(atom.x, atom.y, atom.z) - centroid_m;
            let p_rot = r * p;
            let final_p = p_rot + centroid_t;
            atom.x = final_p.x;
            atom.y = final_p.y;
            atom.z = final_p.z;
        }
    }
}
