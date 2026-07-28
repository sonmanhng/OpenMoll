//! Pairwise and multiple sequence alignment (Needleman–Wunsch, Smith–Waterman),
//! BLOSUM62 scoring, and identity / similarity matrices.

const GAP_OPEN: i32 = -10;
const GAP_EXTEND: i32 = -1;

/// Standard 20 amino-acid order for BLOSUM62 indexing.
const AA_ORDER: [char; 20] = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
    'V',
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairwiseAlgorithm {
    NeedlemanWunsch,
    SmithWaterman,
}

#[derive(Debug, Clone)]
pub struct PairwiseAlignment {
    pub aligned_a: String,
    pub aligned_b: String,
    pub score: i32,
    /// Columns where both sequences have a residue (no gap).
    pub matched_columns: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct PairwiseStats {
    pub identity: f64,
    pub similarity: f64,
    pub aligned_length: usize,
    pub matches: usize,
    pub similar: usize,
    pub gaps: usize,
}

#[derive(Debug, Clone)]
pub struct SequenceRecord {
    pub label: String,
    pub sequence: String,
}

#[derive(Debug, Clone)]
pub struct MultipleSequenceAlignment {
    pub labels: Vec<String>,
    pub rows: Vec<String>,
    pub consensus: String,
}

#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    pub labels: Vec<String>,
    pub identity: Vec<Vec<f64>>,
    pub similarity: Vec<Vec<f64>>,
}

pub fn needleman_wunsch(a: &str, b: &str) -> PairwiseAlignment {
    pairwise_align(a, b, PairwiseAlgorithm::NeedlemanWunsch)
}

pub fn smith_waterman(a: &str, b: &str) -> PairwiseAlignment {
    pairwise_align(a, b, PairwiseAlgorithm::SmithWaterman)
}

pub fn pairwise_stats(alignment: &PairwiseAlignment) -> PairwiseStats {
    let cols = alignment.aligned_a.chars().zip(alignment.aligned_b.chars());
    let mut matches = 0usize;
    let mut similar = 0usize;
    let mut aligned_length = 0usize;
    let mut gaps = 0usize;

    for (ca, cb) in cols {
        aligned_length += 1;
        if ca == '-' || cb == '-' {
            gaps += 1;
            continue;
        }
        if ca == cb {
            matches += 1;
            similar += 1;
        } else if blosum62(ca, cb) > 0 {
            similar += 1;
        }
    }

    let denom = aligned_length.max(1) as f64;
    PairwiseStats {
        identity: matches as f64 / denom,
        similarity: similar as f64 / denom,
        aligned_length,
        matches,
        similar,
        gaps,
    }
}

pub fn similarity_matrix(sequences: &[SequenceRecord]) -> SimilarityMatrix {
    let n = sequences.len();
    let mut identity = vec![vec![0.0; n]; n];
    let mut similarity = vec![vec![0.0; n]; n];
    let labels: Vec<String> = sequences.iter().map(|s| s.label.clone()).collect();

    for i in 0..n {
        identity[i][i] = 1.0;
        similarity[i][i] = 1.0;
        for j in (i + 1)..n {
            let aln = needleman_wunsch(&sequences[i].sequence, &sequences[j].sequence);
            let stats = pairwise_stats(&aln);
            identity[i][j] = stats.identity;
            identity[j][i] = stats.identity;
            similarity[i][j] = stats.similarity;
            similarity[j][i] = stats.similarity;
        }
    }

    SimilarityMatrix {
        labels,
        identity,
        similarity,
    }
}

/// Progressive pairwise merging MSA (representative-based gap propagation).
pub fn progressive_msa(sequences: &[SequenceRecord]) -> MultipleSequenceAlignment {
    if sequences.is_empty() {
        return MultipleSequenceAlignment {
            labels: Vec::new(),
            rows: Vec::new(),
            consensus: String::new(),
        };
    }

    if sequences.len() == 1 {
        return MultipleSequenceAlignment {
            labels: vec![sequences[0].label.clone()],
            rows: vec![sequences[0].sequence.clone()],
            consensus: sequences[0].sequence.clone(),
        };
    }

    let mut groups: Vec<(String, Vec<String>)> = sequences
        .iter()
        .map(|s| (s.label.clone(), vec![s.sequence.clone()]))
        .collect();

    while groups.len() > 1 {
        let mut best = (0usize, 1usize, -1.0f64);
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let rep_i = groups[i].1[0].replace('-', "");
                let rep_j = groups[j].1[0].replace('-', "");
                let aln = needleman_wunsch(&rep_i, &rep_j);
                let id = pairwise_stats(&aln).identity;
                if id > best.2 {
                    best = (i, j, id);
                }
            }
        }

        let (i, j, _) = best;
        let group_b = groups.remove(j);
        let group_a = &groups[i];
        let merged = merge_groups(&group_a.1, &group_b.1);
        let label = format!("{}+{}", group_a.0, group_b.0);
        groups[i] = (label, merged);
    }

    let rows = groups[0].1.clone();

    // Re-label rows by matching ungapped sequence content to input labels.
    let mut out_labels = Vec::with_capacity(rows.len());
    for row in &rows {
        let row_u = row.replace('-', "");
        let label = sequences
            .iter()
            .find(|s| s.sequence == row_u)
            .map(|s| s.label.clone())
            .unwrap_or_else(|| row_u.chars().take(12).collect());
        out_labels.push(label);
    }

    let consensus = consensus_sequence(&rows);
    MultipleSequenceAlignment {
        labels: out_labels,
        rows,
        consensus,
    }
}

fn merge_groups(a: &[String], b: &[String]) -> Vec<String> {
    let rep_a = a[0].replace('-', "");
    let rep_b = b[0].replace('-', "");
    let aln = needleman_wunsch(&rep_a, &rep_b);
    let new_a: Vec<String> = a
        .iter()
        .map(|row| map_gaps(row, &a[0], &aln.aligned_a))
        .collect();
    let new_b: Vec<String> = b
        .iter()
        .map(|row| map_gaps(row, &b[0], &aln.aligned_b))
        .collect();
    new_a.into_iter().chain(new_b).collect()
}

fn map_gaps(row: &str, rep: &str, new_rep: &str) -> String {
    let mut row_chars = Vec::new();
    for (r, p) in row.chars().zip(rep.chars()) {
        if p != '-' {
            row_chars.push(r);
        }
    }
    let mut out = String::new();
    let mut idx = 0usize;
    for c in new_rep.chars() {
        if c == '-' {
            out.push('-');
        } else {
            let ch = row_chars.get(idx).copied().unwrap_or('X');
            out.push(ch);
            idx += 1;
        }
    }
    out
}

pub fn consensus_sequence(rows: &[String]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let width = rows[0].len();
    let mut out = String::with_capacity(width);
    for col in 0..width {
        let mut counts = std::collections::BTreeMap::<char, usize>::new();
        for row in rows {
            if let Some(c) = row.chars().nth(col) {
                if c != '-' {
                    *counts.entry(c).or_insert(0) += 1;
                }
            }
        }
        if counts.is_empty() {
            out.push('-');
        } else if let Some((&ch, _)) = counts.iter().max_by_key(|(_, n)| *n) {
            out.push(ch);
        }
    }
    out
}

fn pairwise_align(a: &str, b: &str, algo: PairwiseAlgorithm) -> PairwiseAlignment {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let n = a_chars.len();
    let m = b_chars.len();

    if n == 0 && m == 0 {
        return PairwiseAlignment {
            aligned_a: String::new(),
            aligned_b: String::new(),
            score: 0,
            matched_columns: Vec::new(),
        };
    }

    match algo {
        PairwiseAlgorithm::NeedlemanWunsch => nw_align(&a_chars, &b_chars),
        PairwiseAlgorithm::SmithWaterman => sw_align(&a_chars, &b_chars),
    }
}

fn nw_align(a: &[char], b: &[char]) -> PairwiseAlignment {
    let n = a.len();
    let m = b.len();
    let mut score = vec![vec![0i32; m + 1]; n + 1];
    let mut trace = vec![vec![0u8; m + 1]; n + 1]; // 0 diag, 1 up, 2 left

    for i in 1..=n {
        score[i][0] = GAP_OPEN + GAP_EXTEND * (i as i32 - 1);
        trace[i][0] = 1;
    }
    for j in 1..=m {
        score[0][j] = GAP_OPEN + GAP_EXTEND * (j as i32 - 1);
        trace[0][j] = 2;
    }

    for i in 1..=n {
        for j in 1..=m {
            let match_score = score[i - 1][j - 1] + blosum62(a[i - 1], b[j - 1]);
            let delete = score[i - 1][j] + gap_penalty(&score, i, j, true);
            let insert = score[i][j - 1] + gap_penalty(&score, i, j, false);
            let (best, dir) = best_of3(match_score, delete, insert);
            score[i][j] = best;
            trace[i][j] = dir;
        }
    }

    traceback(
        a,
        b,
        &trace,
        n,
        m,
        score[n][m],
        PairwiseAlgorithm::NeedlemanWunsch,
    )
}

fn sw_align(a: &[char], b: &[char]) -> PairwiseAlignment {
    let n = a.len();
    let m = b.len();
    let mut score = vec![vec![0i32; m + 1]; n + 1];
    let mut trace = vec![vec![0u8; m + 1]; n + 1];

    let mut max_score = 0i32;
    let mut max_i = 0usize;
    let mut max_j = 0usize;

    for i in 1..=n {
        for j in 1..=m {
            let match_score = score[i - 1][j - 1] + blosum62(a[i - 1], b[j - 1]);
            let delete = score[i - 1][j] + GAP_EXTEND;
            let insert = score[i][j - 1] + GAP_EXTEND;
            let mut best = 0i32;
            let mut dir = 3u8; // stop
            if match_score > best {
                best = match_score;
                dir = 0;
            }
            if delete > best {
                best = delete;
                dir = 1;
            }
            if insert > best {
                best = insert;
                dir = 2;
            }
            score[i][j] = best;
            trace[i][j] = dir;
            if best > max_score {
                max_score = best;
                max_i = i;
                max_j = j;
            }
        }
    }

    if max_score == 0 {
        return PairwiseAlignment {
            aligned_a: a.iter().collect(),
            aligned_b: b.iter().collect(),
            score: 0,
            matched_columns: (0..n.min(m)).map(|i| (i, i)).collect(),
        };
    }

    traceback(
        a,
        b,
        &trace,
        max_i,
        max_j,
        max_score,
        PairwiseAlgorithm::SmithWaterman,
    )
}

fn traceback(
    a: &[char],
    b: &[char],
    trace: &[Vec<u8>],
    mut i: usize,
    mut j: usize,
    final_score: i32,
    algo: PairwiseAlgorithm,
) -> PairwiseAlignment {
    let mut aligned_a_rev = Vec::new();
    let mut aligned_b_rev = Vec::new();
    let mut matched_rev = Vec::new();

    loop {
        if i == 0 && j == 0 {
            break;
        }
        if algo == PairwiseAlgorithm::SmithWaterman && trace[i][j] == 3 {
            break;
        }

        let dir = if i == 0 {
            2u8
        } else if j == 0 {
            1u8
        } else {
            trace[i][j]
        };

        match dir {
            0 => {
                aligned_a_rev.push(a[i - 1]);
                aligned_b_rev.push(b[j - 1]);
                matched_rev.push((i - 1, j - 1));
                i -= 1;
                j -= 1;
            }
            1 => {
                aligned_a_rev.push(a[i - 1]);
                aligned_b_rev.push('-');
                i -= 1;
            }
            _ => {
                aligned_a_rev.push('-');
                aligned_b_rev.push(b[j - 1]);
                j -= 1;
            }
        }
    }

    aligned_a_rev.reverse();
    aligned_b_rev.reverse();
    matched_rev.reverse();

    PairwiseAlignment {
        aligned_a: aligned_a_rev.iter().collect(),
        aligned_b: aligned_b_rev.iter().collect(),
        score: final_score,
        matched_columns: matched_rev,
    }
}

fn gap_penalty(score: &[Vec<i32>], i: usize, j: usize, from_above: bool) -> i32 {
    if from_above {
        if i > 1 && score[i - 1][j] == score[i - 2][j] + GAP_EXTEND {
            GAP_EXTEND
        } else {
            GAP_OPEN
        }
    } else if j > 1 && score[i][j - 1] == score[i][j - 2] + GAP_EXTEND {
        GAP_EXTEND
    } else {
        GAP_OPEN
    }
}

fn best_of3(a: i32, b: i32, c: i32) -> (i32, u8) {
    if a >= b && a >= c {
        (a, 0)
    } else if b >= c {
        (b, 1)
    } else {
        (c, 2)
    }
}

fn aa_index(c: char) -> Option<usize> {
    let u = c.to_ascii_uppercase();
    AA_ORDER.iter().position(|&x| x == u)
}

fn blosum62(a: char, b: char) -> i32 {
    let Some(i) = aa_index(a) else { return -4 };
    let Some(j) = aa_index(b) else { return -4 };
    BLOSUM62[i][j]
}

/// BLOSUM62 matrix (Henikoff & Henikoff).
const BLOSUM62: [[i32; 20]; 20] = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nw_identical_sequences() {
        let aln = needleman_wunsch("ACDEFG", "ACDEFG");
        assert_eq!(aln.aligned_a, "ACDEFG");
        assert_eq!(aln.aligned_b, "ACDEFG");
        let stats = pairwise_stats(&aln);
        assert!((stats.identity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nw_inserts_gaps_for_indels() {
        let aln = needleman_wunsch("ACDEFG", "ACDDEFG");
        let stats = pairwise_stats(&aln);
        assert!(stats.identity > 0.85);
        assert!(aln.aligned_a.contains('-') || aln.aligned_b.contains('-'));
    }

    #[test]
    fn smith_waterman_finds_local_motif() {
        let aln = smith_waterman(
            "AAAAACDEFGAAAA",
            "ZZZACDEFGZZZZ",
        );
        let stats = pairwise_stats(&aln);
        assert!(stats.identity > 0.9);
        assert!(aln.aligned_a.chars().filter(|&c| c != '-').count() <= 6);
    }

    #[test]
    fn similarity_matrix_symmetry() {
        let seqs = vec![
            SequenceRecord {
                label: "A".into(),
                sequence: "ACDEFG".into(),
            },
            SequenceRecord {
                label: "B".into(),
                sequence: "ACDDEFG".into(),
            },
        ];
        let mat = similarity_matrix(&seqs);
        assert!((mat.identity[0][1] - mat.identity[1][0]).abs() < 1e-9);
    }

    #[test]
    fn progressive_msa_three_sequences() {
        let seqs = vec![
            SequenceRecord {
                label: "s1".into(),
                sequence: "ACDEFG".into(),
            },
            SequenceRecord {
                label: "s2".into(),
                sequence: "ACDDEFG".into(),
            },
            SequenceRecord {
                label: "s3".into(),
                sequence: "ACDEFGH".into(),
            },
        ];
        let msa = progressive_msa(&seqs);
        assert_eq!(msa.rows.len(), 3);
        assert!(msa.rows.iter().all(|r| r.len() == msa.rows[0].len()));
    }
}
