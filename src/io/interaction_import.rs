use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// JSON schema matching examples/interactions.json
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImportedInteractionFile {
    pub metadata: Option<InteractionMetadata>,
    pub interactions: InteractionGroups,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InteractionMetadata {
    pub protein_file: Option<String>,
    pub ligand_file: Option<String>,
    pub binding_energy_kcal_mol: Option<String>,
    pub summary: Option<HashMap<String, serde_json::Value>>,
    pub total_interactions: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct InteractionGroups {
    #[serde(default)]
    pub hydrogen_bonds: Vec<ImportedInteraction>,
    #[serde(default)]
    pub hydrophobic_contacts: Vec<ImportedInteraction>,
    #[serde(default)]
    pub pi_stacking: Vec<ImportedInteraction>,
    #[serde(default)]
    pub pi_cation: Vec<ImportedInteraction>,
    #[serde(default)]
    pub salt_bridges: Vec<ImportedInteraction>,
    #[serde(default)]
    pub halogen_bonds: Vec<ImportedInteraction>,
    #[serde(default)]
    pub metal_coordination: Vec<ImportedInteraction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImportedInteraction {
    /// "hydrogen_bond", "hydrophobic_contact", "pi_stacking", etc.
    #[serde(rename = "type")]
    pub interaction_type: String,
    pub subtype: Option<String>,
    /// e.g. "Lig:S1"
    pub lig_atom: Option<String>,
    /// e.g. "A:ASP171:O"
    pub prot_atom: Option<String>,
    /// e.g. "A:ASP171"
    pub prot_res: Option<String>,
    #[serde(rename = "distance_A")]
    pub distance_a: Option<f32>,
    pub strength: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsed / normalised form used by the renderer
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ExtInteractionType {
    HydrogenBond,
    Hydrophobic,
    PiStacking,
    PiCation,
    SaltBridge,
    HalogenBond,
    MetalCoordination,
}

impl ExtInteractionType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            s if s.contains("hydrogen") => Self::HydrogenBond,
            s if s.contains("hydrophob") => Self::Hydrophobic,
            s if s.contains("pi_stack") || s.contains("pi-stack") => Self::PiStacking,
            s if s.contains("pi_cation") || s.contains("pi-cation") => Self::PiCation,
            s if s.contains("salt") => Self::SaltBridge,
            s if s.contains("halogen") => Self::HalogenBond,
            s if s.contains("metal") => Self::MetalCoordination,
            _ => Self::HydrogenBond,
        }
    }
}

/// A single normalised interaction entry, ready for rendering.
#[derive(Debug, Clone)]
pub struct ExtInteraction {
    pub itype: ExtInteractionType,
    /// Ligand atom label, e.g. "S1" (stripped of "Lig:" prefix)
    pub lig_atom: String,
    /// Residue label, e.g. "ASP171 (A)" — display format
    pub res_label: String,
    pub distance: f32,
    pub strength: Option<String>,
}

/// The full parsed external interaction dataset.
#[derive(Debug, Clone)]
pub struct ExternalInteractionData {
    pub source_file: String,
    pub interactions: Vec<ExtInteraction>,
    pub metadata_summary: String,
}

impl ExternalInteractionData {
    /// Parse from a JSON string.
    pub fn from_json(json: &str, filename: &str) -> Result<Self, String> {
        let file: ImportedInteractionFile =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;

        let mut interactions = Vec::new();

        let all_groups: Vec<(&str, &Vec<ImportedInteraction>)> = vec![
            ("hydrogen_bond",       &file.interactions.hydrogen_bonds),
            ("hydrophobic_contact", &file.interactions.hydrophobic_contacts),
            ("pi_stacking",         &file.interactions.pi_stacking),
            ("pi_cation",           &file.interactions.pi_cation),
            ("salt_bridge",         &file.interactions.salt_bridges),
            ("halogen_bond",        &file.interactions.halogen_bonds),
            ("metal_coordination",  &file.interactions.metal_coordination),
        ];

        for (type_hint, group) in all_groups {
            for entry in group.iter() {
                let itype = ExtInteractionType::from_str(
                    entry.interaction_type.as_str()
                        .if_empty_then(type_hint),
                );

                // lig_atom: strip "Lig:" prefix
                let lig_atom = entry.lig_atom.as_deref().unwrap_or("?")
                    .trim_start_matches("Lig:")
                    .to_string();

                // prot_res: "A:ASP171" → "ASP171 (A)"
                let res_label = parse_res_label(
                    entry.prot_res.as_deref().unwrap_or("?"),
                );

                interactions.push(ExtInteraction {
                    itype,
                    lig_atom,
                    res_label,
                    distance: entry.distance_a.unwrap_or(0.0),
                    strength: entry.strength.clone(),
                });
            }
        }

        let metadata_summary = if let Some(meta) = &file.metadata {
            format!(
                "{} — ΔG {}",
                meta.protein_file.as_deref().unwrap_or("unknown"),
                meta.binding_energy_kcal_mol.as_deref().unwrap_or("?"),
            )
        } else {
            filename.to_string()
        };

        Ok(Self {
            source_file: filename.to_string(),
            interactions,
            metadata_summary,
        })
    }
}

/// "A:ASP171" or "A:ASP171:O" → "ASP171 (A)"
/// ":HOH50" → "HOH50"
fn parse_res_label(raw: &str) -> String {
    let parts: Vec<&str> = raw.split(':').collect();
    match parts.as_slice() {
        [chain, resname] => {
            if chain.is_empty() {
                resname.to_string()
            } else {
                format!("{} ({})", resname, chain)
            }
        }
        [chain, resname, _atom] => {
            if chain.is_empty() {
                resname.to_string()
            } else {
                format!("{} ({})", resname, chain)
            }
        }
        _ => raw.to_string(),
    }
}

trait StrExt {
    fn if_empty_then<'a>(&'a self, fallback: &'a str) -> &'a str;
}
impl StrExt for str {
    fn if_empty_then<'a>(&'a self, fallback: &'a str) -> &'a str {
        if self.is_empty() { fallback } else { self }
    }
}
