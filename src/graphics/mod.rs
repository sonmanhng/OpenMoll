pub mod ribbon;
pub mod surface;

use crate::core::protein::Protein;
use crate::io::ProteinLoadedEvent;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use std::collections::HashMap;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[derive(Debug, Clone, PartialEq)]
pub struct ChainVizState {
    pub spacefill: bool,
    pub wireframe: bool,
    pub sticks: bool,
    pub ribbon: bool,
    pub surface: bool,
    pub eps_surface: bool,
    pub use_custom_color: bool,
    pub custom_color: [f32; 3],
}

impl Default for ChainVizState {
    fn default() -> Self {
        Self {
            spacefill: false,
            wireframe: false,
            sticks: false,
            ribbon: true, // Ribbon default for protein chains
            surface: false,
            eps_surface: false,
            use_custom_color: false,
            custom_color: [0.8, 0.8, 0.8], // Default light gray
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LigandVizState {
    pub spacefill: bool,
    pub wireframe: bool,
    pub sticks: bool,
    pub surface: bool,
    pub use_custom_color: bool,
    pub custom_color: [f32; 3],
}

impl Default for LigandVizState {
    fn default() -> Self {
        Self {
            spacefill: true, // Spacefill default for ligands
            wireframe: false,
            sticks: false,
            surface: false,
            use_custom_color: false,
            custom_color: [0.3, 0.8, 0.3], // Default green
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VizState {
    pub chain_states: HashMap<String, ChainVizState>,
    pub ligand_states: HashMap<usize, LigandVizState>,

    pub surface_resolution: f32,
    pub surface_iso_level: f32,
    pub show_interactions: bool,
    pub show_2d_map: bool,
    pub selected_ligand_2d: Option<usize>,
    pub hbond_dist_threshold: f32,
    pub hydrophobic_dist_threshold: f32,
}

impl Default for VizState {
    fn default() -> Self {
        Self {
            chain_states: HashMap::new(),
            ligand_states: HashMap::new(),
            surface_resolution: 0.4,
            surface_iso_level: 0.4,
            show_interactions: false,
            show_2d_map: false,
            selected_ligand_2d: None,
            hbond_dist_threshold: 3.5,
            hydrophobic_dist_threshold: 4.0,
        }
    }
}

pub struct ProteinObject {
    pub name: String,
    pub protein: Protein,
    pub viz_state: VizState,
    pub visible: bool,
    pub changed: bool,
    /// Set by MD playback to skip full mesh rebuild and only update transforms
    pub transform_dirty: bool,
    /// Cached scene center computed at spawn time; reused every frame during MD
    pub center: Vec3,
    pub interactions: Vec<crate::core::interactions::Interaction>,
}

#[derive(Resource, Default)]
pub struct ObjectManager {
    pub objects: Vec<ProteinObject>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Molecular Dynamics state
// ─────────────────────────────────────────────────────────────────────────────

/// Playback state for a loaded XTC trajectory.
#[derive(Resource)]
pub struct MdState {
    /// Path to the topology PDB (for display only)
    pub topology_path: Option<std::path::PathBuf>,
    /// Path to the loaded XTC file (for display only)
    pub trajectory_path: Option<std::path::PathBuf>,
    /// Parsed trajectory frames (positions in Å)
    pub trajectory: Option<crate::io::xtc_parser::XtcTrajectory>,
    /// Current frame index
    pub current_frame: usize,
    /// Whether playback is running
    pub is_playing: bool,
    /// Frames per second for playback
    pub playback_speed: f32,
    /// Loop when reaching end
    pub loop_playback: bool,
    /// Accumulated timer (seconds) for frame advance
    pub frame_timer: f32,
    /// Index into ObjectManager.objects to animate
    pub linked_object_idx: Option<usize>,
    /// Snapshot of original atom positions (nm-relative, Å) before MD animation
    pub original_positions: Vec<Vec<Vec<[f32; 3]>>>, // [chain][residue][atom]
    /// Whether original positions have been captured
    pub positions_captured: bool,
    /// Loading status message
    pub status_msg: String,
    /// Whether XTC is currently loading in the background
    pub is_loading: bool,
}

impl Default for MdState {
    fn default() -> Self {
        Self {
            topology_path: None,
            trajectory_path: None,
            trajectory: None,
            current_frame: 0,
            is_playing: false,
            playback_speed: 10.0,
            loop_playback: true,
            frame_timer: 0.0,
            linked_object_idx: None,
            original_positions: Vec::new(),
            positions_captured: false,
            status_msg: "No trajectory loaded.".into(),
            is_loading: false,
        }
    }
}

/// A single picked atom for measurement.
#[derive(Debug, Clone)]
pub struct AtomPickTarget {
    pub world_pos: Vec3,       // position in world space (centered)
    pub label: String,         // e.g. "CA GLY 24 (A)"
}

/// State for the distance measurement tool.
/// Ctrl + right-click picks atoms one at a time.
#[derive(Resource, Default)]
pub struct MeasureState {
    pub first: Option<AtomPickTarget>,
    pub second: Option<AtomPickTarget>,
    /// Computed distance in Ångström, valid when both picks are set.
    pub distance_angstrom: Option<f32>,
}

impl MeasureState {
    pub fn clear(&mut self) {
        self.first = None;
        self.second = None;
        self.distance_angstrom = None;
    }

    pub fn push(&mut self, pick: AtomPickTarget) {
        match (&self.first, &self.second) {
            (None, _) => {
                self.first = Some(pick);
                self.distance_angstrom = None;
            }
            (Some(_), None) => {
                let d = self.first.as_ref().unwrap().world_pos
                    .distance(pick.world_pos);
                self.second = Some(pick);
                self.distance_angstrom = Some(d);
            }
            // Both set → start fresh
            (Some(_), Some(_)) => {
                self.first = Some(pick);
                self.second = None;
                self.distance_angstrom = None;
            }
        }
    }
}

#[derive(Component)]
pub struct AtomMesh {
    pub object_index: usize,
}

#[derive(Component)]
pub struct RibbonRef {
    pub obj_idx: usize,
}

#[derive(Component)]
pub struct SurfaceRef {
    pub obj_idx: usize,
}

/// Marks a sphere entity as representing a specific atom (for MD transform sync).
#[derive(Component)]
pub struct AtomRef {
    pub obj_idx: usize,
    pub serial: i32,
}

/// Marks a bond-cylinder entity (first or second half) for MD transform sync.
#[derive(Component)]
pub struct BondRef {
    pub obj_idx: usize,
    pub serial_a: i32,
    pub serial_b: i32,
    /// 0 = first half (pa → mid), 1 = second half (mid → pb)
    pub half: u8,
}

pub struct GraphicsPlugin;

impl Plugin for GraphicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PanOrbitCameraPlugin)
            .init_resource::<ObjectManager>()
            .init_resource::<MeasureState>()
            .init_resource::<MdState>()
            .add_systems(Startup, setup_scene)
            .add_systems(Update, handle_protein_load)
            .add_systems(Update, tick_md_playback)
            .add_systems(Update, apply_md_frame.after(tick_md_playback))
            .add_systems(Update, sync_md_transforms.after(apply_md_frame))
            .add_systems(Update, spawn_protein_meshes.after(sync_md_transforms))
            .add_systems(Update, draw_interactions_system)
            .add_systems(Update, atom_pick_system)
            .add_systems(Update, update_camera_enabled)
            .add_systems(Update, update_headlight)
            .add_systems(Update, draw_measure_gizmos);
    }
}

fn setup_scene(mut commands: Commands, mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line_width = 3.0; // Make lines thicker and more visible
    config.depth_bias = -1.0; // Draw on top of other meshes

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 500.0,
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-10.0, 10.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        PanOrbitCamera {
            zoom_sensitivity: 1.4,
            zoom_lower_limit: Some(0.5),
            zoom_upper_limit: Some(500.0),
            ..default()
        },
    ));
}

fn handle_protein_load(
    mut loaded_events: EventReader<ProteinLoadedEvent>,
    mut object_manager: ResMut<ObjectManager>,
) {
    for event in loaded_events.read() {
        let name = if event.0.name.is_empty() || event.0.name == "Unknown" {
            format!("obj_{}", object_manager.objects.len() + 1)
        } else {
            event.0.name.clone()
        };
        let mut viz_state = VizState::default();
        for chain in &event.0.chains {
            viz_state
                .chain_states
                .insert(chain.id.clone(), ChainVizState::default());
        }
        for (i, ligand) in event.0.ligands.iter().enumerate() {
            let mut state = LigandVizState::default();
            if crate::core::protein::WATER_NAMES.contains(&ligand.name.as_str()) || 
               crate::core::protein::COMMON_IONS.contains(&ligand.name.as_str()) ||
               ligand.name == "POPC" || ligand.name == "POPE" || ligand.name == "DPPC" || ligand.name == "CHOL"
            {
                state.spacefill = false;
            }
            viz_state.ligand_states.insert(i, state);
        }

        object_manager.objects.push(ProteinObject {
            name,
            protein: event.0.clone(),
            viz_state,
            visible: true,
            changed: true,
            transform_dirty: false,
            center: Vec3::ZERO,
            interactions: Vec::new(),
        });
    }
}

fn spawn_protein_meshes(
    mut commands: Commands,
    mut object_manager: ResMut<ObjectManager>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing_atoms: Query<(Entity, &AtomMesh)>,
) {
    let mut any_changed = false;
    for obj in &object_manager.objects {
        if obj.changed {
            any_changed = true;
            break;
        }
    }

    // If there are no objects but there are still meshes, we must clear them
    if object_manager.objects.is_empty() && !existing_atoms.is_empty() {
        any_changed = true;
    }

    if !any_changed {
        for (_, atom_mesh) in existing_atoms.iter() {
            if atom_mesh.object_index >= object_manager.objects.len() {
                any_changed = true;
                break;
            }
        }
    }

    if !any_changed {
        return;
    }

    for (entity, atom_mesh) in existing_atoms.iter() {
        if atom_mesh.object_index < object_manager.objects.len() {
            if object_manager.objects[atom_mesh.object_index].changed {
                commands.entity(entity).despawn();
            }
        } else {
            commands.entity(entity).despawn();
        }
    }

    let anchor_center = if let Some(first_obj) = object_manager.objects.first() {
        let mut center = Vec3::ZERO;
        let mut count = 0;
        for chain in &first_obj.protein.chains {
            for residue in &chain.residues {
                for atom in &residue.atoms {
                    center += Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32);
                    count += 1;
                }
            }
        }
        if count > 0 { center / count as f32 } else { Vec3::ZERO }
    } else {
        Vec3::ZERO
    };

    for (obj_idx, obj) in object_manager.objects.iter_mut().enumerate() {
        if !obj.changed {
            continue;
        }
        obj.changed = false;
        if !obj.visible {
            continue;
        }

        let protein = &obj.protein;

        // Use the global scene anchor so multiple loaded structures align correctly
        let center = anchor_center;

        // Cache center for use by sync_md_transforms during playback
        obj.center = center;

        // Cache interactions if they are enabled
        if obj.viz_state.show_interactions || obj.viz_state.show_2d_map {
            obj.interactions = crate::core::interactions::detect_interactions(
                &obj.protein,
                obj.viz_state.hbond_dist_threshold,
                obj.viz_state.hydrophobic_dist_threshold,
            );
        } else {
            obj.interactions.clear();
        }

        let mat_c     = materials.add(Color::srgb(0.50, 0.50, 0.50));
        let mat_o     = materials.add(Color::srgb(1.00, 0.00, 0.00));
        let mat_n     = materials.add(Color::srgb(0.00, 0.00, 1.00));
        let mat_s     = materials.add(Color::srgb(1.00, 1.00, 0.00));
        let mat_p     = materials.add(Color::srgb(1.00, 0.50, 0.00));
        let mat_other = materials.add(Color::srgb(0.80, 0.80, 0.80));
        let mat_hoh_o = materials.add(Color::srgb(0.20, 0.60, 1.00)); // distinct water-oxygen color

        // ── Element → material helper (no allocation per atom) ────────────
        macro_rules! elem_mat {
            ($el:expr) => {{
                match $el {
                    'C' => mat_c.clone(),
                    'O' => mat_o.clone(),
                    'N' => mat_n.clone(),
                    'S' => mat_s.clone(),
                    'P' => mat_p.clone(),
                    _   => mat_other.clone(),
                }
            }};
        }

        // Total atom count – used to pick LOD tier
        let total_atoms: usize = protein.chains.iter()
            .flat_map(|c| &c.residues)
            .map(|r| r.atoms.len())
            .sum::<usize>()
            + protein.ligands.iter().map(|l| l.atoms.len()).sum::<usize>();

        // LOD: large systems get simpler meshes to stay GPU-batchable
        let (sf_rings, sf_sectors) = if total_atoms > 30_000 { (8, 4) } else { (12, 6) };
        let (hoh_rings, hoh_sectors) = (4, 3); // water is always low-poly

        // Spacefill for chains
        let mesh_sf    = meshes.add(Sphere::new(1.5).mesh().uv(sf_rings, sf_sectors));
        let mesh_sf_hoh = meshes.add(Sphere::new(1.0).mesh().uv(hoh_rings, hoh_sectors));

        for chain in &protein.chains {
            if let Some(state) = obj.viz_state.chain_states.get(&chain.id) {
                if state.spacefill {
                    // Pre-create one custom_color material per chain (shared across all atoms)
                    let custom_mat = if state.use_custom_color {
                        Some(materials.add(Color::srgb(
                            state.custom_color[0],
                            state.custom_color[1],
                            state.custom_color[2],
                        )))
                    } else { None };

                    for residue in &chain.residues {
                        let is_hoh = residue.name == "HOH" || residue.name == "WAT";
                        let mesh_to_use = if is_hoh { mesh_sf_hoh.clone() } else { mesh_sf.clone() };

                        for atom in &residue.atoms {
                            // Skip hydrogen atoms in water — they're invisible at macro scale
                            // and each H = 1 extra entity/draw-call
                            if is_hoh {
                                let el = atom.get_element_char();
                                if el == 'H' || el == 'D' { continue; }
                            }

                            let mat = if let Some(ref cm) = custom_mat {
                                cm.clone()
                            } else {
                                let el = atom.get_element_char();
                                if is_hoh { mat_hoh_o.clone() } else { elem_mat!(el) }
                            };
                            commands.spawn((
                                PbrBundle {
                                    mesh: mesh_to_use.clone(),
                                    material: mat,
                                    transform: Transform::from_xyz(
                                        atom.x as f32 - center.x,
                                        atom.y as f32 - center.y,
                                        atom.z as f32 - center.z,
                                    ),
                                    ..default()
                                },
                                AtomMesh { object_index: obj_idx },
                                AtomRef { obj_idx, serial: atom.serial },
                            ));
                        }
                    }
                }
            }
        }

        // Spacefill for ligands
        for (i, ligand) in protein.ligands.iter().enumerate() {
            if let Some(state) = obj.viz_state.ligand_states.get(&i) {
                if state.spacefill {
                    let is_water = crate::core::protein::WATER_NAMES.contains(&ligand.name.as_str());
                    let custom_mat = if state.use_custom_color {
                        Some(materials.add(Color::srgb(
                            state.custom_color[0],
                            state.custom_color[1],
                            state.custom_color[2],
                        )))
                    } else { None };
                    let lig_mesh = if is_water {
                        mesh_sf_hoh.clone()
                    } else {
                        mesh_sf.clone()
                    };
                    for atom in &ligand.atoms {
                        // Skip H in water ligands
                        if is_water {
                            let el = atom.get_element_char();
                            if el == 'H' || el == 'D' { continue; }
                        }
                        let mat = if let Some(ref cm) = custom_mat {
                            cm.clone()
                        } else {
                            let el = atom.get_element_char();
                            if is_water { mat_hoh_o.clone() } else { elem_mat!(el) }
                        };
                        commands.spawn((
                            PbrBundle {
                                mesh: lig_mesh.clone(),
                                material: mat,
                                transform: Transform::from_xyz(
                                    atom.x as f32 - center.x,
                                    atom.y as f32 - center.y,
                                    atom.z as f32 - center.z,
                                ),
                                ..default()
                            },
                            AtomMesh { object_index: obj_idx },
                            AtomRef { obj_idx, serial: atom.serial },
                        ));
                    }
                }
            }
        }

        // Ligand-specific meshes: thinner bonds, small joint spheres
        let mesh_lig_bond  = meshes.add(Cylinder::new(0.07, 1.0));
        let mesh_lig_joint = meshes.add(Sphere::new(0.12).mesh().uv(8, 4));

        // Sticks for ligands
        for (lig_idx, ligand) in protein.ligands.iter().enumerate() {
            if let Some(state) = obj.viz_state.ligand_states.get(&lig_idx) {
                if state.sticks {
                    let is_water_lig = crate::core::protein::WATER_NAMES.contains(&ligand.name.as_str());
                    // Pre-create shared custom mat for this ligand (avoid per-atom alloc)
                    let custom_mat_lig = if state.use_custom_color {
                        Some(materials.add(Color::srgb(
                            state.custom_color[0],
                            state.custom_color[1],
                            state.custom_color[2],
                        )))
                    } else { None };

                    // Collect atom world positions and materials
                    let mut stick_atoms: std::collections::HashMap<i32, (Vec3, char, Handle<StandardMaterial>)>
                        = std::collections::HashMap::new();

                    for atom in &ligand.atoms {
                        let el = atom.get_element_char();
                        // Skip H in water sticks (only O matters for HOH display)
                        if is_water_lig && (el == 'H') { continue; }

                        let pos = Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32) - center;
                        let mat = if let Some(ref cm) = custom_mat_lig {
                            cm.clone()
                        } else {
                            if is_water_lig { mat_hoh_o.clone() } else { elem_mat!(el) }
                        };

                        // Joint sphere at atom position
                        commands.spawn((
                            PbrBundle {
                                mesh: mesh_lig_joint.clone(),
                                material: mat.clone(),
                                transform: Transform::from_translation(pos),
                                ..default()
                            },
                            AtomMesh { object_index: obj_idx },
                            AtomRef { obj_idx, serial: atom.serial },
                        ));

                        stick_atoms.insert(atom.serial, (pos, el, mat));
                    }

                    // Spawn one cylinder per bond (full length, centered at midpoint)
                    // sa/sb: atom serials for BondRef tracking
                    macro_rules! draw_lig_bond {
                        ($pa:expr, $pb:expr, $ma:expr, $mb:expr, $sa:expr, $sb:expr) => {{
                            let diff = $pb - $pa;
                            let dist = diff.length();
                            if dist > 0.01 {
                                let mid = ($pa + $pb) * 0.5;
                                let rot = Quat::from_rotation_arc(Vec3::Y, diff / dist);

                                // First half (pa → mid): material of atom a
                                commands.spawn((
                                    PbrBundle {
                                        mesh: mesh_lig_bond.clone(),
                                        material: $ma,
                                        transform: Transform::from_translation(($pa + mid) * 0.5)
                                            .with_rotation(rot)
                                            .with_scale(Vec3::new(1.0, dist * 0.5, 1.0)),
                                        ..default()
                                    },
                                    AtomMesh { object_index: obj_idx },
                                    BondRef { obj_idx, serial_a: $sa, serial_b: $sb, half: 0 },
                                ));
                                // Second half (mid → pb): material of atom b
                                commands.spawn((
                                    PbrBundle {
                                        mesh: mesh_lig_bond.clone(),
                                        material: $mb,
                                        transform: Transform::from_translation((mid + $pb) * 0.5)
                                            .with_rotation(rot)
                                            .with_scale(Vec3::new(1.0, dist * 0.5, 1.0)),
                                        ..default()
                                    },
                                    AtomMesh { object_index: obj_idx },
                                    BondRef { obj_idx, serial_a: $sa, serial_b: $sb, half: 1 },
                                ));
                            }
                        }};
                    }

                    // Use CONECT bonds from PDB if available
                    let mut has_bonds_in_file = false;
                    for &(sa, sb) in &protein.bonds {
                        if let (Some((pa, _, ma)), Some((pb, _, mb))) =
                            (stick_atoms.get(&sa), stick_atoms.get(&sb))
                        {
                            has_bonds_in_file = true;
                            draw_lig_bond!(*pa, *pb, ma.clone(), mb.clone(), sa, sb);
                        }
                    }

                    // Fallback: distance-based bond detection
                    if !has_bonds_in_file {
                        // 1.9 Å covers all common covalent bonds (C-C 1.54, C=O 1.20, C-N 1.47, C-S 1.82)
                        let threshold_sq = 1.9_f32 * 1.9_f32;
                        let min_sq = 0.25_f32; // 0.5 Å min (avoid self-bonds)
                        let keys: Vec<i32> = stick_atoms.keys().cloned().collect();
                        for i in 0..keys.len() {
                            for j in (i + 1)..keys.len() {
                                let (pa, ea, ma) = &stick_atoms[&keys[i]];
                                let (pb, eb, mb) = &stick_atoms[&keys[j]];
                                let dsq = pa.distance_squared(*pb);

                                let dynamic_threshold_sq = if *ea == 'H' || *eb == 'H' {
                                    1.4_f32 * 1.4_f32
                                } else {
                                    threshold_sq
                                };

                                if dsq > min_sq && dsq < dynamic_threshold_sq {
                                    let sa = keys[i];
                                    let sb = keys[j];
                                    draw_lig_bond!(*pa, *pb, ma.clone(), mb.clone(), sa, sb);
                                }
                            }
                        }
                    }
                }
            }
        }


        // Sticks for chains – residue-aware bond detection
        let mesh_stick_chain = meshes.add(Cylinder::new(0.2, 1.0));
        let mesh_stick_joint = meshes.add(Sphere::new(0.2));
        for chain in &protein.chains {
            if let Some(state) = obj.viz_state.chain_states.get(&chain.id) {
                if state.sticks {
                    // Pre-create shared custom mat for this chain (avoid per-atom alloc)
                    let custom_mat_chain = if state.use_custom_color {
                        Some(materials.add(Color::srgb(
                            state.custom_color[0],
                            state.custom_color[1],
                            state.custom_color[2],
                        )))
                    } else { None };

                    // Collect atoms grouped by residue index so we never connect
                    // atoms from different residues unless it is the peptide bond.
                    let residues = &chain.residues;

                    // Threshold: typical covalent bond ≤ 1.9 Å.
                    // Use a slightly generous 2.0 Å to handle longer bonds (C-S, C-P etc.)
                    // but NEVER 2.1+ which risks cross-residue false positives.
                    let bond_threshold_sq = 2.0_f32 * 2.0;
                    let min_dist_sq = 0.16_f32; // 0.4^2

                    for (res_idx, residue) in residues.iter().enumerate() {
                        let n_atoms = residue.atoms.len();

                        // ── Intra-residue bonds ─────────────────────────────────────────
                        // Build a small spatial grid per residue for O(N) lookup.
                        let cell = 2.5_f32;
                        let mut local_grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
                        for (ai, atom) in residue.atoms.iter().enumerate() {
                            let p = Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32);
                            let key = (
                                (p.x / cell).floor() as i32,
                                (p.y / cell).floor() as i32,
                                (p.z / cell).floor() as i32,
                            );
                            local_grid.entry(key).or_default().push(ai);
                        }

                        for ai in 0..n_atoms {
                            let atom_a = &residue.atoms[ai];
                            let pa = Vec3::new(atom_a.x as f32, atom_a.y as f32, atom_a.z as f32)
                                - center;
                            let el_a = atom_a.get_element_char();

                            let is_hoh = residue.name == "HOH" || residue.name == "WAT";
                            if is_hoh && (el_a == 'H' || el_a == 'D') { continue; }

                            // Spawn joint sphere
                            let mat_joint = if let Some(ref cm) = custom_mat_chain {
                                cm.clone()
                            } else {
                                if is_hoh { mat_hoh_o.clone() } else { elem_mat!(el_a) }
                            };
                            commands.spawn((
                                PbrBundle {
                                    mesh: mesh_stick_joint.clone(),
                                    material: mat_joint,
                                    transform: Transform::from_translation(pa),
                                    ..default()
                                },
                                AtomMesh { object_index: obj_idx },
                                AtomRef { obj_idx, serial: atom_a.serial },
                            ));

                            // Find bonded atoms within same residue
                            let pa_raw = pa + center;
                            let cx = (pa_raw.x / cell).floor() as i32;
                            let cy = (pa_raw.y / cell).floor() as i32;
                            let cz = (pa_raw.z / cell).floor() as i32;
                            for dx in -1..=1 {
                                for dy in -1..=1 {
                                    for dz in -1..=1 {
                                        if let Some(neighbours) =
                                            local_grid.get(&(cx + dx, cy + dy, cz + dz))
                                        {
                                            for &bi in neighbours {
                                                if bi <= ai {
                                                    continue;
                                                }
                                                let atom_b = &residue.atoms[bi];
                                                let pb = Vec3::new(
                                                    atom_b.x as f32,
                                                    atom_b.y as f32,
                                                    atom_b.z as f32,
                                                ) - center;
                                                let dsq = pa.distance_squared(pb);
                                                let el_b = atom_b.get_element_char();
                                                        
                                                let dynamic_threshold_sq = if el_a == 'H' || el_b == 'H' {
                                                    1.4_f32 * 1.4_f32
                                                } else {
                                                    bond_threshold_sq
                                                };

                                                if dsq > min_dist_sq && dsq < dynamic_threshold_sq {
                                                    if is_hoh && (el_b == 'H' || el_b == 'D') { continue; }

                                                    let ma = if let Some(ref cm) = custom_mat_chain {
                                                        cm.clone()
                                                    } else {
                                                        if is_hoh { mat_hoh_o.clone() } else { elem_mat!(el_a) }
                                                    };
                                                    let mb = if let Some(ref cm) = custom_mat_chain {
                                                        cm.clone()
                                                    } else {
                                                        if is_hoh { mat_hoh_o.clone() } else { elem_mat!(el_b) }
                                                    };
                                                    let dist = dsq.sqrt();
                                                    let mid = (pa + pb) * 0.5;
                                                    let dir = pb - pa;
                                                    let rot = Quat::from_rotation_arc(
                                                        Vec3::Y,
                                                        dir.normalize(),
                                                    );
                                                    commands.spawn((
                                                        PbrBundle {
                                                            mesh: mesh_stick_chain.clone(),
                                                            material: ma,
                                                            transform: Transform::from_translation(
                                                                (pa + mid) * 0.5,
                                                            )
                                                            .with_rotation(rot)
                                                            .with_scale(Vec3::new(1.0, dist * 0.5, 1.0)),
                                                            ..default()
                                                        },
                                                        AtomMesh { object_index: obj_idx },
                                                        BondRef { obj_idx, serial_a: atom_a.serial, serial_b: atom_b.serial, half: 0 },
                                                    ));
                                                    commands.spawn((
                                                        PbrBundle {
                                                            mesh: mesh_stick_chain.clone(),
                                                            material: mb,
                                                            transform: Transform::from_translation(
                                                                (mid + pb) * 0.5,
                                                            )
                                                            .with_rotation(rot)
                                                            .with_scale(Vec3::new(1.0, dist * 0.5, 1.0)),
                                                            ..default()
                                                        },
                                                        AtomMesh { object_index: obj_idx },
                                                        BondRef { obj_idx, serial_a: atom_a.serial, serial_b: atom_b.serial, half: 1 },
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // ── Inter-residue peptide bond: C(i) → N(i+1) ──────────────────
                        // Only do this for the adjacent next residue, nothing else.
                        if res_idx + 1 < residues.len() {
                            let next = &residues[res_idx + 1];
                            // Find backbone C in current residue
                            let c_atom = residue.atoms.iter().find(|a| a.name.trim() == "C");
                            // Find backbone N in next residue
                            let n_atom = next.atoms.iter().find(|a| a.name.trim() == "N");
                            if let (Some(c_a), Some(n_a)) = (c_atom, n_atom) {
                                let pc =
                                    Vec3::new(c_a.x as f32, c_a.y as f32, c_a.z as f32) - center;
                                let pn =
                                    Vec3::new(n_a.x as f32, n_a.y as f32, n_a.z as f32) - center;
                                let dsq = pc.distance_squared(pn);
                                // Peptide bond C-N is ~1.33 Å; allow up to 1.5 Å
                                if dsq > 0.25 && dsq < 2.25 {
                                    let dist = dsq.sqrt();
                                    let mid = (pc + pn) * 0.5;
                                    let dir = pn - pc;
                                    let rot = Quat::from_rotation_arc(Vec3::Y, dir.normalize());
                                    let mc = if let Some(ref cm) = custom_mat_chain { cm.clone() } else { mat_c.clone() };
                                    let mn = if let Some(ref cm) = custom_mat_chain { cm.clone() } else { mat_n.clone() };
                                    commands.spawn((
                                        PbrBundle {
                                            mesh: mesh_stick_chain.clone(),
                                            material: mc,
                                            transform: Transform::from_translation(
                                                (pc + mid) * 0.5,
                                            )
                                            .with_rotation(rot)
                                            .with_scale(Vec3::new(1.0, dist * 0.5, 1.0)),
                                            ..default()
                                        },
                                        AtomMesh { object_index: obj_idx },
                                        BondRef { obj_idx, serial_a: c_a.serial, serial_b: n_a.serial, half: 0 },
                                    ));
                                    commands.spawn((
                                        PbrBundle {
                                            mesh: mesh_stick_chain.clone(),
                                            material: mn,
                                            transform: Transform::from_translation(
                                                (mid + pn) * 0.5,
                                            )
                                            .with_rotation(rot)
                                            .with_scale(Vec3::new(1.0, dist * 0.5, 1.0)),
                                            ..default()
                                        },
                                        AtomMesh { object_index: obj_idx },
                                        BondRef { obj_idx, serial_a: c_a.serial, serial_b: n_a.serial, half: 1 },
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Wireframe for chains
        let mesh_wf = meshes.add(Sphere::new(0.2).mesh().uv(8, 4));
        for chain in &protein.chains {
            if let Some(state) = obj.viz_state.chain_states.get(&chain.id) {
                if state.wireframe {
                    let custom_mat_chain = if state.use_custom_color {
                        Some(materials.add(Color::srgb(
                            state.custom_color[0],
                            state.custom_color[1],
                            state.custom_color[2],
                        )))
                    } else { None };

                    for residue in &chain.residues {
                        let is_hoh = residue.name == "HOH" || residue.name == "WAT";
                        for atom in &residue.atoms {
                            let el = atom.get_element_char();
                            if is_hoh && (el == 'H' || el == 'D') { continue; }

                            let mat = if let Some(ref cm) = custom_mat_chain {
                                cm.clone()
                            } else {
                                if is_hoh { mat_hoh_o.clone() } else { elem_mat!(el) }
                            };
                            commands.spawn((
                                PbrBundle {
                                    mesh: mesh_wf.clone(),
                                    material: mat,
                                    transform: Transform::from_xyz(
                                        atom.x as f32 - center.x,
                                        atom.y as f32 - center.y,
                                        atom.z as f32 - center.z,
                                    ),
                                    ..default()
                                },
                                AtomMesh {
                                    object_index: obj_idx,
                                },
                            ));
                        }
                    }
                }
            }
        }

        // Wireframe for ligands
        for (i, ligand) in protein.ligands.iter().enumerate() {
            if let Some(state) = obj.viz_state.ligand_states.get(&i) {
                if state.wireframe {
                    let custom_mat_lig = if state.use_custom_color {
                        Some(materials.add(Color::srgb(
                            state.custom_color[0],
                            state.custom_color[1],
                            state.custom_color[2],
                        )))
                    } else { None };
                    let is_water_lig = crate::core::protein::WATER_NAMES.contains(&ligand.name.as_str());

                    for atom in &ligand.atoms {
                        let el = atom.get_element_char();
                        if is_water_lig && (el == 'H' || el == 'D') { continue; }

                        let mat = if let Some(ref cm) = custom_mat_lig {
                            cm.clone()
                        } else {
                            if is_water_lig { mat_hoh_o.clone() } else { elem_mat!(el) }
                        };
                        commands.spawn((
                            PbrBundle {
                                mesh: mesh_wf.clone(),
                                material: mat,
                                transform: Transform::from_xyz(
                                    atom.x as f32 - center.x,
                                    atom.y as f32 - center.y,
                                    atom.z as f32 - center.z,
                                ),
                                ..default()
                            },
                            AtomMesh {
                                object_index: obj_idx,
                            },
                        ));
                    }
                }
            }
        }

        if let Some(ribbon_mesh) = ribbon::generate_bevy_mesh(protein, &obj.viz_state.chain_states)
        {
            let ribbon_material = materials.add(StandardMaterial {
                base_color: Color::WHITE,
                double_sided: true,
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(ribbon_mesh),
                    material: ribbon_material,
                    transform: Transform::from_xyz(-center.x, -center.y, -center.z),
                    ..default()
                },
                AtomMesh {
                    object_index: obj_idx,
                },
                RibbonRef {
                    obj_idx,
                },
            ));
        }

        if let Some(surface_mesh) = surface::generate_surface_mesh(protein, &obj.viz_state) {
            let surface_material = materials.add(StandardMaterial {
                base_color: Color::WHITE,
                alpha_mode: AlphaMode::Opaque,
                cull_mode: Some(bevy::render::render_resource::Face::Back),
                ..default()
            });

            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(surface_mesh),
                    material: surface_material,
                    transform: Transform::from_xyz(-center.x, -center.y, -center.z),
                    ..default()
                },
                AtomMesh {
                    object_index: obj_idx,
                },
                SurfaceRef {
                    obj_idx,
                },
            ));
        }
    }
}

use crate::core::interactions::{detect_interactions, InteractionType};

// ─────────────────────────────────────────────────────────────────────────────
// Atom picking — ray-sphere intersection for distance measurement
// ─────────────────────────────────────────────────────────────────────────────

/// Fired when the user Ctrl+right-clicks on the viewport.
/// The system reads mouse position, builds a ray, finds the nearest atom.
pub fn atom_pick_system(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    object_manager: Res<ObjectManager>,
    mut measure_state: ResMut<MeasureState>,
    mut egui_contexts: bevy_egui::EguiContexts,
) {
    // Only trigger on Ctrl + Right-click (pressed this frame)
    if !mouse_buttons.just_pressed(MouseButton::Right) {
        return;
    }
    let ctrl = keyboard.pressed(KeyCode::ControlLeft)
        || keyboard.pressed(KeyCode::ControlRight)
        || keyboard.pressed(KeyCode::SuperLeft)  // macOS Cmd
        || keyboard.pressed(KeyCode::SuperRight);
    if !ctrl {
        return;
    }

    // Don't pick when hovering over egui panels
    if egui_contexts.ctx_mut().is_pointer_over_area() {
        return;
    }

    let Ok(window) = windows.get_single() else { return };
    let Some(cursor_pos) = window.cursor_position() else { return };

    // Use the first camera found
    let Some((camera, cam_transform)) = cameras.iter().next() else { return };
    let Some(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };

    // Compute scene center (same logic as spawn_protein_meshes)
    let scene_center = compute_scene_center(&object_manager);

    // Radius thresholds per display mode
    let pick_radius_large = 1.5_f32; // spacefill
    let pick_radius_small = 0.4_f32; // wireframe/sticks

    let mut best_t = f32::MAX;
    let mut best_pick: Option<AtomPickTarget> = None;

    for obj in &object_manager.objects {
        if !obj.visible { continue; }

        let protein = &obj.protein;

        // Check chain atoms
        for chain in &protein.chains {
            let viz = obj.viz_state.chain_states.get(&chain.id);
            let radius = if viz.map(|v| v.spacefill).unwrap_or(false) {
                pick_radius_large
            } else {
                pick_radius_small
            };

            for residue in &chain.residues {
                for atom in &residue.atoms {
                    let world_pos = Vec3::new(
                        atom.x as f32 - scene_center.x,
                        atom.y as f32 - scene_center.y,
                        atom.z as f32 - scene_center.z,
                    );
                    if let Some(t) = ray_sphere_intersect(ray.origin, *ray.direction, world_pos, radius) {
                        if t < best_t {
                            best_t = t;
                            best_pick = Some(AtomPickTarget {
                                world_pos,
                                label: format!(
                                    "{} {}{} ({})",
                                    atom.name.trim(),
                                    residue.name,
                                    residue.seq_num,
                                    chain.id
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Check ligand atoms
        for (lig_idx, ligand) in protein.ligands.iter().enumerate() {
            let viz = obj.viz_state.ligand_states.get(&lig_idx);
            let radius = if viz.map(|v| v.spacefill).unwrap_or(true) {
                pick_radius_large
            } else {
                pick_radius_small
            };

            for atom in &ligand.atoms {
                let world_pos = Vec3::new(
                    atom.x as f32 - scene_center.x,
                    atom.y as f32 - scene_center.y,
                    atom.z as f32 - scene_center.z,
                );
                if let Some(t) = ray_sphere_intersect(ray.origin, *ray.direction, world_pos, radius) {
                    if t < best_t {
                        best_t = t;
                        best_pick = Some(AtomPickTarget {
                            world_pos,
                            label: format!(
                                "{} {} (ligand {})",
                                atom.name.trim(),
                                ligand.name,
                                lig_idx
                            ),
                        });
                    }
                }
            }
        }
    }

    if let Some(pick) = best_pick {
        measure_state.push(pick);
    }
}

/// Ray-sphere intersection. Returns t (distance along ray) or None.
fn ray_sphere_intersect(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    radius: f32,
) -> Option<f32> {
    let oc = origin - center;
    let b = oc.dot(direction);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let t = -b - discriminant.sqrt();
    if t > 0.001 {
        Some(t)
    } else {
        let t2 = -b + discriminant.sqrt();
        if t2 > 0.001 { Some(t2) } else { None }
    }
}

/// Compute the centroid of all protein atoms (same as spawn_protein_meshes).
fn compute_scene_center(object_manager: &ObjectManager) -> Vec3 {
    if let Some(first_obj) = object_manager.objects.first() {
        let mut center = Vec3::ZERO;
        let mut count = 0usize;
        for chain in &first_obj.protein.chains {
            for residue in &chain.residues {
                for atom in &residue.atoms {
                    center += Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32);
                    count += 1;
                }
            }
        }
        if count > 0 { center / count as f32 } else { Vec3::ZERO }
    } else {
        Vec3::ZERO
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurement gizmo — draws line + highlight spheres for picked atoms
// ─────────────────────────────────────────────────────────────────────────────

pub fn draw_measure_gizmos(
    mut gizmos: Gizmos,
    measure_state: Res<MeasureState>,
) {
    let highlight_color   = Color::srgb(1.0, 0.85, 0.1);   // warm yellow
    let line_color        = Color::srgb(1.0, 0.85, 0.1);
    let second_color      = Color::srgb(0.3, 1.0, 0.5);    // green for second pick
    let sphere_radius     = 0.6_f32;

    // Draw first atom highlight
    if let Some(ref a) = measure_state.first {
        gizmos.sphere(a.world_pos, Quat::IDENTITY, sphere_radius, highlight_color);
        // Pulsing cross (3 axis lines)
        let arm = 1.2_f32;
        gizmos.line(a.world_pos - Vec3::X * arm, a.world_pos + Vec3::X * arm, highlight_color);
        gizmos.line(a.world_pos - Vec3::Y * arm, a.world_pos + Vec3::Y * arm, highlight_color);
        gizmos.line(a.world_pos - Vec3::Z * arm, a.world_pos + Vec3::Z * arm, highlight_color);
    }

    // Draw second atom highlight
    if let Some(ref b) = measure_state.second {
        gizmos.sphere(b.world_pos, Quat::IDENTITY, sphere_radius, second_color);
        let arm = 1.2_f32;
        gizmos.line(b.world_pos - Vec3::X * arm, b.world_pos + Vec3::X * arm, second_color);
        gizmos.line(b.world_pos - Vec3::Y * arm, b.world_pos + Vec3::Y * arm, second_color);
        gizmos.line(b.world_pos - Vec3::Z * arm, b.world_pos + Vec3::Z * arm, second_color);
    }

    // Draw dashed line + tick marks between the two atoms
    if let (Some(ref a), Some(ref b)) = (&measure_state.first, &measure_state.second) {
        let p1 = a.world_pos;
        let p2 = b.world_pos;
        let dir = (p2 - p1).normalize();
        let dist = p1.distance(p2);

        // Dashed line
        let dash = 0.4_f32;
        let gap  = 0.25_f32;
        let mut t = 0.0_f32;
        while t < dist {
            let start = p1 + dir * t;
            let end   = p1 + dir * (t + dash).min(dist);
            gizmos.line(start, end, line_color);
            t += dash + gap;
        }

        // Mid-point tick perpendicular to the line
        let mid = (p1 + p2) * 0.5;
        let perp = if dir.abs().dot(Vec3::Y) < 0.9 {
            dir.cross(Vec3::Y).normalize()
        } else {
            dir.cross(Vec3::X).normalize()
        };
        let tick = 0.6_f32;
        gizmos.line(mid - perp * tick, mid + perp * tick, line_color);

        // End caps
        let cap = 0.5_f32;
        gizmos.line(p1 - perp * cap, p1 + perp * cap, highlight_color);
        gizmos.line(p2 - perp * cap, p2 + perp * cap, second_color);
    }
}

fn draw_interactions_system(mut gizmos: Gizmos, object_manager: Res<ObjectManager>) {
    for obj in &object_manager.objects {
        if obj.visible && obj.viz_state.show_interactions {
            let interactions = detect_interactions(
                &obj.protein,
                obj.viz_state.hbond_dist_threshold,
                obj.viz_state.hydrophobic_dist_threshold,
            );
            println!(
                "Interactions found for {}: {}",
                obj.name,
                interactions.len()
            );

            // Wait, even better, if interactions is empty, we just skip.
            // Calculate center (same logic as spawn)
            let mut center = Vec3::ZERO;
            let mut count = 0;
            for chain in &obj.protein.chains {
                for residue in &chain.residues {
                    for atom in &residue.atoms {
                        center += Vec3::new(atom.x as f32, atom.y as f32, atom.z as f32);
                        count += 1;
                    }
                }
            }
            if count > 0 {
                center /= count as f32;
            }

            for interaction in interactions {
                let color = match interaction.i_type {
                    InteractionType::HydrogenBond => Color::srgb(1.0, 1.0, 0.0), // Yellow
                    InteractionType::SaltBridge => Color::srgb(0.0, 1.0, 1.0),   // Cyan
                    InteractionType::Hydrophobic => Color::srgb(1.0, 0.5, 0.0),  // Orange
                    InteractionType::PiPiStacking => Color::srgb(1.0, 0.0, 1.0), // Magenta
                };

                // Draw a dashed line simulation
                let p1 = interaction.p1 - center;
                let p2 = interaction.p2 - center;

                let dir = (p2 - p1).normalize();
                let dist = p1.distance(p2);
                let dash_len = 0.2;
                let mut current_d = 0.0;

                while current_d < dist {
                    let start = p1 + dir * current_d;
                    let end = p1 + dir * f32::min(current_d + dash_len, dist);
                    gizmos.line(start, end, color);
                    current_d += dash_len * 2.0;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Molecular Dynamics animation systems
// ─────────────────────────────────────────────────────────────────────────────

/// Advance the MD frame index when playback is active.
fn tick_md_playback(
    time: Res<Time>,
    mut md_state: ResMut<MdState>,
) {
    if !md_state.is_playing {
        return;
    }
    let Some(ref traj) = md_state.trajectory else {
        return;
    };
    let n_frames = traj.frame_count();
    if n_frames == 0 {
        md_state.is_playing = false;
        return;
    }

    md_state.frame_timer += time.delta_seconds();
    let frame_duration = 1.0 / md_state.playback_speed.max(0.1);

    while md_state.frame_timer >= frame_duration {
        md_state.frame_timer -= frame_duration;
        md_state.current_frame += 1;
        if md_state.current_frame >= n_frames {
            if md_state.loop_playback {
                md_state.current_frame = 0;
            } else {
                md_state.current_frame = n_frames - 1;
                md_state.is_playing = false;
                break;
            }
        }
    }
}

/// Apply the current MD frame's atom positions to the linked ProteinObject.
///
/// Uses `atom.serial - 1` as the XTC position index so that atoms are
/// correctly mapped even when the topology PDB is a mixed membrane+protein
/// system (e.g. GROMACS output where lipids come before the peptide).
fn apply_md_frame(
    md_state: Res<MdState>,
    mut object_manager: ResMut<ObjectManager>,
) {
    if !md_state.is_changed() {
        return;
    }

    let Some(ref traj) = md_state.trajectory else {
        return;
    };
    let Some(obj_idx) = md_state.linked_object_idx else {
        return;
    };
    let Some(obj) = object_manager.objects.get_mut(obj_idx) else {
        return;
    };
    let frame_idx = md_state.current_frame.min(traj.frame_count().saturating_sub(1));
    let Some(frame) = traj.frames.get(frame_idx) else {
        return;
    };

    let mut applied = 0;
    let mut skipped = 0;

    // The PDB parser separates ATOMs (into chains) and HETATMs (into ligands).
    // Furthermore, PDB serial numbers often have gaps (e.g. missing atoms),
    // but the XTC trajectory contains exactly `N` densely packed coordinates.
    // To perfectly map the XTC coordinate index back to the right atom, we
    // collect all atoms and sort them by their original PDB serial number.
    let mut atom_refs = Vec::new();
    for chain in &mut obj.protein.chains {
        for residue in &mut chain.residues {
            for atom in &mut residue.atoms {
                atom_refs.push(atom);
            }
        }
    }
    for ligand in &mut obj.protein.ligands {
        for atom in &mut ligand.atoms {
            atom_refs.push(atom);
        }
    }
    
    // The PDB parser preserves atom order from the file (pdbtbx iterates chains/residues/atoms
    // in the order they appear in the file). GROMACS XTC files use the same atom ordering as
    // the topology PDB. So no sorting is needed — we enumerate in file order = XTC order.
    //
    // NOTE: For small normal PDBs (<99999 atoms) where serials are plain decimal,
    // the file order and serial order are identical, so this is still correct.
    // For large hybrid-36 PDBs we sanitize serials to sequential 1-based, also preserving order.

    for (xtc_idx, atom) in atom_refs.into_iter().enumerate() {
        if let Some(p) = frame.positions.get(xtc_idx) {
            atom.x = p[0] as f64;
            atom.y = p[1] as f64;
            atom.z = p[2] as f64;
            applied += 1;
        } else {
            skipped += 1;
        }
    }

    let _ = (applied, skipped); // suppress unused warnings

    // Use transform_dirty instead of obj.changed to avoid full mesh rebuild every frame.
    // sync_md_transforms will update all entity Transforms in-place.
    obj.transform_dirty = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// MD transform sync — updates atom/bond entity Transforms in-place each frame
// This avoids the catastrophic full mesh despawn+respawn that caused lag.
// ─────────────────────────────────────────────────────────────────────────────

fn sync_md_transforms(
    mut object_manager: ResMut<ObjectManager>,
    mut atom_q: Query<(&AtomRef, &mut Transform), Without<BondRef>>,
    mut bond_q: Query<(&BondRef, &mut Transform), Without<AtomRef>>,
    mut ribbon_q: Query<(&RibbonRef, &mut Handle<Mesh>), Without<SurfaceRef>>,
    mut surface_q: Query<(&SurfaceRef, &mut Handle<Mesh>), Without<RibbonRef>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // Fast early exit — nothing to sync if no object is dirty
    let any_dirty = object_manager.objects.iter().any(|o| o.transform_dirty);
    if !any_dirty {
        return;
    }

    // Build per-object position caches: serial → centered world pos
    // We collect into a Vec of Option<HashMap> indexed by obj_idx.
    let n_objs = object_manager.objects.len();
    let mut caches: Vec<Option<HashMap<i32, Vec3>>> = (0..n_objs).map(|_| None).collect();

    for (obj_idx, obj) in object_manager.objects.iter().enumerate() {
        if !obj.transform_dirty {
            continue;
        }
        let center = obj.center;
        let mut cache: HashMap<i32, Vec3> = HashMap::with_capacity(64_000);

        for chain in &obj.protein.chains {
            for residue in &chain.residues {
                for atom in &residue.atoms {
                    cache.insert(
                        atom.serial,
                        Vec3::new(
                            atom.x as f32 - center.x,
                            atom.y as f32 - center.y,
                            atom.z as f32 - center.z,
                        ),
                    );
                }
            }
        }
        for ligand in &obj.protein.ligands {
            for atom in &ligand.atoms {
                cache.insert(
                    atom.serial,
                    Vec3::new(
                        atom.x as f32 - center.x,
                        atom.y as f32 - center.y,
                        atom.z as f32 - center.z,
                    ),
                );
            }
        }
        caches[obj_idx] = Some(cache);

        // Regenerate Ribbon mesh if dirty
        for (ribbon_ref, mut handle) in ribbon_q.iter_mut() {
            if ribbon_ref.obj_idx == obj_idx {
                if let Some(new_mesh) = ribbon::generate_bevy_mesh(&obj.protein, &obj.viz_state.chain_states) {
                    *handle = meshes.add(new_mesh);
                }
            }
        }

        // Regenerate Surface mesh if dirty
        for (surface_ref, mut handle) in surface_q.iter_mut() {
            if surface_ref.obj_idx == obj_idx {
                if let Some(new_mesh) = surface::generate_surface_mesh(&obj.protein, &obj.viz_state) {
                    *handle = meshes.add(new_mesh);
                }
            }
        }
    }

    // Update atom sphere positions (translation only)
    for (atom_ref, mut tf) in &mut atom_q {
        let oi = atom_ref.obj_idx;
        if oi >= n_objs {
            continue;
        }
        if let Some(ref cache) = caches[oi] {
            if let Some(&pos) = cache.get(&atom_ref.serial) {
                tf.translation = pos;
            }
        }
    }

    // Update bond cylinder positions+rotation+scale
    for (bond_ref, mut tf) in &mut bond_q {
        let oi = bond_ref.obj_idx;
        if oi >= n_objs {
            continue;
        }
        if let Some(ref cache) = caches[oi] {
            if let (Some(&pa), Some(&pb)) =
                (cache.get(&bond_ref.serial_a), cache.get(&bond_ref.serial_b))
            {
                let diff = pb - pa;
                let dist = diff.length();

                // PBC artifact suppression: real covalent bonds are < 2.5 Å.
                // If dist > 3.0 Å, the two atoms have wrapped around the periodic box
                // and the cylinder would span the entire simulation box. Hide it.
                if dist > 3.0 {
                    tf.scale = Vec3::ZERO;
                    continue;
                }

                if dist > 0.01 {
                    let mid = (pa + pb) * 0.5;
                    let rot = Quat::from_rotation_arc(Vec3::Y, diff / dist);
                    let half_center = if bond_ref.half == 0 {
                        (pa + mid) * 0.5
                    } else {
                        (mid + pb) * 0.5
                    };
                    tf.translation = half_center;
                    tf.rotation = rot;
                    tf.scale = Vec3::new(1.0, dist * 0.5, 1.0);
                }
            }
        }
    }

    // Clear dirty flags now that all transforms are updated
    for obj in object_manager.objects.iter_mut() {
        obj.transform_dirty = false;
    }
}

fn update_camera_enabled(
    mut contexts: bevy_egui::EguiContexts,
    mut q_camera: Query<&mut PanOrbitCamera>,
) {
    let wants_pointer = contexts.ctx_mut().wants_pointer_input();
    let wants_keyboard = contexts.ctx_mut().wants_keyboard_input();
    let is_focused = wants_pointer || wants_keyboard;
    
    for mut cam in q_camera.iter_mut() {
        if cam.enabled == is_focused {
            cam.enabled = !is_focused;
        }
    }
}

fn update_headlight(
    camera_q: Query<&Transform, With<Camera>>,
    mut light_q: Query<&mut Transform, (With<DirectionalLight>, Without<Camera>)>,
) {
    if let Ok(cam_transform) = camera_q.get_single() {
        for mut light_transform in light_q.iter_mut() {
            *light_transform = *cam_transform;
        }
    }
}
