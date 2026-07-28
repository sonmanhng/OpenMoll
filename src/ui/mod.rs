use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use futures_lite::future;
use rfd::AsyncFileDialog;
use std::path::PathBuf;
use crate::graphics::MdState;

/// Holds the Bevy image handle for the app logo, loaded at startup.
#[derive(Resource)]
pub struct LogoTexture(pub Handle<Image>);

/// Holds the egui TextureId resolved from the logo handle (set each frame).
#[derive(Resource, Default)]
pub struct LogoEguiId(pub Option<egui::TextureId>);

use crate::core::alignment::{
    align_structures_sequence_aware, apply_alignment, default_protein_chain_id,
    find_chain, StructureAlignmentMode,
};
use crate::core::discovery::{analyze_binding_sites, residue_druggability_note};
use crate::core::protein::{LigandType, MoleculeType, SecondaryStructure};
use crate::core::seq_align::{
    pairwise_stats, progressive_msa, similarity_matrix, needleman_wunsch, smith_waterman,
    PairwiseAlgorithm, SequenceRecord,
};
use crate::core::sequence::{analyze_sequences, chain_sequence, format_fasta};
use crate::graphics::{ChainVizState, LigandVizState, MeasureState, ObjectManager};
use crate::io::LoadFileEvent;
use crate::io::export::TransformTarget;

pub mod interaction_map;

pub struct UiPlugin;

#[derive(Component)]
struct FileDialogTask(Task<Option<PathBuf>>);

/// Async task: pick a JSON interaction file, read its content
#[derive(Component)]
struct JsonImportTask(Task<Option<(String, String)>>); // (filename, content)

/// Async task: pick an XTC trajectory file, parse it in background thread.
/// Returns (path, trajectory_result).
#[derive(Component)]
struct XtcImportTask(Task<Option<(PathBuf, Result<crate::io::xtc_parser::XtcTrajectory, String>)>>);

/// Async task: pick a PDB file as MD topology, parse it, strip non-protein chains.
/// Returns (path, filtered_protein)
#[derive(Component)]
struct MdTopologyTask(Task<Option<(PathBuf, crate::core::protein::Protein)>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectorTab {
    Structure,
    Sequence,
    Display,
    Interactions,
    Surface,
    Alignment,
    Discovery,
    MolecularDynamics,
    Transform,
}

impl Default for InspectorTab {
    fn default() -> Self {
        Self::Structure
    }
}

/// Which item is being dragged in the 2D interaction map
#[derive(Debug, Clone)]
pub enum MapDragItem {
    Atom(usize),          // ligand atom index
    Residue(String),      // residue key "NAME SEQ (CHAIN)"
}

#[derive(Resource)]
pub struct ConsoleState {
    pub align_target: usize,
    pub align_mobile: usize,
    pub align_mode: StructureAlignmentMode,
    pub align_target_chain: String,
    pub align_mobile_chain: String,
    pub pairwise_algo: PairwiseAlgorithm,
    pub msa_cache: Option<crate::core::seq_align::MultipleSequenceAlignment>,
    pub similarity_cache: Option<crate::core::seq_align::SimilarityMatrix>,
    pub input: String,
    pub logs: Vec<String>,
    selected_object: Option<usize>,
    active_tab: InspectorTab,
    /// 2D interaction map viewport state
    pub map_zoom: f32,
    pub map_pan: egui::Vec2,
    /// Overridden positions for dragged ligand atoms (index → layout Vec2)
    pub map_atom_overrides: std::collections::HashMap<usize, egui::Vec2>,
    /// Overridden positions for dragged residue nodes (label → layout Vec2)
    pub map_residue_overrides: std::collections::HashMap<String, egui::Vec2>,
    /// Currently dragged item: Some(("atom", idx)) or Some(("res", 0)) with label stored separately
    pub map_drag_item: Option<MapDragItem>,
    /// External interaction data imported from JSON file
    pub external_interactions: Option<crate::io::interaction_import::ExternalInteractionData>,
    pub transform_target: Option<TransformTarget>,
    pub transform_xyz: [f32; 3],
}

impl Default for ConsoleState {
    fn default() -> Self {
        Self {
            align_target: 0,
            align_mobile: 0,
            align_mode: StructureAlignmentMode::SequenceGlobal,
            align_target_chain: "A".into(),
            align_mobile_chain: "A".into(),
            pairwise_algo: PairwiseAlgorithm::NeedlemanWunsch,
            msa_cache: None,
            similarity_cache: None,
            input: String::new(),
            logs: vec!["Ready. Open a PDB/mmCIF structure to begin.".into()],
            selected_object: None,
            active_tab: InspectorTab::Structure,
            map_zoom: 1.0,
            map_pan: egui::Vec2::ZERO,
            map_atom_overrides: std::collections::HashMap::new(),
            map_residue_overrides: std::collections::HashMap::new(),
            map_drag_item: None,
            external_interactions: None,
            transform_target: None,
            transform_xyz: [0.0, 0.0, 0.0],
        }
    }
}

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            .init_resource::<ConsoleState>()
            .init_resource::<LogoEguiId>()
            .add_systems(Startup, setup_logo)
            .add_systems(Update, render_ui)
            .add_systems(Update, poll_file_dialog_task)
            .add_systems(Update, poll_json_import_task)
            .add_systems(Update, poll_xtc_import_task)
            .add_systems(Update, poll_md_topology_task);
    }
}

fn setup_logo(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // Load logo.png from the project root (relative to executable or cargo run dir)
    let logo_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logo.png");
    if let Ok(bytes) = std::fs::read(&logo_path) {
        if let Ok(dyn_img) = image::load_from_memory(&bytes) {
            let rgba = dyn_img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let bevy_img = Image::new(
                bevy::render::render_resource::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                bevy::render::render_resource::TextureDimension::D2,
                rgba.into_raw(),
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
            );
            let handle = images.add(bevy_img);
            commands.insert_resource(LogoTexture(handle));
        }
    }
}

fn render_ui(
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut object_manager: ResMut<ObjectManager>,
    mut console_state: ResMut<ConsoleState>,
    mut measure_state: ResMut<MeasureState>,
    mut md_state: ResMut<MdState>,
    logo_handle: Option<Res<LogoTexture>>,
    mut logo_id: ResMut<LogoEguiId>,
) {
    // Resolve logo texture handle → egui TextureId each frame
    if let Some(ref handle) = logo_handle {
        logo_id.0 = Some(contexts.add_image(handle.0.clone_weak()));
    }
    let logo_texture_id = logo_id.0;

    let ctx = contexts.ctx_mut();
    apply_dark_theme(ctx);

    if object_manager.objects.is_empty() {
        console_state.selected_object = None;
    } else {
        let selected = console_state.selected_object.unwrap_or(0);
        console_state.selected_object = Some(selected.min(object_manager.objects.len() - 1));
    }

    let mut do_open = false;
    let mut do_align = false;
    let mut do_json_import = false;
    let mut do_md_import_structure = false;
    let mut do_md_import_trajectory = false;
    let mut delete_object: Option<usize> = None;

    render_top_bar(
        ctx,
        &mut object_manager,
        &mut console_state,
        &mut do_open,
        &mut do_align,
        logo_texture_id,
    );

    render_left_panel(ctx, &object_manager, &mut console_state, &mut do_open);

    render_right_panel(
        ctx,
        &mut object_manager,
        &mut console_state,
        &mut md_state,
        &mut do_align,
        &mut delete_object,
        &mut do_md_import_structure,
        &mut do_md_import_trajectory,
    );

    render_console(ctx, &mut console_state);
    render_measure_overlay(ctx, &mut measure_state);
    render_central_view(ctx, &object_manager, &mut console_state, &mut do_json_import);

    if do_open {
        spawn_file_dialog(&mut commands);
    }
    if do_align {
        run_alignment(&mut object_manager, &mut console_state);
    }
    if do_json_import {
        spawn_json_import_dialog(&mut commands);
    }
    if do_md_import_structure {
        spawn_md_topology_dialog(&mut commands);
    }
    if do_md_import_trajectory {
        spawn_xtc_import_dialog(&mut commands);
    }

    if let Some(idx) = delete_object {
        if idx < object_manager.objects.len() {
            let name = object_manager.objects[idx].name.clone();
            object_manager.objects.remove(idx);
            for obj in &mut object_manager.objects {
                obj.changed = true;
            }
            console_state.logs.push(format!("Removed structure: {}", name));
            console_state.selected_object = if object_manager.objects.is_empty() {
                None
            } else {
                Some(idx.min(object_manager.objects.len() - 1))
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Color palette  (modern light theme — easyticket style)
// ─────────────────────────────────────────────────────────────────────────────


/// Soft off-white background  #f5f6f8
const APP_BG: egui::Color32 = egui::Color32::from_rgb(245, 246, 248);
/// Pure white card surface
const CARD: egui::Color32 = egui::Color32::from_rgb(255, 255, 255);
/// Subtle hover tint
const CARD_HOVER: egui::Color32 = egui::Color32::from_rgb(248, 250, 252);
/// Sidebar / panel background (same as app bg)
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(250, 251, 252);
/// Top-bar background — same as CARD
#[allow(dead_code)]
const TOPBAR_BG: egui::Color32 = CARD;
/// Primary blue accent  #4a5fd5
const ACCENT: egui::Color32 = egui::Color32::from_rgb(74, 95, 213);
/// Deeper blue for hover
#[allow(dead_code)]
const ACCENT_DARK: egui::Color32 = egui::Color32::from_rgb(58, 78, 188);
/// Light blue tag background
const TAG_BLUE_BG: egui::Color32 = egui::Color32::from_rgb(235, 240, 255);
/// Bright orange accent  #ff5722
const ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 87, 34);
/// Light orange tag background
const TAG_ORANGE_BG: egui::Color32 = egui::Color32::from_rgb(255, 243, 235);
/// Success green
const GREEN: egui::Color32 = egui::Color32::from_rgb(76, 175, 80);
const TAG_GREEN_BG: egui::Color32 = egui::Color32::from_rgb(232, 245, 233);
/// Primary dark text
const TEXT: egui::Color32 = egui::Color32::from_rgb(30, 32, 38);
/// Secondary / muted text
const MUTED: egui::Color32 = egui::Color32::from_rgb(108, 117, 125);
/// Subtle border
const BORDER: egui::Color32 = egui::Color32::from_rgb(228, 231, 236);
/// Hairline separator
const SEPARATOR: egui::Color32 = egui::Color32::from_rgb(238, 240, 244);

const ROUND_SM: f32 = 8.0;
const ROUND_MD: f32 = 12.0;
#[allow(dead_code)]
const ROUND_LG: f32 = 16.0;

// ─────────────────────────────────────────────────────────────────────────────
// Theme application
// ─────────────────────────────────────────────────────────────────────────────

fn apply_dark_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();

    visuals.override_text_color = Some(TEXT);
    visuals.panel_fill = APP_BG;
    visuals.window_fill = CARD;
    visuals.faint_bg_color = PANEL_BG;
    visuals.extreme_bg_color = egui::Color32::from_rgb(235, 237, 240);

    // Selection
    visuals.selection.bg_fill = TAG_BLUE_BG;
    visuals.selection.stroke = egui::Stroke::new(1.5_f32, ACCENT);
    visuals.hyperlink_color = ACCENT;

    // Widgets
    visuals.widgets.noninteractive.bg_fill = CARD;
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32, TEXT);
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.widgets.noninteractive.rounding = egui::Rounding::same(ROUND_SM);

    visuals.widgets.inactive.bg_fill = CARD;
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0_f32, TEXT);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.widgets.inactive.rounding = egui::Rounding::same(ROUND_SM);

    visuals.widgets.hovered.bg_fill = CARD_HOVER;
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0_f32, ACCENT);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.5_f32, ACCENT);
    visuals.widgets.hovered.rounding = egui::Rounding::same(ROUND_SM);

    visuals.widgets.active.bg_fill = TAG_BLUE_BG;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.5_f32, ACCENT);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(2.0_f32, ACCENT);
    visuals.widgets.active.rounding = egui::Rounding::same(ROUND_SM);

    visuals.widgets.open.bg_fill = CARD_HOVER;
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT);
    visuals.widgets.open.rounding = egui::Rounding::same(ROUND_SM);

    // Shadows for depth
    visuals.window_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 2.0),
        blur: 12.0,
        spread: 0.0,
        color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 15),
    };

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(12.0, 10.0);
    style.spacing.button_padding = egui::vec2(18.0, 10.0);
    style.spacing.slider_width = 180.0;
    style.spacing.window_margin = egui::Margin::same(16.0);
    style.spacing.indent = 20.0;
    style.spacing.scroll.bar_width = 5.0;
    ctx.set_style(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reusable frame helpers
// ─────────────────────────────────────────────────────────────────────────────

fn card_shadow() -> egui::epaint::Shadow {
    egui::epaint::Shadow {
        offset: egui::vec2(0.0, 2.0),
        blur: 8.0,
        spread: 0.0,
        color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 8),
    }
}

fn card_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(CARD)
        .rounding(egui::Rounding::same(ROUND_MD))
        .shadow(card_shadow())
        .inner_margin(egui::Margin::same(14.0))
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
}

fn card_frame_selected() -> egui::Frame {
    egui::Frame::none()
        .fill(TAG_BLUE_BG)
        .rounding(egui::Rounding::same(ROUND_MD))
        .shadow(card_shadow())
        .inner_margin(egui::Margin::same(14.0))
        .stroke(egui::Stroke::new(2.0_f32, ACCENT))
}

fn panel_frame(fill: egui::Color32) -> egui::Frame {
    egui::Frame::none()
        .fill(fill)
        .inner_margin(egui::Margin::same(12.0))
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
}

fn inset_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(APP_BG)
        .rounding(egui::Rounding::same(ROUND_SM))
        .inner_margin(egui::Margin::same(10.0))
        .stroke(egui::Stroke::new(1.0_f32, SEPARATOR))
}

// ─────────────────────────────────────────────────────────────────────────────
// Button helpers
// ─────────────────────────────────────────────────────────────────────────────

fn primary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(egui::Color32::WHITE).strong().size(14.0))
            .fill(ACCENT)
            .stroke(egui::Stroke::NONE)
            .rounding(egui::Rounding::same(ROUND_SM))
            .min_size(egui::vec2(0.0, 38.0)),
    )
}

fn secondary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(TEXT).size(14.0))
            .fill(CARD)
            .stroke(egui::Stroke::new(1.5_f32, BORDER))
            .rounding(egui::Rounding::same(ROUND_SM))
            .min_size(egui::vec2(0.0, 38.0)),
    )
}

fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(egui::Color32::WHITE).strong().size(14.0))
            .fill(ACCENT)
            .stroke(egui::Stroke::NONE)
            .rounding(egui::Rounding::same(ROUND_SM))
            .min_size(egui::vec2(0.0, 38.0)),
    )
}

fn ghost_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(MUTED).size(13.0))
            .fill(egui::Color32::TRANSPARENT)
            .stroke(egui::Stroke::new(1.0_f32, SEPARATOR))
            .rounding(egui::Rounding::same(ROUND_SM))
            .min_size(egui::vec2(0.0, 32.0)),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tag / badge helpers
// ─────────────────────────────────────────────────────────────────────────────

fn tag_pill(ui: &mut egui::Ui, label: &str, fg: egui::Color32, bg: egui::Color32) {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(11.5),
        fg,
    );
    let size = galley.size() + egui::vec2(16.0, 8.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, egui::Rounding::same(16.0), bg);
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
}

fn dot_badge(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

// ─────────────────────────────────────────────────────────────────────────────
// Nav tab helper
// ─────────────────────────────────────────────────────────────────────────────

fn nav_tab(ui: &mut egui::Ui, active: bool, label: &str) -> egui::Response {
    let text = if active {
        egui::RichText::new(label).strong().color(TEXT).size(14.0)
    } else {
        egui::RichText::new(label).color(MUTED).size(14.0)
    };
    let btn = egui::Button::new(text)
        .fill(egui::Color32::TRANSPARENT)
        .stroke(egui::Stroke::NONE)
        .frame(false);
    let response = ui.add(btn);
    if active {
        let rect = response.rect;
        let y = rect.bottom() + 3.0;
        ui.painter().hline(rect.x_range(), y, egui::Stroke::new(2.5_f32, ACCENT));
    }
    response
}

fn inspector_tab_button(
    ui: &mut egui::Ui,
    console_state: &mut ConsoleState,
    tab: InspectorTab,
    label: &str,
) {
    let active = console_state.active_tab == tab;
    if nav_tab(ui, active, label).clicked() {
        console_state.active_tab = tab;
    }
    ui.add_space(2.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Top bar
// ─────────────────────────────────────────────────────────────────────────────

fn render_top_bar(
    ctx: &egui::Context,
    object_manager: &mut ObjectManager,
    console_state: &mut ConsoleState,
    do_open: &mut bool,
    _do_align: &mut bool,
    logo_texture_id: Option<egui::TextureId>,
) {
    egui::TopBottomPanel::top("top_bar")
        .exact_height(58.0)
        .frame(
            egui::Frame::none()
                .fill(CARD)
                .inner_margin(egui::Margin::symmetric(22.0, 0.0))
                .stroke(egui::Stroke::new(1.0_f32, SEPARATOR)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                // Logo — use PNG if loaded, otherwise fallback circle
                if let Some(tex_id) = logo_texture_id {
                    let logo_size = egui::vec2(36.0, 36.0);
                    let (logo_rect, _) =
                        ui.allocate_exact_size(logo_size, egui::Sense::hover());
                    // Clip to rounded square
                    let rounding = egui::Rounding::same(8.0);
                    ui.painter().image(
                        tex_id,
                        logo_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    // Subtle rounded border
                    ui.painter().rect_stroke(
                        logo_rect,
                        rounding,
                        egui::Stroke::new(1.0_f32, BORDER),
                    );
                } else {
                    // Fallback: accent circle with "M"
                    let (logo_rect, _) =
                        ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::hover());
                    ui.painter().circle_filled(logo_rect.center(), 15.0, ACCENT);
                    ui.painter().text(
                        logo_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "M",
                        egui::FontId::proportional(16.0),
                        egui::Color32::WHITE,
                    );
                }

                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("OpenMoll")
                        .size(18.0)
                        .strong()
                        .color(TEXT),
                );

                // Right-side controls
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if primary_button(ui, "Open structure").clicked() {
                        *do_open = true;
                    }
                    ui.add_space(8.0);
                    // Structure count badge
                    let count = object_manager.objects.len();
                    if count > 0 {
                        tag_pill(
                            ui,
                            &format!("{} structure{}", count, if count == 1 { "" } else { "s" }),
                            ACCENT,
                            TAG_BLUE_BG,
                        );
                    }
                    // Keep selected_object in sync even without the combo
                    if !object_manager.objects.is_empty()
                        && console_state.selected_object.is_none()
                    {
                        console_state.selected_object = Some(0);
                    }
                });
            });
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Left panel
// ─────────────────────────────────────────────────────────────────────────────

fn render_left_panel(
    ctx: &egui::Context,
    object_manager: &ObjectManager,
    console_state: &mut ConsoleState,
    _do_open: &mut bool,
) {
    egui::SidePanel::left("project_panel")
        .resizable(true)
        .default_width(272.0)
        .width_range(220.0..=360.0)
        .frame(panel_frame(PANEL_BG))
        .show(ctx, |ui| {
            ui.add_space(8.0);
            ui.label(label_sm("STRUCTURES"));
            ui.add_space(6.0);

            egui::ScrollArea::vertical()
                .id_source("object_list")
                .show(ui, |ui| {
                    if object_manager.objects.is_empty() {
                        inset_frame().show(ui, |ui| {
                            ui.label(muted_sm("No structure loaded."));
                            ui.label(muted_sm("Use 'Open structure' in the toolbar."));
                        });
                    }

                    for (idx, obj) in object_manager.objects.iter().enumerate() {
                        let selected = console_state.selected_object == Some(idx);
                        let frame = if selected { card_frame_selected() } else { card_frame() };
                        frame.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                dot_badge(ui, if selected { ACCENT } else { MUTED });
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new(&obj.name)
                                        .strong()
                                        .size(14.0)
                                        .color(if selected { ACCENT } else { TEXT }),
                                );
                            });
                            ui.add_space(3.0);
                            ui.label(muted_sm(&format!(
                                "{} chains  ·  {} ligands  ·  {} atoms",
                                obj.protein.chains.len(),
                                obj.protein.ligands.len(),
                                obj.protein.atom_count()
                            )));
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                tag_pill(
                                    ui,
                                    &format!("{} chains", obj.protein.chains.len()),
                                    ACCENT,
                                    TAG_BLUE_BG,
                                );
                                ui.add_space(4.0);
                                if !obj.protein.ligands.is_empty() {
                                    tag_pill(
                                        ui,
                                        &format!("{} ligands", obj.protein.ligands.len()),
                                        ORANGE,
                                        TAG_ORANGE_BG,
                                    );
                                }
                            });
                            let response = ui.interact(
                                ui.min_rect(),
                                ui.id().with(idx),
                                egui::Sense::click(),
                            );
                            if response.clicked() {
                                console_state.selected_object = Some(idx);
                            }
                            if response.hovered() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                        });
                        ui.add_space(6.0);
                    }
                });
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Right panel (Inspector)
// ─────────────────────────────────────────────────────────────────────────────

fn render_right_panel(
    ctx: &egui::Context,
    object_manager: &mut ObjectManager,
    console_state: &mut ConsoleState,
    md_state: &mut MdState,
    do_align: &mut bool,
    delete_object: &mut Option<usize>,
    do_md_import_structure: &mut bool,
    do_md_import_trajectory: &mut bool,
) {
    egui::SidePanel::right("inspector_panel")
        .resizable(true)
        .default_width(420.0)
        .width_range(360.0..=560.0)
        .frame(panel_frame(PANEL_BG))
        .show(ctx, |ui| {
            ui.add_space(4.0);

            // ── Inspector header with tabs ───────────────────────────────────
            card_frame().show(ui, |ui| {
                section_title(ui, "Inspector");

                // MD tab is always available (doesn't need a loaded structure)
                let is_md_tab = console_state.active_tab == InspectorTab::MolecularDynamics;

                if object_manager.objects.is_empty() && !is_md_tab {
                    ui.add_space(4.0);
                    ui.label(muted_sm("Import a structure to enable analysis controls."));
                    // Still show MD tab button even when no structures
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        inspector_tab_button(ui, console_state, InspectorTab::MolecularDynamics, "⚛ MD");
                    });
                    return;
                }

                if !is_md_tab {
                    let Some(idx) = console_state.selected_object else {
                        ui.add_space(4.0);
                        ui.label(muted_sm("Select a structure from the left panel."));
                        return;
                    };
                    if idx >= object_manager.objects.len() {
                        return;
                    }
                }

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    inspector_tab_button(ui, console_state, InspectorTab::Structure, "Structure");
                    inspector_tab_button(ui, console_state, InspectorTab::Sequence, "Sequence");
                    inspector_tab_button(ui, console_state, InspectorTab::Display, "Display");
                    inspector_tab_button(ui, console_state, InspectorTab::Interactions, "Interactions");
                    inspector_tab_button(ui, console_state, InspectorTab::Surface, "Surface");
                    inspector_tab_button(ui, console_state, InspectorTab::Alignment, "Alignment");
                    inspector_tab_button(ui, console_state, InspectorTab::Discovery, "Discovery");
                    inspector_tab_button(ui, console_state, InspectorTab::Transform, "Transform");
                    inspector_tab_button(ui, console_state, InspectorTab::MolecularDynamics, "⚛ MD");
                });
            });

            ui.add_space(8.0);

            // ── Tab content ──────────────────────────────────────────────────
            egui::ScrollArea::vertical()
                .id_source("inspector_scroll")
                .show(ui, |ui| {
                    // Molecular Dynamics tab — works standalone
                    if console_state.active_tab == InspectorTab::MolecularDynamics {
                        card_frame().show(ui, |ui| {
                            render_molecular_dynamics_tab(
                                ui, md_state, object_manager,
                                do_md_import_structure, do_md_import_trajectory,
                            );
                        });
                        return;
                    }

                    if object_manager.objects.is_empty() {
                        return;
                    }
                    let Some(idx) = console_state.selected_object else {
                        return;
                    };
                    if idx >= object_manager.objects.len() {
                        return;
                    }

                    if console_state.active_tab == InspectorTab::Alignment {
                        card_frame().show(ui, |ui| {
                            render_alignment_tab(ui, object_manager, console_state, do_align);
                        });
                        return;
                    }

                    let obj = &mut object_manager.objects[idx];
                    card_frame().show(ui, |ui| {
                        match console_state.active_tab {
                            InspectorTab::Structure => {
                                render_structure_tab(ui, obj, idx, console_state, delete_object)
                            }
                            InspectorTab::Sequence => render_sequence_tab(ui, obj, console_state),
                            InspectorTab::Display => render_display_tab(ui, obj),
                            InspectorTab::Interactions => render_interactions_tab(ui, obj),
                            InspectorTab::Surface => render_surface_tab(ui, obj),
                            InspectorTab::Discovery => render_discovery_tab(ui, obj),
                            InspectorTab::Transform => render_transform_tab(ui, obj, console_state),
                            InspectorTab::Alignment | InspectorTab::MolecularDynamics => unreachable!(),
                        }
                    });
                });
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Structure
// ─────────────────────────────────────────────────────────────────────────────


fn render_structure_tab(
    ui: &mut egui::Ui,
    obj: &mut crate::graphics::ProteinObject,
    idx: usize,
    _console_state: &mut ConsoleState,
    delete_object: &mut Option<usize>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&obj.name).size(17.0).strong().color(TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if secondary_button(ui, "Remove").clicked() {
                *delete_object = Some(idx);
            }
        });
    });

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("OVERVIEW"));
    ui.add_space(4.0);

    let stats = structure_stats(obj);
    metric_grid(
        ui,
        &[
            ("Chains", stats.chains.to_string()),
            ("Residues", stats.residues.to_string()),
            ("Atoms", stats.atoms.to_string()),
            ("Ligands", stats.ligands.to_string()),
            ("Ions", stats.ions.to_string()),
            ("Bonds", obj.protein.bonds.len().to_string()),
            ("Radius", format!("{:.1} Å", obj.protein.bounding_radius())),
        ],
    );

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("CHAINS"));

    for chain in &obj.protein.chains {
        let ss = secondary_counts(chain);
        ui.add_space(6.0);
        inset_frame().show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(strong(&format!("Chain {}", chain.id)));
                ui.add_space(4.0);
                tag_pill(
                    ui,
                    molecule_type_label(chain.molecule_type),
                    ACCENT,
                    TAG_BLUE_BG,
                );
            });
            ui.add_space(2.0);
            ui.label(muted_sm(&format!(
                "{} residues  ·  helix {}  ·  sheet {}  ·  coil {}",
                chain.residues.len(),
                ss.0,
                ss.1,
                ss.2
            )));
        });
    }

    if !obj.protein.ligands.is_empty() {
        ui.add_space(10.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(label_sm("LIGANDS & IONS"));
        for (lig_idx, ligand) in obj.protein.ligands.iter().enumerate() {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(strong(&format!(
                    "{} · {}:{}",
                    ligand.name, ligand.chain_id, ligand.seq_num
                )));
                ui.add_space(4.0);
                tag_pill(
                    ui,
                    ligand_type_label(ligand.ligand_type),
                    if ligand.ligand_type == LigandType::Ion { MUTED } else { ORANGE },
                    if ligand.ligand_type == LigandType::Ion { PANEL_BG } else { TAG_ORANGE_BG },
                );
                ui.label(muted_sm(&format!("{} atoms", ligand.atoms.len())));
            });
            let _ = lig_idx;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Sequence
// ─────────────────────────────────────────────────────────────────────────────

fn render_sequence_tab(
    ui: &mut egui::Ui,
    obj: &crate::graphics::ProteinObject,
    console_state: &mut ConsoleState,
) {
    ui.label(strong("Sequence analysis"));
    ui.add_space(2.0);
    ui.label(muted_sm(
        "Per-chain sequence extraction from loaded residues.",
    ));

    let analyses = analyze_sequences(&obj.protein);
    if analyses.is_empty() {
        ui.add_space(6.0);
        ui.label(muted_sm("No polymer chain sequence was found."));
        return;
    }

    for analysis in analyses {
        ui.add_space(8.0);
        divider(ui);
        ui.add_space(6.0);
        inset_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(strong(&format!("Chain {}", analysis.chain_id)));
                ui.add_space(4.0);
                ui.label(muted_sm(molecule_type_label(analysis.molecule_type)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ghost_button(ui, "Log FASTA").clicked() {
                        console_state
                            .logs
                            .push(format_fasta(&analysis, &obj.name).trim_end().into());
                    }
                });
            });
            ui.add_space(6.0);
            metric_grid(
                ui,
                &[
                    ("Residues", analysis.residue_count.to_string()),
                    ("Atoms", analysis.atom_count.to_string()),
                    ("MW estimate", format!("{:.1} kDa", analysis.estimated_mw_da / 1000.0)),
                    ("Hydrophobic", format!("{:.0}%", analysis.hydrophobic_fraction * 100.0)),
                    ("Polar", format!("{:.0}%", analysis.polar_fraction * 100.0)),
                    ("Charged", format!("{:.0}%", analysis.charged_fraction * 100.0)),
                    ("Aromatic", format!("{:.0}%", analysis.aromatic_fraction * 100.0)),
                    ("Helix", format!("{:.0}%", analysis.helix_fraction * 100.0)),
                    ("Sheet", format!("{:.0}%", analysis.sheet_fraction * 100.0)),
                    ("Coil", format!("{:.0}%", analysis.coil_fraction * 100.0)),
                ],
            );
            ui.add_space(6.0);
            ui.label(label_sm("SEQUENCE"));
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(sequence_preview(&analysis.sequence, 180))
                    .color(ACCENT)
                    .size(11.0)
                    .monospace(),
            );

            ui.add_space(6.0);
            ui.label(label_sm("COMPOSITION"));
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                for (code, count) in &analysis.composition {
                    ui.label(
                        egui::RichText::new(format!("{}:{}", code, count))
                            .color(MUTED)
                            .size(11.0),
                    );
                }
            });

            ui.add_space(6.0);
            ui.label(label_sm("MOTIF CANDIDATES"));
            ui.add_space(2.0);
            if analysis.motifs.is_empty() {
                ui.label(muted_sm("No basic motif candidates detected."));
            } else {
                for motif in analysis.motifs.iter().take(16) {
                    ui.label(egui::RichText::new(format!(
                        "{} | {}-{} | {}",
                        motif.name, motif.start, motif.end, motif.sequence
                    )).color(TEXT).size(12.0));
                }
                if analysis.motifs.len() > 16 {
                    ui.label(muted_sm(&format!(
                        "{} additional motif candidates hidden",
                        analysis.motifs.len() - 16
                    )));
                }
            }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Display
// ─────────────────────────────────────────────────────────────────────────────

fn render_display_tab(ui: &mut egui::Ui, obj: &mut crate::graphics::ProteinObject) {
    ui.label(strong("Display presets"));
    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        if accent_button(ui, "Cartoon").clicked() {
            for state in obj.viz_state.chain_states.values_mut() {
                *state = ChainVizState::default();
            }
            for state in obj.viz_state.ligand_states.values_mut() {
                state.spacefill = false;
                state.sticks = true;
                state.wireframe = false;
                state.surface = false;
            }
            obj.changed = true;
        }
        if secondary_button(ui, "Binding site").clicked() {
            for state in obj.viz_state.chain_states.values_mut() {
                state.ribbon = true;
                state.sticks = false;
                state.spacefill = false;
                state.surface = true;
                state.eps_surface = false;
            }
            for state in obj.viz_state.ligand_states.values_mut() {
                state.spacefill = false;
                state.sticks = true;
                state.surface = false;
            }
            obj.viz_state.show_interactions = true;
            obj.changed = true;
        }
        if secondary_button(ui, "Atom detail").clicked() {
            for state in obj.viz_state.chain_states.values_mut() {
                state.ribbon = false;
                state.sticks = true;
                state.spacefill = false;
                state.surface = false;
                state.eps_surface = false;
            }
            for state in obj.viz_state.ligand_states.values_mut() {
                state.spacefill = true;
                state.sticks = true;
            }
            obj.changed = true;
        }
    });

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("PROTEIN CHAINS"));

    let mut keys: Vec<_> = obj.viz_state.chain_states.keys().cloned().collect();
    keys.sort();
    for key in keys {
        ui.add_space(6.0);
        inset_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(strong(&format!("Chain {}", key)));
                if let Some(state) = obj.viz_state.chain_states.get_mut(&key) {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.checkbox(&mut state.use_custom_color, "Custom color").changed() {
                            obj.changed = true;
                        }
                        if state.use_custom_color
                            && ui.color_edit_button_rgb(&mut state.custom_color).changed()
                        {
                            obj.changed = true;
                        }
                    });
                }
            });
            ui.add_space(4.0);
            if let Some(state) = obj.viz_state.chain_states.get_mut(&key) {
                display_toggles_chain(ui, state, &mut obj.changed);
            }
        });
    }

    if !obj.viz_state.ligand_states.is_empty() {
        ui.add_space(10.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(label_sm("LIGANDS"));
        
        let lig_keys: Vec<_> = obj.viz_state.ligand_states.keys().cloned().collect();
        let mut groups: std::collections::HashMap<&str, Vec<usize>> = std::collections::HashMap::new();
        
        for key in &lig_keys {
            let name = obj
                .protein
                .ligands
                .get(*key)
                .map(|l| l.name.as_str())
                .unwrap_or("Ligand");
            groups.entry(name).or_default().push(*key);
        }
        
        let mut sorted_group_names: Vec<_> = groups.keys().cloned().collect();
        sorted_group_names.sort();

        for name in sorted_group_names {
            let keys = groups.get(name).unwrap();
            let count = keys.len();
            let first_key = keys[0];
            
            let mut master_state = obj.viz_state.ligand_states.get(&first_key).cloned().unwrap_or_default();
            let old_state = master_state.clone();

            ui.add_space(6.0);
            inset_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    if count > 1 {
                        ui.label(strong(&format!("{} ({} molecules)", name, count)));
                    } else {
                        ui.label(strong(&format!("{} {}", first_key, name)));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.checkbox(&mut master_state.use_custom_color, "Custom color").changed() {
                            obj.changed = true;
                        }
                        if master_state.use_custom_color
                            && ui.color_edit_button_rgb(&mut master_state.custom_color).changed()
                        {
                            obj.changed = true;
                        }
                    });
                });
                ui.add_space(4.0);
                display_toggles_ligand(ui, &mut master_state, &mut obj.changed);
            });
            
            if master_state != old_state {
                for key in keys {
                    if let Some(state) = obj.viz_state.ligand_states.get_mut(key) {
                        *state = master_state.clone();
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Interactions
// ─────────────────────────────────────────────────────────────────────────────

fn render_interactions_tab(ui: &mut egui::Ui, obj: &mut crate::graphics::ProteinObject) {
    ui.label(strong("Interaction detection"));
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.checkbox(&mut obj.viz_state.show_interactions, "3D lines").changed() {
            obj.changed = true;
        }
        if ui.checkbox(&mut obj.viz_state.show_2d_map, "2D ligand map").changed() {
            obj.changed = true;
        }
    });

    if obj.viz_state.show_2d_map && !obj.protein.ligands.is_empty() {
        ui.add_space(4.0);
        let mut current_idx = obj.viz_state.selected_ligand_2d.unwrap_or(0);
        let current_name = obj
            .protein
            .ligands
            .get(current_idx)
            .map(|l| l.name.clone())
            .unwrap_or_else(|| "Ligand 0".into());

        egui::ComboBox::from_id_source(format!("ligand_map_{}", obj.name))
            .selected_text(current_name)
            .show_ui(ui, |ui| {
                for (idx, ligand) in obj.protein.ligands.iter().enumerate() {
                    let label = format!(
                        "{} {}:{} [{} atoms]",
                        ligand.name, ligand.chain_id, ligand.seq_num, ligand.atoms.len()
                    );
                    if ui.selectable_value(&mut current_idx, idx, label).changed() {
                        obj.viz_state.selected_ligand_2d = Some(idx);
                        obj.changed = true;
                    }
                }
            });
        if obj.viz_state.selected_ligand_2d.is_none() {
            obj.viz_state.selected_ligand_2d = Some(0);
            obj.changed = true;
        }
    }

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("DETECTION THRESHOLDS"));
    ui.add_space(4.0);
    let hbond = ui.add(
        egui::Slider::new(&mut obj.viz_state.hbond_dist_threshold, 2.0..=5.0)
            .text("H-bond Å"),
    );
    let hydro = ui.add(
        egui::Slider::new(&mut obj.viz_state.hydrophobic_dist_threshold, 3.0..=6.0)
            .text("Hydrophobic Å"),
    );
    if hbond.drag_stopped() || hydro.drag_stopped() {
        obj.changed = true;
    }

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("SUMMARY"));
    ui.add_space(4.0);

    let mut hbond_count = 0usize;
    let mut salt_count = 0usize;
    let mut hydro_count = 0usize;
    let mut pi_count = 0usize;
    for inter in &obj.interactions {
        match inter.i_type {
            crate::core::interactions::InteractionType::HydrogenBond => hbond_count += 1,
            crate::core::interactions::InteractionType::SaltBridge => salt_count += 1,
            crate::core::interactions::InteractionType::Hydrophobic => hydro_count += 1,
            crate::core::interactions::InteractionType::PiPiStacking => pi_count += 1,
        }
    }
    metric_grid(
        ui,
        &[
            ("H-bonds", hbond_count.to_string()),
            ("Salt bridges", salt_count.to_string()),
            ("Hydrophobic", hydro_count.to_string()),
            ("Pi-stacking", pi_count.to_string()),
            ("Total", obj.interactions.len().to_string()),
        ],
    );

    if !obj.interactions.is_empty() {
        ui.add_space(8.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(label_sm("NEAREST CONTACTS"));
        ui.add_space(4.0);
        let mut interactions: Vec<_> = obj.interactions.iter().collect();
        interactions.sort_by(|a, b| {
            a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal)
        });
        for inter in interactions.into_iter().take(32) {
            ui.label(egui::RichText::new(format!(
                "{:.2} Å  {}  {}{} {}  {} → {}",
                inter.dist,
                interaction_label(&inter.i_type),
                inter.res_name,
                inter.res_seq,
                inter.chain_id,
                inter.atom1_name,
                inter.atom2_name
            )).color(TEXT).size(12.0));
        }
    } else {
        ui.label(muted_sm(
            "Enable interaction detection, then adjust thresholds if no contacts appear.",
        ));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Surface
// ─────────────────────────────────────────────────────────────────────────────

fn render_surface_tab(ui: &mut egui::Ui, obj: &mut crate::graphics::ProteinObject) {
    ui.label(strong("Surface generation"));
    ui.add_space(6.0);
    let res = ui.add(
        egui::Slider::new(&mut obj.viz_state.surface_resolution, 0.1..=2.0).text("Resolution Å"),
    );
    let iso = ui.add(
        egui::Slider::new(&mut obj.viz_state.surface_iso_level, 0.1..=1.0).text("Iso level"),
    );
    if res.drag_stopped() || iso.drag_stopped() {
        obj.changed = true;
    }

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("PER-CHAIN SURFACES"));
    ui.add_space(4.0);

    let mut keys: Vec<_> = obj.viz_state.chain_states.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if let Some(state) = obj.viz_state.chain_states.get_mut(&key) {
            ui.horizontal(|ui| {
                ui.label(muted_sm(&format!("Chain {}", key)));
                ui.add_space(8.0);
                if ui.checkbox(&mut state.surface, "Molecular").changed() {
                    obj.changed = true;
                }
                if ui.checkbox(&mut state.eps_surface, "Electrostatic").changed() {
                    obj.changed = true;
                }
            });
        }
    }

    ui.add_space(8.0);
    ui.label(muted_sm(
        "Electrostatic coloring uses a fast residue-charge approximation: Asp/Glu negative, Arg/Lys positive, His partial positive.",
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Alignment
// ─────────────────────────────────────────────────────────────────────────────

fn render_alignment_tab(
    ui: &mut egui::Ui,
    object_manager: &ObjectManager,
    console_state: &mut ConsoleState,
    do_align: &mut bool,
) {
    ui.label(strong("Structure superposition (Kabsch)"));
    ui.add_space(6.0);

    if object_manager.objects.len() < 2 {
        ui.label(muted_sm("Load at least two structures to run superposition."));
    } else {
        object_selector(ui, "Target", &mut console_state.align_target, object_manager);
        ui.add_space(4.0);
        object_selector(ui, "Mobile", &mut console_state.align_mobile, object_manager);
        ui.add_space(6.0);

        chain_id_field(
            ui,
            "Target chain",
            &mut console_state.align_target_chain,
            object_manager,
            console_state.align_target,
        );
        ui.add_space(4.0);
        chain_id_field(
            ui,
            "Mobile chain",
            &mut console_state.align_mobile_chain,
            object_manager,
            console_state.align_mobile,
        );
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label(muted_sm("Mode"));
            ui.radio_value(
                &mut console_state.align_mode,
                StructureAlignmentMode::SequenceGlobal,
                "NW + Kabsch",
            );
            ui.radio_value(
                &mut console_state.align_mode,
                StructureAlignmentMode::SequenceLocal,
                "SW + Kabsch",
            );
            ui.radio_value(
                &mut console_state.align_mode,
                StructureAlignmentMode::OrderBased,
                "Order Cα",
            );
        });
        ui.add_space(8.0);
        if accent_button(ui, "Run superposition").clicked() {
            *do_align = true;
        }
        ui.add_space(4.0);
        ui.label(muted_sm(
            "Sequence-aware modes align one-letter sequences first, then superpose matched Cα pairs.",
        ));
    }

    ui.add_space(12.0);
    divider(ui);
    ui.add_space(8.0);
    ui.label(strong("Pairwise sequence alignment"));
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(muted_sm("Algorithm"));
        ui.radio_value(
            &mut console_state.pairwise_algo,
            PairwiseAlgorithm::NeedlemanWunsch,
            "Needleman–Wunsch",
        );
        ui.radio_value(
            &mut console_state.pairwise_algo,
            PairwiseAlgorithm::SmithWaterman,
            "Smith–Waterman",
        );
    });

    if object_manager.objects.len() >= 2 {
        let t_idx = console_state.align_target.min(object_manager.objects.len() - 1);
        let m_idx = console_state.align_mobile.min(object_manager.objects.len() - 1);
        if let (Some(t_chain), Some(m_chain)) = (
            find_chain(
                &object_manager.objects[t_idx].protein,
                &console_state.align_target_chain,
            ),
            find_chain(
                &object_manager.objects[m_idx].protein,
                &console_state.align_mobile_chain,
            ),
        ) {
            let seq_t = chain_sequence(t_chain);
            let seq_m = chain_sequence(m_chain);
            let aln = match console_state.pairwise_algo {
                PairwiseAlgorithm::NeedlemanWunsch => needleman_wunsch(&seq_t, &seq_m),
                PairwiseAlgorithm::SmithWaterman => smith_waterman(&seq_t, &seq_m),
            };
            let stats = pairwise_stats(&aln);
            ui.add_space(6.0);
            metric_grid(
                ui,
                &[
                    ("Score", aln.score.to_string()),
                    ("Identity", format!("{:.1}%", stats.identity * 100.0)),
                    ("Similarity", format!("{:.1}%", stats.similarity * 100.0)),
                    ("Aligned cols", stats.aligned_length.to_string()),
                    ("Matches", stats.matches.to_string()),
                    ("Gaps", stats.gaps.to_string()),
                ],
            );
            ui.add_space(4.0);
            ui.label(label_sm("ALIGNMENT PREVIEW"));
            ui.label(
                egui::RichText::new(format_alignment_preview(&aln.aligned_a, 72))
                    .monospace()
                    .size(11.0)
                    .color(TEXT),
            );
            ui.label(
                egui::RichText::new(format_alignment_preview(&aln.aligned_b, 72))
                    .monospace()
                    .size(11.0)
                    .color(MUTED),
            );
        }
    }

    ui.add_space(12.0);
    divider(ui);
    ui.add_space(8.0);
    ui.label(strong("Multiple alignment & matrix"));
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.button("Compute similarity matrix").clicked() {
            console_state.similarity_cache = Some(build_similarity_from_objects(object_manager));
            console_state
                .logs
                .push("Computed pairwise identity/similarity matrix.".into());
        }
        if ui.button("Run progressive MSA").clicked() {
            let records = sequence_records_from_objects(object_manager);
            if records.len() >= 2 {
                console_state.msa_cache = Some(progressive_msa(&records));
                console_state
                    .logs
                    .push(format!("MSA complete ({} sequences).", records.len()));
            } else {
                console_state
                    .logs
                    .push("Need at least two polymer chains across loaded structures.".into());
            }
        }
    });

    if let Some(mat) = &console_state.similarity_cache {
        ui.add_space(8.0);
        ui.label(label_sm("IDENTITY % (lower triangle) / SIMILARITY % (upper)"));
        similarity_matrix_ui(ui, mat);
    }

    if let Some(msa) = &console_state.msa_cache {
        ui.add_space(8.0);
        ui.label(label_sm("MSA (progressive, BLOSUM62)"));
        for (label, row) in msa.labels.iter().zip(&msa.rows) {
            ui.label(
                egui::RichText::new(format!("{label}  {}", format_alignment_preview(row, 64)))
                    .monospace()
                    .size(11.0)
                    .color(TEXT),
            );
        }
        ui.label(
            egui::RichText::new(format!(
                "consensus  {}",
                format_alignment_preview(&msa.consensus, 64)
            ))
            .monospace()
            .size(11.0)
            .color(ACCENT),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Discovery
// ─────────────────────────────────────────────────────────────────────────────

fn render_discovery_tab(ui: &mut egui::Ui, obj: &mut crate::graphics::ProteinObject) {
    ui.label(strong("Drug-discovery workspace"));
    ui.add_space(4.0);
    ui.label(muted_sm(
        "Pocket analysis computed from residues within 5.0 Å of each ligand.",
    ));

    let has_ligands = !obj.protein.ligands.is_empty();
    if !has_ligands {
        ui.add_space(8.0);
        inset_frame().show(ui, |ui| {
            ui.label(muted_sm("No ligand or ion was parsed from this structure."));
        });
        return;
    }

    let sites = analyze_binding_sites(&obj.protein, &obj.interactions, 5.0);
    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("SITE SUMMARY"));
    ui.add_space(4.0);
    metric_grid(
        ui,
        &[
            ("Ligand sites", sites.len().to_string()),
            ("Pocket residues", sites.iter().map(|s| s.residue_count()).sum::<usize>().to_string()),
            ("Fingerprints", sites.iter().map(|s| s.fingerprint.total()).sum::<usize>().to_string()),
        ],
    );

    for site in sites.iter().take(8) {
        ui.add_space(8.0);
        inset_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(strong(&format!("{} | ligand #{}", site.ligand_label(), site.ligand_index)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(muted_sm(&format!("{} atoms", site.ligand_atom_count)));
                });
            });
            ui.add_space(2.0);
            ui.label(muted_sm(residue_druggability_note(site)));
            ui.add_space(6.0);
            metric_grid(
                ui,
                &[
                    ("Residues", site.residue_count().to_string()),
                    ("Atom contacts", site.contact_count().to_string()),
                    ("Pocket radius", format!("{:.1} Å", site.radius)),
                    ("Hydrophobic", site.hydrophobic_residues.to_string()),
                    ("Polar", site.polar_residues.to_string()),
                    ("Charged", site.charged_residues.to_string()),
                    ("H-bonds", site.fingerprint.hydrogen_bonds.to_string()),
                    ("Salt bridges", site.fingerprint.salt_bridges.to_string()),
                    ("Pi-stacking", site.fingerprint.pi_stacking.to_string()),
                ],
            );
            ui.add_space(4.0);
            ui.label(label_sm("NEAREST RESIDUES"));
            ui.add_space(2.0);
            for residue in site.residues.iter().take(8) {
                ui.label(egui::RichText::new(format!(
                    "{}{} {}  {:.2} Å  {} contacts",
                    residue.res_name, residue.res_seq, residue.chain_id,
                    residue.min_distance, residue.atom_contacts,
                )).color(TEXT).size(12.0));
            }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Molecular Dynamics
// ─────────────────────────────────────────────────────────────────────────────

fn render_molecular_dynamics_tab(
    ui: &mut egui::Ui,
    md_state: &mut MdState,
    object_manager: &ObjectManager,
    do_md_import_structure: &mut bool,
    do_md_import_trajectory: &mut bool,
) {
    // ── Header ──────────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label(strong("Molecular Dynamics"));
        ui.add_space(8.0);
        if md_state.trajectory.is_some() {
            tag_pill(ui, "Loaded", GREEN, TAG_GREEN_BG);
        } else if md_state.is_loading {
            tag_pill(ui, "Loading…", ORANGE, TAG_ORANGE_BG);
        } else {
            tag_pill(ui, "No trajectory", MUTED, PANEL_BG);
        }
    });
    ui.add_space(2.0);
    ui.label(muted_sm(
        "Import a topology (.pdb) and trajectory (.xtc) to visualize MD simulation.",
    ));

    ui.add_space(10.0);
    divider(ui);
    ui.add_space(6.0);
    ui.label(label_sm("DATA FILES"));
    ui.add_space(6.0);

    // ── Import buttons ───────────────────────────────────────────────────────
    inset_frame().show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.horizontal(|ui| {
            if primary_button(ui, "Import Structure").clicked() {
                *do_md_import_structure = true;
            }
            ui.add_space(4.0);
            if let Some(ref p) = md_state.topology_path {
                let fname = p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("topology.pdb");
                ui.label(egui::RichText::new(fname).color(TEXT).size(12.5));
                ui.add_space(4.0);
                tag_pill(ui, "PDB", ACCENT, TAG_BLUE_BG);
            } else {
                ui.label(muted_sm("No topology loaded"));
            }
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if secondary_button(ui, "Import Trajectory").clicked() {
                *do_md_import_trajectory = true;
            }
            ui.add_space(4.0);
            if let Some(ref p) = md_state.trajectory_path {
                let fname = p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("trajectory.xtc");
                ui.label(egui::RichText::new(fname).color(TEXT).size(12.5));
                ui.add_space(4.0);
                tag_pill(ui, "XTC", ORANGE, TAG_ORANGE_BG);
            } else {
                ui.label(muted_sm("No trajectory loaded"));
            }
        });
    });

    // ── Status message ───────────────────────────────────────────────────────
    if !md_state.status_msg.is_empty() {
        ui.add_space(6.0);
        ui.label(muted_sm(&md_state.status_msg));
    }

    // ── Trajectory info ──────────────────────────────────────────────────────
    if let Some(ref traj) = md_state.trajectory {
        ui.add_space(10.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(label_sm("TRAJECTORY INFO"));
        ui.add_space(4.0);

        let (t_start, t_end) = traj.time_range_ps().unwrap_or((0.0, 0.0));
        metric_grid(
            ui,
            &[
                ("Frames", traj.frame_count().to_string()),
                ("Atoms", traj.n_atoms.to_string()),
                ("Time start", format!("{:.1} ps", t_start)),
                ("Time end", format!("{:.1} ps", t_end)),
                ("Duration", format!("{:.1} ps", t_end - t_start)),
                ("Frame Δt", if traj.frame_count() > 1 {
                    format!("{:.2} ps", (t_end - t_start) / (traj.frame_count() - 1) as f32)
                } else {
                    "—".into()
                }),
            ],
        );

        ui.add_space(10.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(label_sm("LINK TO STRUCTURE"));
        ui.add_space(4.0);

        // Structure selector — link trajectory to a loaded ProteinObject
        let current_obj_name = md_state.linked_object_idx
            .and_then(|i| object_manager.objects.get(i))
            .map(|o| o.name.clone())
            .unwrap_or_else(|| "— None —".into());

        egui::ComboBox::from_id_source("md_linked_object")
            .selected_text(&current_obj_name)
            .show_ui(ui, |ui| {
                // "— None —" option
                ui.selectable_value(
                    &mut md_state.linked_object_idx,
                    None,
                    "— None —",
                );
                // One entry per loaded structure
                for (i, obj) in object_manager.objects.iter().enumerate() {
                    ui.selectable_value(
                        &mut md_state.linked_object_idx,
                        Some(i),
                        &obj.name,
                    );
                }
            });

        ui.add_space(10.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(label_sm("PLAYBACK"));
        ui.add_space(6.0);

        // ── Current frame display ─────────────────────────────────────────────
        let n_frames = traj.frame_count();
        let current_time_ps = traj.frames.get(md_state.current_frame)
            .map(|f| f.time_ps)
            .unwrap_or(0.0);

        // Large time display
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("{:.1} ps", current_time_ps))
                    .size(24.0)
                    .strong()
                    .color(ACCENT),
            );
            ui.add_space(8.0);
            ui.label(muted_sm(&format!(
                "frame {}/{}",
                md_state.current_frame + 1, n_frames
            )));
        });

        ui.add_space(4.0);

        // Frame slider
        let mut frame_idx = md_state.current_frame;
        let slider = ui.add(
            egui::Slider::new(&mut frame_idx, 0..=n_frames.saturating_sub(1))
                .show_value(false)
                .text("Frame"),
        );
        if slider.drag_stopped() || slider.changed() {
            md_state.current_frame = frame_idx;
            md_state.is_playing = false;
            md_state.frame_timer = 0.0;
        }

        ui.add_space(6.0);

        // ── Play controls ─────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            // |<< First
            if ghost_button(ui, "⏮").clicked() {
                md_state.current_frame = 0;
                md_state.is_playing = false;
                md_state.frame_timer = 0.0;
            }
            ui.add_space(2.0);
            // -1
            if ghost_button(ui, "◀").clicked() {
                if md_state.current_frame > 0 {
                    md_state.current_frame -= 1;
                }
                md_state.is_playing = false;
                md_state.frame_timer = 0.0;
            }
            ui.add_space(2.0);
            // Play / Pause
            if md_state.is_playing {
                if accent_button(ui, "⏸ Pause").clicked() {
                    md_state.is_playing = false;
                }
            } else {
                if primary_button(ui, "▶ Play").clicked() {
                    md_state.is_playing = true;
                    md_state.frame_timer = 0.0;
                    // If at last frame, restart
                    if md_state.current_frame + 1 >= n_frames {
                        md_state.current_frame = 0;
                    }
                }
            }
            ui.add_space(2.0);
            // +1
            if ghost_button(ui, "▶").clicked() {
                if md_state.current_frame + 1 < n_frames {
                    md_state.current_frame += 1;
                }
                md_state.is_playing = false;
                md_state.frame_timer = 0.0;
            }
            ui.add_space(2.0);
            // >>| Last
            if ghost_button(ui, "⏭").clicked() {
                md_state.current_frame = n_frames.saturating_sub(1);
                md_state.is_playing = false;
                md_state.frame_timer = 0.0;
            }
        });

        ui.add_space(8.0);

        // ── Speed & loop ──────────────────────────────────────────────────────
        ui.label(label_sm("PLAYBACK SETTINGS"));
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label(muted_sm("Speed"));
            ui.add_space(4.0);
            ui.add(
                egui::Slider::new(&mut md_state.playback_speed, 1.0..=120.0)
                    .text("fps")
                    .logarithmic(true),
            );
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut md_state.loop_playback, "Loop");
        });

    } else if !md_state.is_loading {
        ui.add_space(10.0);
        inset_frame().show(ui, |ui| {
            ui.label(muted_sm("Import a trajectory file to begin."));
            ui.add_space(4.0);
            ui.label(muted_sm(
                "Supported: GROMACS XTC (.xtc). Compatible with topology loaded via 'Import Structure'.",
            ));
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector tab: Transform
// ─────────────────────────────────────────────────────────────────────────────

fn render_transform_tab(ui: &mut egui::Ui, obj: &mut crate::graphics::ProteinObject, console_state: &mut ConsoleState) {
    ui.label(strong("Coordinate Adjustment & Export"));
    ui.add_space(8.0);
    
    // Select Target
    ui.horizontal(|ui| {
        ui.label(label_sm("Select Target:"));
        egui::ComboBox::from_id_source("transform_target_combo")
            .selected_text(match console_state.transform_target {
                None => "Select...".to_string(),
                Some(TransformTarget::Protein) => "Main Protein".to_string(),
                Some(TransformTarget::Ligand(i)) => {
                    if i < obj.protein.ligands.len() {
                        format!("Ligand {} ({})", i, obj.protein.ligands[i].name)
                    } else {
                        "Unknown Ligand".to_string()
                    }
                }
            })
            .show_ui(ui, |ui| {
                if ui.selectable_value(&mut console_state.transform_target, Some(TransformTarget::Protein), "Main Protein").clicked() {
                    console_state.transform_xyz = [0.0, 0.0, 0.0];
                }
                for (i, lig) in obj.protein.ligands.iter().enumerate() {
                    if ui.selectable_value(&mut console_state.transform_target, Some(TransformTarget::Ligand(i)), format!("Ligand {} ({})", i, lig.name)).clicked() {
                        console_state.transform_xyz = [0.0, 0.0, 0.0];
                    }
                }
            });
    });
    
    ui.add_space(8.0);
    divider(ui);
    ui.add_space(8.0);
    
    if let Some(target) = console_state.transform_target {
        ui.label(label_sm("Translate (XYZ):"));
        
        let mut x = console_state.transform_xyz[0];
        let mut y = console_state.transform_xyz[1];
        let mut z = console_state.transform_xyz[2];
        
        let mut changed = false;
        
        ui.horizontal(|ui| {
            ui.label("X");
            if ui.add(egui::Slider::new(&mut x, -50.0..=50.0).text("Å")).changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("Y");
            if ui.add(egui::Slider::new(&mut y, -50.0..=50.0).text("Å")).changed() { changed = true; }
        });
        ui.horizontal(|ui| {
            ui.label("Z");
            if ui.add(egui::Slider::new(&mut z, -50.0..=50.0).text("Å")).changed() { changed = true; }
        });
        
        if changed {
            let dx = x - console_state.transform_xyz[0];
            let dy = y - console_state.transform_xyz[1];
            let dz = z - console_state.transform_xyz[2];
            
            console_state.transform_xyz = [x, y, z];
            
            // Apply delta to coordinates
            match target {
                TransformTarget::Protein => {
                    for chain in &mut obj.protein.chains {
                        for res in &mut chain.residues {
                            for atom in &mut res.atoms {
                                atom.x += dx as f64;
                                atom.y += dy as f64;
                                atom.z += dz as f64;
                            }
                        }
                    }
                }
                TransformTarget::Ligand(idx) => {
                    if idx < obj.protein.ligands.len() {
                        for atom in &mut obj.protein.ligands[idx].atoms {
                            atom.x += dx as f64;
                            atom.y += dy as f64;
                            atom.z += dz as f64;
                        }
                    }
                }
            }
            obj.transform_dirty = true;
        }
        
        ui.add_space(16.0);
        section_title(ui, "Export");
        ui.add_space(4.0);
        
        ui.horizontal(|ui| {
            if primary_button(ui, "Export Selected (PDB)").clicked() {
                let content = crate::io::export::export_pdb(&obj.protein, Some(target));
                spawn_save_dialog_generic(content, "export.pdb", "pdb");
            }
            if let TransformTarget::Ligand(idx) = target {
                if idx < obj.protein.ligands.len() {
                    if primary_button(ui, "Export Ligand (SDF)").clicked() {
                        let content = crate::io::export::export_sdf(&obj.protein.ligands[idx]);
                        spawn_save_dialog_generic(content, "ligand.sdf", "sdf");
                    }
                }
            }
        });
    } else {
        ui.label(muted_sm("Select a target above to adjust its coordinates."));
    }
    
    ui.add_space(8.0);
    divider(ui);
    ui.add_space(8.0);
    
    if secondary_button(ui, "Export Whole Scene (PDB)").clicked() {
        let content = crate::io::export::export_pdb(&obj.protein, None);
        spawn_save_dialog_generic(content, "scene.pdb", "pdb");
    }
}

fn spawn_save_dialog_generic(content: String, default_name: &str, ext: &str) {
    let task_pool = bevy::tasks::IoTaskPool::get();
    let ext_str = ext.to_string();
    let default_name_str = default_name.to_string();
    let task = task_pool.spawn(async move {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(&default_name_str)
            .add_filter("Structure", &[&ext_str])
            .save_file()
            .await;
        if let Some(f) = file {
            let _ = std::fs::write(f.path(), content);
        }
    });
    task.detach();
}

// ─────────────────────────────────────────────────────────────────────────────
// Console (bottom panel)
// ─────────────────────────────────────────────────────────────────────────────

fn render_console(ctx: &egui::Context, console_state: &mut ConsoleState) {
    egui::TopBottomPanel::bottom("console_panel")
        .exact_height(32.0)
        .frame(
            egui::Frame::none()
                .fill(PANEL_BG)
                .inner_margin(egui::Margin::symmetric(16.0, 0.0))
                .stroke(egui::Stroke::new(1.0_f32, SEPARATOR)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.label(egui::RichText::new("›").color(ACCENT).size(13.0).strong());
                ui.add_space(6.0);
                // Show the most recent log entry as a status message
                if let Some(last) = console_state.logs.last() {
                    ui.label(egui::RichText::new(last).color(MUTED).size(12.0));
                } else {
                    ui.label(muted_sm("Ready."));
                }
            });
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurement overlay — floating window above viewport
// ─────────────────────────────────────────────────────────────────────────────

fn render_measure_overlay(ctx: &egui::Context, measure_state: &mut MeasureState) {
    // Only show if at least one atom is picked
    if measure_state.first.is_none() {
        return;
    }

    // Accent yellow matching gizmo color
    let yellow = egui::Color32::from_rgb(255, 218, 26);
    let green  = egui::Color32::from_rgb(76, 220, 130);

    egui::Window::new("measure_overlay")
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -52.0))
        .fixed_size(egui::vec2(340.0, 0.0))
        .frame(
            egui::Frame::none()
                .fill(CARD)
                .rounding(egui::Rounding::same(ROUND_MD))
                .shadow(card_shadow())
                .inner_margin(egui::Margin::same(14.0))
                .stroke(egui::Stroke::new(1.5_f32, BORDER)),
        )
        .show(ctx, |ui| {
            // Header row
            ui.horizontal(|ui| {
                // Ruler icon dot
                let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 5.0, yellow);
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Distance measurement").strong().color(TEXT).size(13.5));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ghost_button(ui, "Clear").clicked() {
                        measure_state.clear();
                    }
                });
            });
            ui.add_space(2.0);
            ui.label(muted_sm("Ctrl + Right-click to pick an atom"));

            ui.add_space(8.0);
            divider(ui);
            ui.add_space(6.0);

            // Atom 1
            ui.horizontal(|ui| {
                let (dot, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().circle_filled(dot.center(), 6.0, yellow);
                ui.add_space(4.0);
                ui.label(label_sm("ATOM 1"));
            });
            ui.add_space(2.0);
            if let Some(ref a) = measure_state.first {
                ui.label(
                    egui::RichText::new(&a.label).color(TEXT).size(13.0).strong(),
                );
                ui.label(muted_sm(&format!(
                    "({:.2}, {:.2}, {:.2})",
                    a.world_pos.x, a.world_pos.y, a.world_pos.z
                )));
            }

            ui.add_space(8.0);

            // Atom 2
            ui.horizontal(|ui| {
                let (dot, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().circle_filled(dot.center(), 6.0, green);
                ui.add_space(4.0);
                ui.label(label_sm("ATOM 2"));
            });
            ui.add_space(2.0);
            if let Some(ref b) = measure_state.second {
                ui.label(
                    egui::RichText::new(&b.label).color(TEXT).size(13.0).strong(),
                );
                ui.label(muted_sm(&format!(
                    "({:.2}, {:.2}, {:.2})",
                    b.world_pos.x, b.world_pos.y, b.world_pos.z
                )));
            } else {
                ui.label(muted_sm("Not selected — Ctrl+Right-click a second atom"));
            }

            // Distance result
            if let Some(dist) = measure_state.distance_angstrom {
                ui.add_space(10.0);
                divider(ui);
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    ui.label(muted_sm("Distance"));
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("{:.3} Å", dist))
                            .color(yellow)
                            .size(22.0)
                            .strong(),
                    );
                    // Contextual note
                    ui.add_space(12.0);
                    let note = if dist < 1.2 {
                        "covalent ?"
                    } else if dist < 2.0 {
                        "very short"
                    } else if dist < 3.5 {
                        "H-bond range"
                    } else if dist < 4.5 {
                        "close contact"
                    } else if dist < 8.0 {
                        "medium range"
                    } else {
                        "long range"
                    };
                    tag_pill(ui, note, ACCENT, TAG_BLUE_BG);
                });
            }
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Central viewport
// ─────────────────────────────────────────────────────────────────────────────

fn render_central_view(ctx: &egui::Context, object_manager: &ObjectManager, console_state: &mut ConsoleState, do_json_import: &mut bool) {
    let map_obj_idx = object_manager
        .objects
        .iter()
        .position(|obj| obj.viz_state.show_2d_map && !obj.interactions.is_empty());

    if let Some(idx) = map_obj_idx {
        // 2D interaction map — fill the whole central area
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(APP_BG).inner_margin(12.0))
            .show(ctx, |ui| {
                card_frame().show(ui, |ui| {
                    // ── Toolbar row ───────────────────────────────────────────
                    ui.horizontal(|ui| {
                        ui.label(strong("2D Interaction Map"));
                        ui.add_space(8.0);
                        tag_pill(ui, "Live", GREEN, TAG_GREEN_BG);
                        // Show external data badge if loaded
                        if let Some(ref ext) = console_state.external_interactions {
                            ui.add_space(4.0);
                            tag_pill(ui, "JSON", ACCENT, TAG_BLUE_BG);
                            ui.label(muted_sm(&format!("{} interactions", ext.interactions.len())));
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // Reset view + clear overrides
                            if ghost_button(ui, "Reset view").clicked() {
                                console_state.map_zoom = 1.0;
                                console_state.map_pan = egui::Vec2::ZERO;
                                console_state.map_atom_overrides.clear();
                                console_state.map_residue_overrides.clear();
                                console_state.map_drag_item = None;
                            }
                            ui.add_space(4.0);
                            // Clear imported JSON
                            if console_state.external_interactions.is_some() {
                                if ghost_button(ui, "Clear JSON").clicked() {
                                    console_state.external_interactions = None;
                                }
                                ui.add_space(4.0);
                            }
                            // Import JSON button — async, non-blocking
                            if ghost_button(ui, "Import JSON").clicked() {
                                *do_json_import = true;
                            }
                            ui.add_space(8.0);
                            // Zoom controls
                            if ghost_button(ui, "−").clicked() {
                                console_state.map_zoom = (console_state.map_zoom / 1.25).max(0.1);
                            }
                            ui.label(muted_sm(&format!("{:.0}%", console_state.map_zoom * 100.0)));
                            if ghost_button(ui, "+").clicked() {
                                console_state.map_zoom = (console_state.map_zoom * 1.25).min(8.0);
                            }
                        });
                    });

                    ui.add_space(4.0);

                    // ── Interactive canvas ────────────────────────────────────
                    let available = ui.available_rect_before_wrap();
                    let (response, painter) = ui.allocate_painter(
                        available.size(),
                        egui::Sense::click_and_drag(),
                    );
                    let rect = response.rect;

                    // Scroll-to-zoom
                    let scroll_delta = ui.input(|i| i.raw_scroll_delta);
                    if response.hovered() && scroll_delta.y != 0.0 {
                        let zoom_factor = if scroll_delta.y > 0.0 { 1.1 } else { 1.0 / 1.1 };
                        let mouse_pos = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
                        let pivot = mouse_pos - rect.center();
                        console_state.map_pan = (console_state.map_pan - pivot) * zoom_factor + pivot;
                        console_state.map_zoom = (console_state.map_zoom * zoom_factor).clamp(0.1, 8.0);
                    }

                    let zoom = console_state.map_zoom;
                    let pan  = console_state.map_pan;

                    // screen → layout space
                    let to_layout = |sp: egui::Pos2| -> egui::Vec2 {
                        (sp - rect.center() - pan) / zoom
                    };

                    let obj = &object_manager.objects[idx];

                    // ── Drag handling ─────────────────────────────────────────
                    if response.drag_started() {
                        // Hit-test: find atom or residue node under cursor
                        if let Some(cursor) = ui.input(|i| i.pointer.press_origin()) {
                            let lp = to_layout(cursor);

                            // Ask interaction_map to compute positions so we can hit-test
                            let (atom_pos, res_pos) =
                                crate::ui::interaction_map::compute_positions_for_hittest(
                                    obj,
                                    &console_state.map_atom_overrides,
                                    &console_state.map_residue_overrides,
                                    console_state.external_interactions.as_ref(),
                                );

                            // Check residue nodes first (larger, on top visually)
                            let node_r = 30.0_f32;
                            let mut hit: Option<crate::ui::MapDragItem> = None;
                            for (label, pos) in &res_pos {
                                if (lp - *pos).length() <= node_r {
                                    hit = Some(crate::ui::MapDragItem::Residue(label.clone()));
                                    break;
                                }
                            }
                            // Then check ligand atoms
                            if hit.is_none() {
                                let atom_r = 14.0_f32;
                                for (idx_a, pos) in &atom_pos {
                                    if (lp - *pos).length() <= atom_r {
                                        hit = Some(crate::ui::MapDragItem::Atom(*idx_a));
                                        break;
                                    }
                                }
                            }
                            console_state.map_drag_item = hit;
                        }
                    }

                    if response.dragged() {
                        let delta_layout = response.drag_delta() / zoom;
                        match &console_state.map_drag_item.clone() {
                            Some(crate::ui::MapDragItem::Atom(atom_idx)) => {
                                let (atom_pos, _) =
                                    crate::ui::interaction_map::compute_positions_for_hittest(
                                        obj,
                                        &console_state.map_atom_overrides,
                                        &console_state.map_residue_overrides,
                                        console_state.external_interactions.as_ref(),
                                    );
                                let cur = atom_pos.get(atom_idx).copied()
                                    .unwrap_or(egui::Vec2::ZERO);
                                console_state.map_atom_overrides
                                    .insert(*atom_idx, cur + delta_layout);
                            }
                            Some(crate::ui::MapDragItem::Residue(label)) => {
                                let (_, res_pos) =
                                    crate::ui::interaction_map::compute_positions_for_hittest(
                                        obj,
                                        &console_state.map_atom_overrides,
                                        &console_state.map_residue_overrides,
                                        console_state.external_interactions.as_ref(),
                                    );
                                let cur = res_pos.get(label).copied()
                                    .unwrap_or(egui::Vec2::ZERO);
                                let new_label = label.clone();
                                console_state.map_residue_overrides
                                    .insert(new_label, cur + delta_layout);
                            }
                            None => {
                                console_state.map_pan += response.drag_delta();
                            }
                        }
                    }

                    if response.drag_stopped() {
                        console_state.map_drag_item = None;
                    }

                    // Fill background
                    painter.rect_filled(rect, egui::Rounding::same(ROUND_MD), CARD);

                    // Show hand cursor when hovering over a draggable node
                    if response.hovered() {
                        if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
                            let lp = to_layout(cursor);
                            let (ap, rp) = crate::ui::interaction_map::compute_positions_for_hittest(
                                obj,
                                &console_state.map_atom_overrides,
                                &console_state.map_residue_overrides,
                                console_state.external_interactions.as_ref(),
                            );
                            let near = rp.values().any(|p| (lp - *p).length() <= 30.0)
                                || ap.values().any(|p| (lp - *p).length() <= 14.0);
                            if near {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                            }
                        }
                    }

                    // Draw map
                    crate::ui::interaction_map::draw_2d_interaction_map_inline(
                        &painter,
                        rect,
                        obj,
                        zoom,
                        pan,
                        &console_state.map_atom_overrides,
                        &console_state.map_residue_overrides,
                        console_state.external_interactions.as_ref(),
                    );

                    // Hint text bottom-right
                    painter.text(
                        rect.right_bottom() + egui::vec2(-12.0, -12.0),
                        egui::Align2::RIGHT_BOTTOM,
                        "Scroll to zoom  ·  Drag to pan",
                        egui::FontId::proportional(11.0),
                        MUTED,
                    );
                });
            });
    } else {
        // ── Viewport toolbar (sits above Bevy 3D render, fully opaque) ─────
        egui::TopBottomPanel::top("viewport_toolbar")
            .exact_height(44.0)
            .frame(
                egui::Frame::none()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::symmetric(16.0, 0.0))
                    .stroke(egui::Stroke::new(1.0_f32, BORDER)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    let (dot_r, _) = ui.allocate_exact_size(
                        egui::vec2(10.0, 10.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().circle_filled(dot_r.center(), 4.0, ACCENT);
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("3D Viewport").strong().color(TEXT).size(13.0));
                    ui.add_space(12.0);
                    ui.label(muted_sm("Drag to orbit  ·  Scroll to zoom  ·  Right-drag to pan"));
                });
            });

        // ── Empty-state hint overlay (only when no structure loaded) ────────
        if object_manager.objects.is_empty() {
            egui::CentralPanel::default()
                .frame(egui::Frame::none().fill(egui::Color32::TRANSPARENT))
                .show(ctx, |ui| {
                    // Centered hint card — rendered on top of the transparent Bevy view
                    let center = ui.max_rect().center();
                    let hint_rect = egui::Rect::from_center_size(
                        center,
                        egui::vec2(300.0, 106.0),
                    );
                    ui.painter().rect_filled(
                        hint_rect,
                        egui::Rounding::same(ROUND_MD),
                        egui::Color32::from_rgba_unmultiplied(22, 27, 40, 220),
                    );
                    ui.painter().rect_stroke(
                        hint_rect,
                        egui::Rounding::same(ROUND_MD),
                        egui::Stroke::new(1.0_f32, BORDER),
                    );
                    // Plus icon circle
                    ui.painter().circle_filled(
                        hint_rect.center() + egui::vec2(0.0, -24.0),
                        18.0,
                        TAG_BLUE_BG,
                    );
                    ui.painter().text(
                        hint_rect.center() + egui::vec2(0.0, -24.0),
                        egui::Align2::CENTER_CENTER,
                        "+",
                        egui::FontId::proportional(22.0),
                        ACCENT,
                    );
                    ui.painter().text(
                        hint_rect.center() + egui::vec2(0.0, 8.0),
                        egui::Align2::CENTER_CENTER,
                        "Open a PDB or mmCIF file",
                        egui::FontId::proportional(14.0),
                        TEXT,
                    );
                    ui.painter().text(
                        hint_rect.center() + egui::vec2(0.0, 28.0),
                        egui::Align2::CENTER_CENTER,
                        "Use 'Open structure' in the toolbar",
                        egui::FontId::proportional(11.5),
                        MUTED,
                    );
                });
        } else {
            // Structure loaded — transparent CentralPanel so Bevy renders freely
            egui::CentralPanel::default()
                .frame(egui::Frame::none().fill(egui::Color32::TRANSPARENT))
                .show(ctx, |_ui| {});
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// File dialog + alignment logic (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

fn spawn_file_dialog(commands: &mut Commands) {
    let task_pool = IoTaskPool::get();
    let task = task_pool.spawn(async move {
        let file = AsyncFileDialog::new()
            .add_filter("Structure Files", &["pdb", "cif", "mmcif"])
            .pick_file()
            .await;
        file.map(|f| f.path().to_path_buf())
    });
    commands.spawn(FileDialogTask(task));
}

fn poll_file_dialog_task(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut FileDialogTask)>,
    mut load_event_writer: EventWriter<LoadFileEvent>,
) {
    for (entity, mut task) in &mut tasks {
        if let Some(result) = future::block_on(future::poll_once(&mut task.0)) {
            if let Some(path) = result {
                load_event_writer.send(LoadFileEvent(path));
            }
            commands.entity(entity).despawn();
        }
    }
}

fn spawn_json_import_dialog(commands: &mut Commands) {
    let task_pool = IoTaskPool::get();
    let task = task_pool.spawn(async move {
        let file = AsyncFileDialog::new()
            .add_filter("Interaction JSON", &["json"])
            .pick_file()
            .await;
        if let Some(f) = file {
            let fname = f.file_name();
            if let Ok(content) = std::fs::read_to_string(f.path()) {
                return Some((fname, content));
            }
        }
        None
    });
    commands.spawn(JsonImportTask(task));
}

fn poll_json_import_task(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut JsonImportTask)>,
    mut console_state: ResMut<ConsoleState>,
) {
    for (entity, mut task) in &mut tasks {
        if let Some(result) = future::block_on(future::poll_once(&mut task.0)) {
            commands.entity(entity).despawn();
            if let Some((fname, content)) = result {
                match crate::io::interaction_import::ExternalInteractionData::from_json(
                    &content, &fname,
                ) {
                    Ok(data) => {
                        console_state.map_atom_overrides.clear();
                        console_state.map_residue_overrides.clear();
                        console_state.map_drag_item = None;
                        let n = data.interactions.len();
                        console_state.logs.push(format!(
                            "Loaded {} interactions from {}", n, fname
                        ));
                        console_state.external_interactions = Some(data);
                    }
                    Err(e) => {
                        console_state.logs.push(format!("JSON import error: {}", e));
                    }
                }
            }
        }
    }
}

fn run_alignment(object_manager: &mut ObjectManager, console_state: &mut ConsoleState) {
    let target_idx = console_state.align_target;
    let mobile_idx = console_state.align_mobile;

    if target_idx >= object_manager.objects.len()
        || mobile_idx >= object_manager.objects.len()
        || target_idx == mobile_idx
    {
        console_state
            .logs
            .push("Choose two different valid structures for alignment.".into());
        return;
    }

    let target = &object_manager.objects[target_idx].protein;
    let mobile = &object_manager.objects[mobile_idx].protein;

    let Some(result) = align_structures_sequence_aware(
        target,
        mobile,
        &console_state.align_target_chain,
        &console_state.align_mobile_chain,
        console_state.align_mode,
    ) else {
        console_state.logs.push(
            "Alignment failed: check chain IDs and ensure matched Cα pairs exist.".into(),
        );
        return;
    };

    let (summary, r, c_t, c_m) = result;
    apply_alignment(
        &mut object_manager.objects[mobile_idx].protein,
        &r,
        &c_t,
        &c_m,
    );
    object_manager.objects[mobile_idx].changed = true;
    console_state.logs.push(format!(
        "Aligned {} → {} | RMSD {:.3} Å | {} Cα pairs | identity {:.1}% | similarity {:.1}%",
        object_manager.objects[mobile_idx].name,
        object_manager.objects[target_idx].name,
        summary.rmsd,
        summary.matched_pairs,
        summary.sequence_identity * 100.0,
        summary.sequence_similarity * 100.0,
    ));
}

fn chain_id_field(
    ui: &mut egui::Ui,
    label: &str,
    chain_id: &mut String,
    object_manager: &ObjectManager,
    obj_idx: usize,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.text_edit_singleline(chain_id);
        if ui.button("Pick…").clicked() {
            if let Some(obj) = object_manager.objects.get(obj_idx) {
                if let Some(id) = default_protein_chain_id(&obj.protein) {
                    *chain_id = id;
                }
            }
        }
    });
}

fn sequence_records_from_objects(object_manager: &ObjectManager) -> Vec<SequenceRecord> {
    let mut records = Vec::new();
    for obj in &object_manager.objects {
        for chain in &obj.protein.chains {
            let seq = chain_sequence(chain);
            if seq.is_empty() {
                continue;
            }
            records.push(SequenceRecord {
                label: format!("{}|{}", obj.name, chain.id),
                sequence: seq,
            });
        }
    }
    records
}

fn build_similarity_from_objects(
    object_manager: &ObjectManager,
) -> crate::core::seq_align::SimilarityMatrix {
    similarity_matrix(&sequence_records_from_objects(object_manager))
}

fn format_alignment_preview(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    format!("{}…", &s[..max_len])
}

fn similarity_matrix_ui(ui: &mut egui::Ui, mat: &crate::core::seq_align::SimilarityMatrix) {
    let n = mat.labels.len();
    if n == 0 {
        return;
    }
    egui::ScrollArea::horizontal().show(ui, |ui| {
        egui::Grid::new("sim_matrix")
            .num_columns(n + 1)
            .spacing(egui::vec2(6.0, 2.0))
            .show(ui, |ui| {
                ui.label("");
                for label in &mat.labels {
                    ui.label(
                        egui::RichText::new(label)
                            .size(10.0)
                            .color(MUTED),
                    );
                }
                ui.end_row();
                for i in 0..n {
                    ui.label(
                        egui::RichText::new(&mat.labels[i])
                            .size(10.0)
                            .color(MUTED),
                    );
                    for j in 0..n {
                        let text = if i == j {
                            "100".to_string()
                        } else if i > j {
                            format!("{:.0}", mat.identity[i][j] * 100.0)
                        } else {
                            format!("{:.0}", mat.similarity[i][j] * 100.0)
                        };
                        ui.label(egui::RichText::new(text).monospace().size(10.0));
                    }
                    ui.end_row();
                }
            });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Combo-box helpers
// ─────────────────────────────────────────────────────────────────────────────

fn object_selector(
    ui: &mut egui::Ui,
    label: &str,
    selected: &mut usize,
    object_manager: &ObjectManager,
) {
    let selected_text = object_manager
        .objects
        .get(*selected)
        .map(|obj| obj.name.clone())
        .unwrap_or_else(|| "None".into());

    ui.horizontal(|ui| {
        ui.label(muted_sm(label));
        egui::ComboBox::from_id_source(format!("{}_selector", label))
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                for (idx, obj) in object_manager.objects.iter().enumerate() {
                    ui.selectable_value(selected, idx, &obj.name);
                }
            });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Display toggle helpers
// ─────────────────────────────────────────────────────────────────────────────

fn display_toggles_chain(ui: &mut egui::Ui, state: &mut ChainVizState, changed: &mut bool) {
    ui.horizontal_wrapped(|ui| {
        if ui.checkbox(&mut state.ribbon, "Ribbon").changed() { *changed = true; }
        if ui.checkbox(&mut state.sticks, "Sticks").changed() { *changed = true; }
        if ui.checkbox(&mut state.spacefill, "Spacefill").changed() { *changed = true; }
        if ui.checkbox(&mut state.wireframe, "Wireframe").changed() { *changed = true; }
        if ui.checkbox(&mut state.surface, "Surface").changed() { *changed = true; }
        if ui.checkbox(&mut state.eps_surface, "EPS").changed() { *changed = true; }
    });
}

fn display_toggles_ligand(ui: &mut egui::Ui, state: &mut LigandVizState, changed: &mut bool) {
    ui.horizontal_wrapped(|ui| {
        if ui.checkbox(&mut state.sticks, "Sticks").changed() { *changed = true; }
        if ui.checkbox(&mut state.spacefill, "Spacefill").changed() { *changed = true; }
        if ui.checkbox(&mut state.wireframe, "Wireframe").changed() { *changed = true; }
        if ui.checkbox(&mut state.surface, "Surface").changed() { *changed = true; }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Data helpers
// ─────────────────────────────────────────────────────────────────────────────

struct StructureStats {
    chains: usize,
    residues: usize,
    atoms: usize,
    ligands: usize,
    ions: usize,
}

fn structure_stats(obj: &crate::graphics::ProteinObject) -> StructureStats {
    let ligands = obj.protein.ligands.iter().filter(|l| l.ligand_type == LigandType::Ligand).count();
    let ions = obj.protein.ligands.iter().filter(|l| l.ligand_type == LigandType::Ion).count();
    StructureStats {
        chains: obj.protein.chains.len(),
        residues: obj.protein.residue_count(),
        atoms: obj.protein.atom_count(),
        ligands,
        ions,
    }
}

fn secondary_counts(chain: &crate::core::protein::Chain) -> (usize, usize, usize) {
    let mut helix = 0;
    let mut sheet = 0;
    let mut coil = 0;
    for residue in &chain.residues {
        match residue.secondary_structure {
            SecondaryStructure::Helix => helix += 1,
            SecondaryStructure::Sheet => sheet += 1,
            SecondaryStructure::Turn | SecondaryStructure::Coil => coil += 1,
        }
    }
    (helix, sheet, coil)
}

fn molecule_type_label(molecule_type: MoleculeType) -> &'static str {
    match molecule_type {
        MoleculeType::Protein => "Protein",
        MoleculeType::RNA => "RNA",
        MoleculeType::DNA => "DNA",
        MoleculeType::SmallMolecule => "Small molecule",
    }
}

fn ligand_type_label(ligand_type: LigandType) -> &'static str {
    match ligand_type {
        LigandType::Ligand => "ligand",
        LigandType::Ion => "ion",
    }
}

fn interaction_label(i_type: &crate::core::interactions::InteractionType) -> &'static str {
    match i_type {
        crate::core::interactions::InteractionType::HydrogenBond => "H-bond",
        crate::core::interactions::InteractionType::SaltBridge => "Salt bridge",
        crate::core::interactions::InteractionType::Hydrophobic => "Hydrophobic",
        crate::core::interactions::InteractionType::PiPiStacking => "Pi-pi",
    }
}

fn sequence_preview(sequence: &str, max_len: usize) -> String {
    if sequence.len() <= max_len {
        return sequence.into();
    }
    format!("{}…", &sequence[..max_len])
}

// ─────────────────────────────────────────────────────────────────────────────
// UI typography helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Bold body text in primary color
fn strong(text: &str) -> egui::RichText {
    egui::RichText::new(text).strong().color(TEXT).size(14.0)
}

/// Small muted secondary text
fn muted_sm(text: &str) -> egui::RichText {
    egui::RichText::new(text).color(MUTED).size(12.5)
}

/// ALL-CAPS section label (like "STRUCTURES", "OVERVIEW")
fn label_sm(text: &str) -> egui::RichText {
    egui::RichText::new(text).color(MUTED).size(11.5).strong()
}

/// Section heading (card title)
fn section_title(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).strong().size(16.0).color(TEXT));
}

/// Thin horizontal rule
fn divider(ui: &mut egui::Ui) {
    let available = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(available, 1.0), egui::Sense::hover());
    ui.painter().hline(rect.x_range(), rect.center().y, egui::Stroke::new(1.0, SEPARATOR));
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid + list row helpers
// ─────────────────────────────────────────────────────────────────────────────

fn metric_grid(ui: &mut egui::Ui, values: &[(&str, String)]) {
    // Build a stable unique ID from the parent widget's ID + first label text
    // This avoids pointer-based IDs that collide when slices share memory
    let first_label = values.first().map(|(l, _)| *l).unwrap_or("empty");
    let grid_id = ui.id().with(first_label).with(values.len());
    egui::Grid::new(grid_id)
        .num_columns(2)
        .spacing(egui::vec2(20.0, 4.0))
        .show(ui, |ui| {
            for (label, value) in values {
                ui.label(muted_sm(*label));
                ui.label(egui::RichText::new(value).color(TEXT).size(13.0).strong());
                ui.end_row();
            }
        });
}

fn pipeline_row(ui: &mut egui::Ui, step: &str, text: &str, enabled: bool) {
    ui.add_space(3.0);
    ui.horizontal(|ui| {
        tag_pill(
            ui,
            step,
            if enabled { ACCENT } else { MUTED },
            if enabled { TAG_BLUE_BG } else { SEPARATOR },
        );
        ui.label(if enabled {
            egui::RichText::new(text).color(TEXT).size(12.5)
        } else {
            egui::RichText::new(text).color(MUTED).size(12.5)
        });
    });
}

fn discovery_row(ui: &mut egui::Ui, title: &str, ready: bool, detail: &str) {
    ui.add_space(5.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong().color(TEXT).size(13.0));
        ui.add_space(4.0);
        tag_pill(
            ui,
            if ready { "Ready" } else { "Planned" },
            if ready { GREEN } else { MUTED },
            if ready { TAG_GREEN_BG } else { SEPARATOR },
        );
    });
    ui.add_space(1.0);
    ui.label(muted_sm(detail));
}

// ─────────────────────────────────────────────────────────────────────────────
// MD file dialog spawners
// ─────────────────────────────────────────────────────────────────────────────

/// Known GROMACS membrane/solvent/detergent residue names to exclude from 3D rendering.
/// Chains whose ALL residues appear in this list will be stripped from the topology PDB
/// so the renderer only handles the biologically interesting molecules (protein, peptide, etc.).
const MEMBRANE_RESIDUE_NAMES: &[&str] = &[
    // Detergents & lipids (GROMACS forcefields)
    "DDM", "DPE", "DPP", "LH0", "LP0",
    "DPPC", "DOPC", "POPE", "POPC", "POPS", "POPA", "POPG",
    "DLPC", "DLPE", "DMPC", "DMPE",
    "CHOL", "CHL1",
    // Solvent
    "TIP3", "TIP4", "TIP5", "WAT", "SOL", "HOH", "TP3", "SPC",
    // Common ions
    "NA", "NA+", "CL", "CL-", "K", "K+", "MG", "CA", "ZN",
    "SOD", "CLA", "POT", "CAL",
];

/// Returns true if all residues in this chain are membrane/solvent/ion molecules.
fn is_membrane_chain(residues: &[crate::core::protein::Residue]) -> bool {
    !residues.is_empty()
        && residues.iter().all(|r| MEMBRANE_RESIDUE_NAMES.contains(&r.name.as_str()))
}

/// Open a file dialog to pick a PDB topology for MD, parse it, keep all chains.
/// We do NOT filter by molecule type — the PMB system has non-standard residue names
/// that would be incorrectly stripped. Instead, we let the full system load and
/// rely on atom.serial-based XTC mapping to animate correctly.
fn spawn_md_topology_dialog(commands: &mut Commands) {
    let task_pool = IoTaskPool::get();
    let task = task_pool.spawn(async move {
        let file = AsyncFileDialog::new()
            .add_filter("Structure Files", &["pdb", "cif", "mmcif"])
            .pick_file()
            .await;
        if let Some(f) = file {
            let path = f.path().to_path_buf();
            if let Some(path_str) = path.to_str() {
                match crate::io::parser::load_pdb(path_str) {
                    Ok(mut protein) => {
                        // We now keep all chains and ligands (including water/solvent).
                        // They will be set to hidden by default in graphics/mod.rs so they don't lag the renderer.
                        return Some((path, protein));
                    }
                    Err(e) => {
                        eprintln!("Failed to load PDB: {:?}", e);
                    }
                }
            }
        }
        None
    });
    commands.spawn(MdTopologyTask(task));
}

/// Open a file dialog to pick an XTC trajectory, then parse it in a blocking thread.
fn spawn_xtc_import_dialog(commands: &mut Commands) {
    let task_pool = IoTaskPool::get();
    let task = task_pool.spawn(async move {
        let file = AsyncFileDialog::new()
            .add_filter("GROMACS Trajectory", &["xtc"])
            .pick_file()
            .await;
        if let Some(f) = file {
            let path = f.path().to_path_buf();
            // Parse XTC synchronously (we're already on a worker thread)
            let result = crate::io::xtc_parser::parse_xtc(path.to_str().unwrap_or(""))
                .map_err(|e| e.to_string());
            Some((path, result))
        } else {
            None
        }
    });
    commands.spawn(XtcImportTask(task));
}

// ─────────────────────────────────────────────────────────────────────────────
// MD task pollers
// ─────────────────────────────────────────────────────────────────────────────

fn poll_md_topology_task(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut MdTopologyTask)>,
    mut md_state: ResMut<MdState>,
    mut protein_loaded_writer: EventWriter<crate::io::ProteinLoadedEvent>,
    mut console_state: ResMut<ConsoleState>,
    object_manager: Res<crate::graphics::ObjectManager>,
) {
    for (entity, mut task) in &mut tasks {
        if let Some(result) = future::block_on(future::poll_once(&mut task.0)) {
            commands.entity(entity).despawn();
            if let Some((path, protein)) = result {
                let n_chains = protein.chains.len();
                let n_atoms: usize = protein.chains.iter()
                    .flat_map(|c| &c.residues)
                    .map(|r| r.atoms.len())
                    .sum();
                console_state.logs.push(format!(
                    "MD topology: {} ({} chains, {} atoms)",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                    n_chains, n_atoms
                ));
                md_state.topology_path = Some(path);
                md_state.status_msg = format!("Topology: {} chains, {} atoms", n_chains, n_atoms);
                // Do NOT auto-link here — object hasn't been added to ObjectManager yet.
                // Trajectory import will auto-link to the last object after it's added.
                // Emit directly as ProteinLoadedEvent (bypasses re-parsing)
                protein_loaded_writer.send(crate::io::ProteinLoadedEvent(protein));
            }
        }
    }
}

fn poll_xtc_import_task(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut XtcImportTask)>,
    mut md_state: ResMut<MdState>,
    mut console_state: ResMut<ConsoleState>,
    mut object_manager: ResMut<crate::graphics::ObjectManager>,
) {
    for (entity, mut task) in &mut tasks {
        if let Some(result) = future::block_on(future::poll_once(&mut task.0)) {
            commands.entity(entity).despawn();
            md_state.is_loading = false;
            if let Some((path, parse_result)) = result {
                match parse_result {
                    Ok(traj) => {
                        let n_frames = traj.frame_count();
                        let n_atoms = traj.n_atoms;
                        let (t0, t1) = traj.time_range_ps().unwrap_or((0.0, 0.0));
                        md_state.trajectory = Some(traj);
                        md_state.trajectory_path = Some(path.clone());
                        md_state.current_frame = 0;
                        md_state.is_playing = false;
                        md_state.frame_timer = 0.0;
                        md_state.status_msg = format!(
                            "Loaded {} frames, {} atoms, {:.0}–{:.0} ps",
                            n_frames, n_atoms, t0, t1
                        );
                        console_state.logs.push(format!(
                            "XTC loaded: {} frames, {} atoms ({:.0}–{:.0} ps)",
                            n_frames, n_atoms, t0, t1
                        ));
                        // Auto-link to last loaded structure if none is set yet
                        if md_state.linked_object_idx.is_none()
                            && !object_manager.objects.is_empty()
                        {
                            md_state.linked_object_idx =
                                Some(object_manager.objects.len() - 1);
                            console_state.logs.push(format!(
                                "MD auto-linked to: {}",
                                object_manager.objects.last()
                                    .map(|o| o.name.as_str())
                                    .unwrap_or("?")
                            ));
                        }
                        // Force apply frame 0 immediately
                        if let Some(obj_idx) = md_state.linked_object_idx {
                            if let Some(ref traj) = md_state.trajectory {
                                if let Some(obj) = object_manager.objects.get_mut(obj_idx) {
                                    if let Some(frame) = traj.frames.first() {
                                        // Apply first frame positions
                                        for chain in &mut obj.protein.chains {
                                            for residue in &mut chain.residues {
                                                for atom in &mut residue.atoms {
                                                    let xtc_idx = (atom.serial.max(1) - 1) as usize;
                                                    if let Some(p) = frame.positions.get(xtc_idx) {
                                                        atom.x = p[0] as f64;
                                                        atom.y = p[1] as f64;
                                                        atom.z = p[2] as f64;
                                                    }
                                                }
                                            }
                                        }
                                        for ligand in &mut obj.protein.ligands {
                                            for atom in &mut ligand.atoms {
                                                let xtc_idx = (atom.serial.max(1) - 1) as usize;
                                                if let Some(p) = frame.positions.get(xtc_idx) {
                                                    atom.x = p[0] as f64;
                                                    atom.y = p[1] as f64;
                                                    atom.z = p[2] as f64;
                                                }
                                            }
                                        }
                                        obj.changed = true;
                                        console_state.logs.push("Applied frame 0".into());
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        md_state.status_msg = format!("Parse error: {}", e);
                        console_state.logs.push(format!("XTC import error: {}", e));
                    }
                }
            }
        }
    }
}
