use crate::core::interactions::InteractionType;
use crate::graphics::ProteinObject;
use bevy_egui::egui;
use bevy_egui::egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Ligand 2D layout
//   1. Project 3D coordinates onto best-fit plane (PCA)
//   2. Scale so median bond length = BOND_PX
//   3. Post-process: push non-bonded overlapping atoms apart
//      (bonded pairs are never moved — preserves real bond geometry)
// ─────────────────────────────────────────────────────────────────────────────

const BOND_PX: f32 = 42.0;

fn compute_ligand_layout(ligand: &crate::core::protein::Ligand) -> HashMap<usize, Vec2> {
    let n = ligand.atoms.len();
    if n == 0 { return HashMap::new(); }
    if n == 1 { let mut m = HashMap::new(); m.insert(0, Vec2::ZERO); return m; }

    // PCA projection
    let mut cent = Vector3::zeros();
    for a in &ligand.atoms {
        cent += Vector3::new(a.x as f32, a.y as f32, a.z as f32);
    }
    cent /= n as f32;

    let mut cov = Matrix3::zeros();
    for a in &ligand.atoms {
        let v = Vector3::new(a.x as f32, a.y as f32, a.z as f32) - cent;
        cov += v * v.transpose();
    }
    let eig = SymmetricEigen::new(cov);
    let mut evecs: Vec<(f32, Vector3<f32>)> = vec![
        (eig.eigenvalues[0], eig.eigenvectors.column(0).into_owned()),
        (eig.eigenvalues[1], eig.eigenvectors.column(1).into_owned()),
        (eig.eigenvalues[2], eig.eigenvectors.column(2).into_owned()),
    ];
    evecs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let ax = evecs[0].1.normalize();
    let ay = evecs[1].1.normalize();

    let proj: Vec<Vec2> = ligand.atoms.iter().map(|a| {
        let v = Vector3::new(a.x as f32, a.y as f32, a.z as f32) - cent;
        Vec2::new(v.dot(&ax), v.dot(&ay))
    }).collect();

    // Build bond list
    let bonds: Vec<(usize, usize)> = if !ligand.bonds.is_empty() {
        ligand.bonds.clone()
    } else {
        let mut p = Vec::new();
        for a in 0..n {
            for b in (a+1)..n {
                let pa = bevy::prelude::Vec3::new(
                    ligand.atoms[a].x as f32, ligand.atoms[a].y as f32, ligand.atoms[a].z as f32);
                let pb = bevy::prelude::Vec3::new(
                    ligand.atoms[b].x as f32, ligand.atoms[b].y as f32, ligand.atoms[b].z as f32);
                if pa.distance(pb) < 2.1 { p.push((a, b)); }
            }
        }
        p
    };

    // Scale to target bond length
    let mut bl: Vec<f32> = bonds.iter()
        .map(|&(a, b)| (proj[a] - proj[b]).length())
        .filter(|&l| l > 0.001).collect();
    let scale = if bl.is_empty() {
        BOND_PX
    } else {
        bl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        BOND_PX / bl[bl.len() / 2]
    };

    let cx = proj.iter().map(|p| p.x).sum::<f32>() / n as f32;
    let cy = proj.iter().map(|p| p.y).sum::<f32>() / n as f32;
    let mut pos: Vec<Vec2> = (0..n).map(|i| {
        Vec2::new((proj[i].x - cx) * scale, (proj[i].y - cy) * scale)
    }).collect();

    // Bond set (skip bonded pairs during separation)
    let bond_set: HashSet<(usize, usize)> = bonds.iter()
        .map(|&(a, b)| (a.min(b), a.max(b))).collect();

    // Heavy atom radii in pixels
    let radii: Vec<f32> = (0..n).map(|i| {
        let el = ligand.atoms[i].element.trim().to_uppercase();
        if el == "C" || el.is_empty() { 5.0_f32 } else { 12.0 }
    }).collect();

    // Push non-bonded overlapping pairs — up to 60 passes
    for _pass in 0..60 {
        let mut any = false;
        for v in 0..n {
            for u in (v+1)..n {
                if bond_set.contains(&(v, u)) { continue; }
                let need = (radii[v] + radii[u] + 2.0).max(26.0);
                let diff = pos[v] - pos[u];
                let d = diff.length();
                if d < need {
                    any = true;
                    let push = if d > 0.001 {
                        diff.normalized() * (need - d) * 0.55
                    } else {
                        Vec2::new(need * 0.55, v as f32 * 0.5)
                    };
                    pos[v] += push;
                    pos[u] -= push;
                }
            }
        }
        if !any { break; }
    }

    // Re-centre
    let mx = pos.iter().map(|p| p.x).sum::<f32>() / n as f32;
    let my = pos.iter().map(|p| p.y).sum::<f32>() / n as f32;
    (0..n).map(|i| (i, pos[i] - Vec2::new(mx, my))).collect()
}

/// Flexible atom name matching: handles "Lig:S1", "S1", " S1 ", "S" variations
fn match_atom_name(pdb_name: &str, json_atom: &str) -> bool {
    let pdb = pdb_name.trim();
    let json = json_atom.trim().trim_start_matches("Lig:");
    if pdb == json { return true; }
    // Try stripping trailing digits from pdb name: "S" matches "S1"
    let pdb_base: String = pdb.chars().take_while(|c| c.is_alphabetic()).collect();
    let json_base: String = json.chars().take_while(|c| c.is_alphabetic()).collect();
    if !pdb_base.is_empty() && !json_base.is_empty() && pdb_base == json_base {
        // Accept if PDB name is prefix of JSON name or vice versa
        return json.starts_with(pdb) || pdb.starts_with(json.split(|c: char| c.is_numeric()).next().unwrap_or(json));
    }
    false
}

/// Find first ligand atom index matching a JSON lig_atom label
fn find_atom_idx(ligand: &crate::core::protein::Ligand, json_atom: &str) -> Option<usize> {
    let json = json_atom.trim().trim_start_matches("Lig:");
    // Exact match first
    if let Some(idx) = ligand.atoms.iter().position(|a| a.name.trim() == json) {
        return Some(idx);
    }
    // Element + number match: "S1" → find atom named "S" or "S1"
    let json_el: String = json.chars().take_while(|c| c.is_alphabetic()).collect();
    let json_num: String = json.chars().skip_while(|c| c.is_alphabetic()).collect();
    if let Some(idx) = ligand.atoms.iter().position(|a| {
        let n = a.name.trim();
        let el: String = n.chars().take_while(|c| c.is_alphabetic()).collect();
        let num: String = n.chars().skip_while(|c| c.is_alphabetic()).collect();
        el == json_el && (json_num.is_empty() || num.is_empty() || num == json_num)
    }) {
        return Some(idx);
    }
    // Element match only (e.g. "S" matches "S1" when there's only one S)
    if let Some(idx) = ligand.atoms.iter().position(|a| {
        let el: String = a.name.trim().chars().take_while(|c| c.is_alphabetic()).collect();
        el == json_el
    }) {
        return Some(idx);
    }
    None
}

pub fn compute_positions_for_hittest(
    obj: &ProteinObject,
    atom_overrides: &HashMap<usize, Vec2>,
    res_overrides: &HashMap<String, Vec2>,
    external: Option<&crate::io::interaction_import::ExternalInteractionData>,
) -> (HashMap<usize, Vec2>, HashMap<String, Vec2>) {
    let cur = obj.viz_state.selected_ligand_2d.unwrap_or(0);
    if cur >= obj.protein.ligands.len() {
        return (HashMap::new(), HashMap::new());
    }
    let ligand = &obj.protein.ligands[cur];

    // Ligand atoms
    let base = compute_ligand_layout(ligand);
    let atom_pos: HashMap<usize, Vec2> = base.into_iter().map(|(idx, p)| {
        (idx, atom_overrides.get(&idx).copied().unwrap_or(p))
    }).collect();

    // Collect unique residue labels
    let residues: Vec<String> = if let Some(ext) = external {
        // From JSON: use ext.interactions residue labels directly
        let mut set = HashSet::new();
        for i in &ext.interactions { set.insert(i.res_label.clone()); }
        let mut v: Vec<_> = set.into_iter().collect(); v.sort(); v
    } else {
        // From PDB interactions
        let relevant: Vec<_> = obj.interactions.iter()
            .filter(|i| i.ligand_index == cur).collect();
        let mut closest: HashMap<(String, u8), &crate::core::interactions::Interaction> =
            HashMap::new();
        for inter in &relevant {
            let rk = format!("{} {} ({})", inter.res_name, inter.res_seq, inter.chain_id);
            let tk: u8 = match inter.i_type {
                InteractionType::HydrogenBond => 0,
                InteractionType::Hydrophobic  => 1,
                InteractionType::SaltBridge   => 2,
                InteractionType::PiPiStacking => 3,
            };
            let e = closest.entry((rk, tk)).or_insert(inter);
            if inter.dist < e.dist { *e = inter; }
        }
        let mut set = HashSet::new();
        for i in closest.values() {
            set.insert(format!("{} {} ({})", i.res_name, i.res_seq, i.chain_id));
        }
        let mut v: Vec<_> = set.into_iter().collect(); v.sort(); v
    };

    // Build name→idx for anchor angle calculation
    let mut name_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, atom) in ligand.atoms.iter().enumerate() {
        name_to_idx.entry(atom.name.clone()).or_insert(i);
    }

    let lig_r = atom_pos.values().map(|p| p.length()).fold(0.0_f32, f32::max);
    let n_res = residues.len().max(1);
    let orbit_r = ((66.0 * n_res as f32) / (2.0 * PI)).max(lig_r + 130.0).max(150.0);

    // Sort by angle of connected ligand atom
    let mut res_sorted: Vec<(String, f32)> = residues.iter().map(|res| {
        let angle = if let Some(ext) = external {
            // Use first matching ligand atom from external interactions
            ext.interactions.iter()
                .filter(|i| &i.res_label == res)
                .filter_map(|i| {
                    find_atom_idx(ligand, &i.lig_atom)
                        .and_then(|idx| atom_pos.get(&idx))
                        .map(|p| p.y.atan2(p.x))
                })
                .next().unwrap_or(0.0)
        } else {
            let angles: Vec<f32> = obj.interactions.iter()
                .filter(|i| i.ligand_index == cur)
                .filter(|i| format!("{} {} ({})", i.res_name, i.res_seq, i.chain_id) == *res)
                .filter_map(|i| name_to_idx.get(&i.atom1_name)
                    .and_then(|&idx| atom_pos.get(&idx))
                    .map(|p| p.y.atan2(p.x)))
                .collect();
            if angles.is_empty() { 0.0 } else { angles.iter().sum::<f32>() / angles.len() as f32 }
        };
        (res.clone(), angle)
    }).collect();
    res_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let step = 2.0 * PI / n_res as f32;
    let anchor = res_sorted.first().map(|(_, a)| *a).unwrap_or(0.0);
    for (i, (_, a)) in res_sorted.iter_mut().enumerate() { *a = anchor + i as f32 * step; }

    let res_pos: HashMap<String, Vec2> = res_sorted.iter().map(|(name, angle)| {
        let base = Vec2::new(angle.cos() * orbit_r, angle.sin() * orbit_r);
        (name.clone(), res_overrides.get(name).copied().unwrap_or(base))
    }).collect();

    (atom_pos, res_pos)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main drawing entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn draw_2d_interaction_map_inline(
    painter: &egui::Painter,
    rect: Rect,
    obj: &ProteinObject,
    zoom: f32,
    pan: Vec2,
    atom_overrides: &HashMap<usize, Vec2>,
    res_overrides: &HashMap<String, Vec2>,
    external: Option<&crate::io::interaction_import::ExternalInteractionData>,
) {
    use crate::io::interaction_import::ExtInteractionType;

    let cur = obj.viz_state.selected_ligand_2d.unwrap_or(0);
    if cur >= obj.protein.ligands.len() { return; }
    let ligand = &obj.protein.ligands[cur];

    let relevant: Vec<_> = obj.interactions.iter()
        .filter(|i| i.ligand_index == cur).collect();

    let painter = painter.with_clip_rect(rect);
    let to_screen = |p: Vec2| -> Pos2 { rect.center() + p * zoom + pan };

    // Use compute_positions_for_hittest to get positions with overrides applied
    let (lig_pos, res_pos) = compute_positions_for_hittest(obj, atom_overrides, res_overrides, external);
    if lig_pos.is_empty() { return; }

    let mut name_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, atom) in ligand.atoms.iter().enumerate() {
        name_to_idx.entry(atom.name.clone()).or_insert(i);
    }

    // Filter interactions
    let mut closest: HashMap<(String, u8), &crate::core::interactions::Interaction> =
        HashMap::new();
    for inter in &relevant {
        let rk = format!("{} {} ({})", inter.res_name, inter.res_seq, inter.chain_id);
        let tk: u8 = match inter.i_type {
            InteractionType::HydrogenBond => 0,
            InteractionType::Hydrophobic  => 1,
            InteractionType::SaltBridge   => 2,
            InteractionType::PiPiStacking => 3,
        };
        let e = closest.entry((rk, tk)).or_insert(inter);
        if inter.dist < e.dist { *e = inter; }
    }
    let filtered: Vec<_> = closest.values().collect();

    // res_pos and lig_pos are already computed via compute_positions_for_hittest above

    // Draw bonds
    let bond_sw = (2.0 * zoom).clamp(0.8, 4.5);
    let bonds: Vec<(usize, usize)> = if !ligand.bonds.is_empty() {
        ligand.bonds.clone()
    } else {
        let mut p = Vec::new();
        for a in 0..ligand.atoms.len() {
            for b in (a+1)..ligand.atoms.len() {
                let pa = bevy::prelude::Vec3::new(
                    ligand.atoms[a].x as f32, ligand.atoms[a].y as f32, ligand.atoms[a].z as f32);
                let pb = bevy::prelude::Vec3::new(
                    ligand.atoms[b].x as f32, ligand.atoms[b].y as f32, ligand.atoms[b].z as f32);
                if pa.distance(pb) < 2.1 { p.push((a, b)); }
            }
        }
        p
    };
    for &(a, b) in &bonds {
        if let (Some(&pa), Some(&pb)) = (lig_pos.get(&a), lig_pos.get(&b)) {
            painter.line_segment([to_screen(pa), to_screen(pb)],
                Stroke::new(bond_sw, Color32::from_rgb(155, 165, 178)));
        }
    }

    // Draw interaction lines
    let int_sw = (2.0 * zoom).clamp(0.8, 3.5);

    if let Some(ext) = external {
        // ── External JSON interactions — show ALL entries ─────────────────
        for inter in &ext.interactions {
            let p_lig = match find_atom_idx(ligand, &inter.lig_atom)
                .and_then(|idx| lig_pos.get(&idx)) {
                Some(&p) => p,
                None => continue,
            };
            let p_res = match res_pos.get(&inter.res_label) {
                Some(&p) => p,
                None => continue,
            };

            let (col, dashed) = match inter.itype {
                ExtInteractionType::HydrogenBond      => (Color32::from_rgb(74, 95, 213),  true),
                ExtInteractionType::Hydrophobic       => (Color32::from_rgb(76, 175, 80),  true),
                ExtInteractionType::SaltBridge        => (Color32::from_rgb(255, 87, 34),  false),
                ExtInteractionType::PiStacking        => (Color32::from_rgb(156, 39, 176), false),
                ExtInteractionType::PiCation          => (Color32::from_rgb(200, 100, 255),false),
                ExtInteractionType::HalogenBond       => (Color32::from_rgb(0, 180, 180),  true),
                ExtInteractionType::MetalCoordination => (Color32::from_rgb(180, 130, 0),  false),
            };
            let sp1 = to_screen(p_lig);
            let sp2 = to_screen(p_res);
            let stk = Stroke::new(int_sw, col);
            if dashed {
                draw_dashed_line(&painter, sp1, sp2,
                    (7.0*zoom).max(3.0), (4.0*zoom).max(2.0), stk);
            } else {
                painter.line_segment([sp1, sp2], stk);
            }
            if inter.distance > 0.001 {
                let mid = Pos2::new((sp1.x+sp2.x)*0.5, (sp1.y+sp2.y)*0.5);
                let dir = (sp2-sp1).normalized();
                let perp = Vec2::new(-dir.y, dir.x) * (10.0*zoom).clamp(6.0, 20.0);
                painter.text(mid+perp, Align2::CENTER_CENTER,
                    format!("{:.1}Å", inter.distance),
                    FontId::proportional((10.0*zoom).clamp(7.0, 15.0)),
                    Color32::from_rgba_unmultiplied(108, 117, 125, 200));
            }
        }
    } else {
        // ── PDB-derived interactions ──────────────────────────────────────
        for inter in &filtered {
            let p_lig = match name_to_idx.get(&inter.atom1_name)
                .and_then(|&idx| lig_pos.get(&idx)) { Some(&p) => p, None => continue };
            let rk = format!("{} {} ({})", inter.res_name, inter.res_seq, inter.chain_id);
            let p_res = match res_pos.get(&rk) { Some(&p) => p, None => continue };

            let (col, dashed) = match inter.i_type {
                InteractionType::HydrogenBond => (Color32::from_rgb(74, 95, 213), true),
                InteractionType::Hydrophobic  => (Color32::from_rgb(76, 175, 80),  true),
                InteractionType::SaltBridge   => (Color32::from_rgb(255, 87, 34),  false),
                InteractionType::PiPiStacking => (Color32::from_rgb(156, 39, 176), false),
            };
            let sp1 = to_screen(p_lig);
            let sp2 = to_screen(p_res);
            let stk = Stroke::new(int_sw, col);
            if dashed {
                draw_dashed_line(&painter, sp1, sp2,
                    (7.0*zoom).max(3.0), (4.0*zoom).max(2.0), stk);
            } else {
                painter.line_segment([sp1, sp2], stk);
            }
            let mid = Pos2::new((sp1.x+sp2.x)*0.5, (sp1.y+sp2.y)*0.5);
            let dir = (sp2-sp1).normalized();
            let perp = Vec2::new(-dir.y, dir.x) * (10.0*zoom).clamp(6.0, 20.0);
            painter.text(mid+perp, Align2::CENTER_CENTER,
                format!("{:.1}Å", inter.dist),
                FontId::proportional((10.0*zoom).clamp(7.0, 15.0)),
                Color32::from_rgba_unmultiplied(108, 117, 125, 200));
        }
    }

    // Draw ligand atoms
    let c_r  = (4.5*zoom).clamp(2.0, 9.0);
    let h_r  = (12.0*zoom).clamp(5.0, 22.0);
    let a_sz = (11.0*zoom).clamp(6.0, 17.0);
    for (&idx, &upos) in &lig_pos {
        let el = ligand.atoms[idx].element.trim().to_uppercase();
        let sp = to_screen(upos);
        if el == "C" || el.is_empty() {
            painter.circle_filled(sp, c_r, Color32::from_rgb(150, 160, 172));
        } else {
            let (bg, fg) = element_colors(&el);
            painter.circle_filled(sp+Vec2::new(1.0,1.0), h_r,
                Color32::from_rgba_unmultiplied(0,0,0,14));
            painter.circle_filled(sp, h_r, bg);
            painter.circle_stroke(sp, h_r,
                Stroke::new(1.5_f32, Color32::from_rgb(200,210,220)));
            painter.text(sp, Align2::CENTER_CENTER,
                el.chars().take(2).collect::<String>(),
                FontId::proportional(a_sz), fg);
        }
    }

    // Draw residue nodes
    let nr  = (28.0*zoom).clamp(10.0, 50.0);
    let nf1 = (12.0*zoom).clamp(7.0, 19.0);
    let nf2 = (9.5*zoom).clamp(6.0, 15.0);
    let no  = (7.0*zoom).clamp(3.0, 13.0);
    for (name, &rpos) in &res_pos {
        let sp = to_screen(rpos);
        painter.circle_filled(sp+Vec2::new(2.0,2.0), nr,
            Color32::from_rgba_unmultiplied(0,0,0,12));
        painter.circle_filled(sp, nr, Color32::WHITE);
        painter.circle_stroke(sp, nr,
            Stroke::new(2.0_f32, Color32::from_rgb(210,220,235)));
        let pts: Vec<&str> = name.splitn(2, ' ').collect();
        let rname = pts.first().copied().unwrap_or(name.as_str());
        let rest  = if pts.len() > 1 { pts[1] } else { "" };
        painter.text(sp+Vec2::new(0.0,-no), Align2::CENTER_CENTER,
            rname, FontId::proportional(nf1), Color32::from_rgb(74,95,213));
        painter.text(sp+Vec2::new(0.0, no), Align2::CENTER_CENTER,
            rest,  FontId::proportional(nf2), Color32::from_rgb(108,117,125));
    }

    // Badge — show source file if external
    let bp = rect.left_top() + Vec2::new(12.0, 12.0);
    let lt = if let Some(ext) = external {
        format!("JSON: {}", ext.source_file)
    } else {
        format!("Ligand: {}", ligand.name)
    };
    let gl = painter.layout_no_wrap(lt.clone(),
        FontId::proportional(12.5), Color32::from_rgb(74,95,213));
    let br = egui::Rect::from_min_size(bp, gl.size() + Vec2::new(14.0,8.0));
    painter.rect_filled(br, 8.0, Color32::from_rgb(235,240,255));
    painter.rect_stroke(br, 8.0,
        Stroke::new(1.0_f32, Color32::from_rgb(200,215,245)));
    painter.text(bp+Vec2::new(7.0,4.0), Align2::LEFT_TOP, lt,
        FontId::proportional(12.5), Color32::from_rgb(74,95,213));

    draw_legend(&painter, Pos2::new(rect.left()+12.0, rect.bottom()-12.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn draw_dashed_line(painter: &egui::Painter, p1: Pos2, p2: Pos2,
    dash: f32, gap: f32, stroke: Stroke) {
    let total = p1.distance(p2);
    if total < 0.001 { return; }
    let dir = (p2-p1).normalized();
    let mut cur = 0.0_f32;
    while cur < total {
        let end = (cur+dash).min(total);
        painter.line_segment([p1+dir*cur, p1+dir*end], stroke);
        cur += dash+gap;
    }
}

fn element_colors(el: &str) -> (Color32, Color32) {
    match el {
        e if e.starts_with('O')                                             => (Color32::from_rgb(220,50,50),   Color32::WHITE),
        e if e.starts_with('N')                                             => (Color32::from_rgb(50,90,220),   Color32::WHITE),
        e if e.starts_with('S')                                             => (Color32::from_rgb(220,190,30),  Color32::BLACK),
        e if e.starts_with('P')                                             => (Color32::from_rgb(220,110,30),  Color32::WHITE),
        e if e=="FE"||e=="MG"||e=="ZN"||e=="CA"||e=="MN"||e=="CO"||e=="CU" => (Color32::from_rgb(140,90,180),  Color32::WHITE),
        e if e.starts_with('F') && e.len()==1                               => (Color32::from_rgb(80,200,80),   Color32::BLACK),
        e if e.starts_with('C') && e.len()>1                                => (Color32::from_rgb(140,90,180),  Color32::WHITE),
        _                                                                    => (Color32::from_rgb(120,120,130), Color32::WHITE),
    }
}

fn draw_legend(painter: &egui::Painter, bottom_left: Pos2) {
    let items: &[(Color32, bool, &str)] = &[
        (Color32::from_rgb(74,95,213),  true,  "Hydrogen Bond"),
        (Color32::from_rgb(76,175,80),  true,  "Hydrophobic"),
        (Color32::from_rgb(255,87,34),  false, "Salt Bridge"),
        (Color32::from_rgb(156,39,176), false, "π-π Stacking"),
    ];
    let row_h = 20.0_f32;
    let h = items.len() as f32 * row_h + 10.0;
    let lr = Rect::from_min_size(
        bottom_left + Vec2::new(0.0, -h-8.0), Vec2::new(178.0, h+8.0));
    painter.rect_filled(lr, 8.0, Color32::WHITE);
    painter.rect_stroke(lr, 8.0,
        Stroke::new(1.0_f32, Color32::from_rgb(228,231,236)));
    for (i, (col, dashed, label)) in items.iter().enumerate() {
        let y  = lr.top() + 9.0 + i as f32 * row_h;
        let x0 = lr.left() + 10.0;
        let x1 = x0 + 24.0;
        let my = y + row_h*0.5 - 2.0;
        let stk = Stroke::new(2.0_f32, *col);
        if *dashed {
            draw_dashed_line(painter,
                Pos2::new(x0,my), Pos2::new(x1,my), 4.0, 3.0, stk);
        } else {
            painter.line_segment([Pos2::new(x0,my), Pos2::new(x1,my)], stk);
        }
        painter.text(Pos2::new(x1+7.0, my), Align2::LEFT_CENTER,
            *label, FontId::proportional(11.5), Color32::from_rgb(30,32,38));
    }
}
