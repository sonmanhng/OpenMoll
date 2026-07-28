/// Gets a basic partial charge map for amino acid residues.
/// This is a simplified model for real-time Coulombic potential visualization.
/// Asp/Glu = -1, Arg/Lys = +1, His = +0.5.
pub fn get_residue_charge(res_name: &str) -> f32 {
    match res_name.to_uppercase().as_str() {
        "ASP" | "GLU" => -1.0,
        "ARG" | "LYS" => 1.0,
        "HIS" => 0.5,
        _ => 0.0,
    }
}

/// Represents a point charge in space.
pub struct PointCharge {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub q: f32,
}

/// Calculates the Coulombic potential at a given point `(x, y, z)` due to a set of charges.
/// The potential V = sum(q_i / r_i) where r_i is the distance to the charge.
pub fn calculate_coulombic_potential(x: f32, y: f32, z: f32, charges: &[PointCharge]) -> f32 {
    let mut potential = 0.0;
    for charge in charges {
        let dx = x - charge.x;
        let dy = y - charge.y;
        let dz = z - charge.z;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        // Avoid division by zero by adding a small epsilon (or limiting min distance)
        let dist = dist_sq.sqrt().max(1.0); // Clamp minimum distance to 1.0 A
        potential += charge.q / dist;
    }
    // Scale potential for visualization purposes
    potential * 10.0
}
