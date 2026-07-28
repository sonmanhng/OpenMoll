pub mod parser;
pub mod interaction_import;
pub mod xtc_parser;

use crate::core::protein::Protein;
use bevy::prelude::*;
use std::path::PathBuf;

#[derive(Event)]
pub struct LoadFileEvent(pub PathBuf);

#[derive(Event)]
pub struct ProteinLoadedEvent(pub Protein);

/// Fired when an XTC trajectory has been parsed in a background task.
#[derive(Event)]
pub struct XtcLoadedEvent(pub xtc_parser::XtcTrajectory);

pub struct IoPlugin;

impl Plugin for IoPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<LoadFileEvent>()
            .add_event::<ProteinLoadedEvent>()
            .add_event::<XtcLoadedEvent>()
            .add_systems(Update, handle_file_load);
    }
}

fn handle_file_load(
    mut load_events: EventReader<LoadFileEvent>,
    mut loaded_events: EventWriter<ProteinLoadedEvent>,
) {
    for event in load_events.read() {
        if let Some(path_str) = event.0.to_str() {
            println!("Loading PDB: {}", path_str);
            match parser::load_pdb(path_str) {
                Ok(protein) => {
                    println!(
                        "Successfully loaded protein with {} chains",
                        protein.chains.len()
                    );
                    loaded_events.send(ProteinLoadedEvent(protein));
                }
                Err(e) => {
                    eprintln!("Failed to load PDB: {:?}", e);
                }
            }
        }
    }
}
pub mod export;
