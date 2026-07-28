mod core;
mod graphics;
mod io;
mod ui;

use bevy::prelude::*;
use graphics::GraphicsPlugin;
use io::IoPlugin;
use ui::UiPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(IoPlugin)
        .add_plugins(GraphicsPlugin)
        .add_plugins(UiPlugin)
        .run();
}
