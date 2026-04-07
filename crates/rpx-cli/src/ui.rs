pub fn title(text: &str) -> String {
    format!("\x1b[1;35m{text}\x1b[0m")
}

pub fn success(text: &str) -> String {
    format!("\x1b[1;32m{text}\x1b[0m")
}

pub fn error(text: &str) -> String {
    format!("\x1b[1;31m{text}\x1b[0m")
}

pub fn dim(text: &str) -> String {
    format!("\x1b[2m{text}\x1b[0m")
}

pub fn label(text: &str) -> String {
    format!("\x1b[35m{text}\x1b[0m")
}

pub fn key_value(key: &str, value: &str) -> String {
    format!("  {} {}", label(&format!("{key}:")), value)
}

pub struct EndpointCardInfo<'a> {
    pub name: &'a str,
    pub id: &'a str,
    pub provider: &'a str,
    pub gpu: &'a str,
    pub backend: &'a str,
    pub status: &'a str,
    pub vram: f64,
    pub cost_per_hour: f64,
}

pub fn endpoint_card(info: &EndpointCardInfo<'_>) -> String {
    [
        title(info.name),
        key_value("ID", info.id),
        key_value("Provider", info.provider),
        key_value("GPU", info.gpu),
        key_value("Backend", info.backend),
        key_value("Status", info.status),
        key_value("VRAM", &format!("{:.1} GB", info.vram)),
        key_value("Cost", &format!("${:.2}/hr", info.cost_per_hour)),
    ]
    .join("\n")
}

pub fn table_header(columns: &[(&str, usize)]) -> String {
    let header: String = columns
        .iter()
        .map(|(name, width)| format!("{name:<width$}"))
        .collect::<Vec<_>>()
        .join(" ");
    let separator = "-".repeat(columns.iter().map(|(_, w)| w + 1).sum::<usize>());
    format!("{header}\n{separator}")
}

pub fn table_row(values: &[(&str, usize)]) -> String {
    values
        .iter()
        .map(|(val, width)| format!("{val:<width$}"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub struct InlineSpinner {
    frame: usize,
    message: String,
}

impl InlineSpinner {
    pub fn new(message: &str) -> Self {
        Self {
            frame: 0,
            message: message.to_string(),
        }
    }

    pub fn tick(&mut self) {
        let frame_char = SPINNER_FRAMES[self.frame % SPINNER_FRAMES.len()];
        eprint!("\r{frame_char} {}", self.message);
        self.frame += 1;
    }

    pub fn finish(&self, final_message: &str) {
        let check = success("✓");
        eprintln!("\r{check} {final_message}");
    }

    pub fn fail(&self, final_message: &str) {
        let x = error("✗");
        eprintln!("\r{x} {final_message}");
    }
}
