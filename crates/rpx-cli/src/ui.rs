#[cfg(feature = "tui")]
use lipgloss::{Border, Style};

// Brand colors
#[cfg(feature = "tui")]
const ACCENT: &str = "#7C3AED";
#[cfg(feature = "tui")]
const SUCCESS: &str = "#22C55E";
#[cfg(feature = "tui")]
const ERROR_COLOR: &str = "#EF4444";
#[cfg(feature = "tui")]
const DIM: &str = "#6B7280";
#[cfg(feature = "tui")]
const LABEL: &str = "#A78BFA";

#[cfg(feature = "tui")]
pub fn title(text: &str) -> String {
    Style::new().bold().foreground(ACCENT).render(text)
}
#[cfg(not(feature = "tui"))]
pub fn title(text: &str) -> String {
    text.to_string()
}

#[cfg(feature = "tui")]
pub fn success(text: &str) -> String {
    Style::new().bold().foreground(SUCCESS).render(text)
}
#[cfg(not(feature = "tui"))]
pub fn success(text: &str) -> String {
    text.to_string()
}

#[cfg(feature = "tui")]
pub fn error(text: &str) -> String {
    Style::new().bold().foreground(ERROR_COLOR).render(text)
}
#[cfg(not(feature = "tui"))]
pub fn error(text: &str) -> String {
    text.to_string()
}

#[cfg(feature = "tui")]
pub fn dim(text: &str) -> String {
    Style::new().foreground(DIM).render(text)
}
#[cfg(not(feature = "tui"))]
pub fn dim(text: &str) -> String {
    text.to_string()
}

#[cfg(feature = "tui")]
pub fn label(text: &str) -> String {
    Style::new().foreground(LABEL).render(text)
}
#[cfg(not(feature = "tui"))]
pub fn label(text: &str) -> String {
    text.to_string()
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

#[cfg(feature = "tui")]
pub fn endpoint_card(info: &EndpointCardInfo<'_>) -> String {
    let EndpointCardInfo { name, id, provider, gpu, backend, status, vram, cost_per_hour } = info;
    let border_style = Style::new()
        .border(Border::rounded())
        .border_foreground(ACCENT)
        .padding((1, 2));

    let content = [
        title(name),
        String::new(),
        key_value("ID", id),
        key_value("Provider", provider),
        key_value("GPU", gpu),
        key_value("Backend", backend),
        key_value("Status", status),
        key_value("VRAM", &format!("{vram:.1} GB")),
        key_value("Cost", &format!("${cost_per_hour:.2}/hr")),
        String::new(),
        dim("  rpx proxy ") + name,
    ]
    .join("\n");

    border_style.render(&content)
}

#[cfg(not(feature = "tui"))]
pub fn endpoint_card(info: &EndpointCardInfo<'_>) -> String {
    format!(
        "{} ({}) — {} on {} via {}",
        info.name, info.id, info.status, info.gpu, info.provider
    )
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
