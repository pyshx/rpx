/// Parse a raw SSE line into its data payload, if any.
/// Returns None for empty lines, comments, and the [DONE] sentinel.
pub fn parse_sse_line(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    let data = line.strip_prefix("data:")?;
    let data = data.trim();
    if data == "[DONE]" {
        return None;
    }
    Some(data.to_string())
}

/// Check if a line is the SSE termination sentinel.
pub fn is_done(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed == "data: [DONE]" || trimmed == "data:[DONE]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_data_line() {
        let result = parse_sse_line(r#"data: {"id":"chatcmpl-1","choices":[]}"#);
        assert_eq!(
            result,
            Some(r#"{"id":"chatcmpl-1","choices":[]}"#.to_string())
        );
    }

    #[test]
    fn parse_data_line_no_space() {
        let result = parse_sse_line(r#"data:{"id":"chatcmpl-1"}"#);
        assert_eq!(result, Some(r#"{"id":"chatcmpl-1"}"#.to_string()));
    }

    #[test]
    fn skip_empty_line() {
        assert_eq!(parse_sse_line(""), None);
        assert_eq!(parse_sse_line("  "), None);
    }

    #[test]
    fn skip_comment() {
        assert_eq!(parse_sse_line(": this is a comment"), None);
    }

    #[test]
    fn skip_done_sentinel() {
        assert_eq!(parse_sse_line("data: [DONE]"), None);
        assert_eq!(parse_sse_line("data:[DONE]"), None);
    }

    #[test]
    fn skip_non_data_line() {
        assert_eq!(parse_sse_line("event: message"), None);
    }

    #[test]
    fn is_done_detection() {
        assert!(is_done("data: [DONE]"));
        assert!(is_done("data:[DONE]"));
        assert!(is_done("  data: [DONE]  "));
        assert!(!is_done("data: {\"id\": 1}"));
        assert!(!is_done(""));
    }
}
