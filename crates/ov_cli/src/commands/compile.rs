use std::time::Duration;

use tokio::time::Instant;

use serde_json::Value;

use crate::client::{CompileAccepted, CompileResult, HttpClient};
use crate::error::{Error, Result};
use crate::output::{OutputFormat, output_success};

pub async fn run(
    client: &HttpClient,
    from_uris: Vec<String>,
    to: String,
    skill: String,
    reason: Option<String>,
    wait: bool,
    timeout: Option<f64>,
    runtime_timeout: Option<f64>,
    output_format: OutputFormat,
    compact: bool,
) -> Result<()> {
    let sources = normalize_sources(from_uris)?;
    if timeout.is_some_and(|seconds| !crate::config::timeout_is_valid(seconds)) {
        return Err(Error::Client(
            "--timeout must be a positive finite number of seconds".into(),
        ));
    }
    let reason = reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let accepted = client
        .create_compile(&sources, to.trim(), skill.trim(), reason, runtime_timeout)
        .await?;
    if !wait {
        render_accepted(&accepted, to.trim(), output_format, compact);
        return Ok(());
    }

    let deadline = timeout.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds));
    let mut polling = Duration::from_millis(500);
    loop {
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return Err(Error::Client(format!(
                "Timed out waiting for compile task {}; the task is still running",
                accepted.task_id
            )));
        }
        let task = client.get_task(&accepted.task_id).await?;
        let status = task
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match status {
            "completed" => {
                render_completed(&task, output_format, compact)?;
                return Ok(());
            }
            "failed" => {
                let error = task
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("UNKNOWN: Compile task failed");
                let (code, message) = error.split_once(": ").unwrap_or(("UNKNOWN", error));
                return Err(Error::api_response(Some(code.into()), message, None, 500));
            }
            "cancelled" => {
                return Err(Error::api_response(
                    Some("CANCELLED".into()),
                    format!("Compile task {} was cancelled", accepted.task_id),
                    None,
                    409,
                ));
            }
            _ => {}
        }

        let sleep_for = deadline
            .map(|value| polling.min(value.saturating_duration_since(Instant::now())))
            .unwrap_or(polling);
        tokio::time::sleep(sleep_for).await;
        polling = (polling * 2).min(Duration::from_secs(2));
    }
}

fn normalize_sources(values: Vec<String>) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for value in values {
        for item in value.split(',') {
            let item = item.trim();
            if item.is_empty() {
                return Err(Error::Client("--from contains an empty directory".into()));
            }
            if !result.iter().any(|existing| existing == item) {
                result.push(item.to_string());
            }
        }
    }
    if result.is_empty() {
        return Err(Error::Client(
            "at least one --from directory is required".into(),
        ));
    }
    Ok(result)
}

fn render_accepted(
    value: &CompileAccepted,
    requested_to: &str,
    format: OutputFormat,
    compact: bool,
) {
    if matches!(format, OutputFormat::Json) {
        output_success(value, format, compact);
    } else {
        println!("task_id: {}", value.task_id);
        println!("status: {}", value.status);
        println!("to: {}", value.to.as_deref().unwrap_or(requested_to));
    }
}

fn render_completed(value: &Value, format: OutputFormat, compact: bool) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        output_success(value, format, compact);
        return Ok(());
    }
    let Some(result_value) = value.get("result").cloned() else {
        println!("status: completed");
        return Ok(());
    };
    let Ok(result) = serde_json::from_value::<CompileResult>(result_value) else {
        println!("status: completed");
        return Ok(());
    };
    println!("to: {}", result.to);
    println!("created: {}", result.created.len());
    println!("updated: {}", result.updated.len());
    println!("unchanged: {}", result.unchanged.len());
    println!("page_count: {}", result.page_count);
    println!("link_count: {}", result.link_count);
    for warning in result.warnings {
        eprintln!("warning: {warning}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::normalize_sources;

    #[test]
    fn expands_comma_separated_and_repeated_sources_stably() {
        let result = normalize_sources(vec![
            "viking://resources/a,viking://resources/b".into(),
            "viking://resources/a".into(),
        ])
        .expect("sources should be valid");
        assert_eq!(result, vec!["viking://resources/a", "viking://resources/b"]);
    }

    #[test]
    fn rejects_empty_source_items() {
        assert!(normalize_sources(vec!["viking://resources/a,".into()]).is_err());
    }
}
