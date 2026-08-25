//! Queue Backend Abstraction
//!
//! This module provides a pluggable backend system for QueueFS, allowing different
//! storage implementations (memory, SQLite, etc.) while maintaining a consistent interface.

use crate::core::errors::{Error, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, types::ValueRef, Connection, Row};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;
use std::time::SystemTime;
use std::time::{Duration, UNIX_EPOCH};
use uuid::Uuid;

/// A message in the queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique identifier for the message
    pub id: String,
    /// Message data
    pub data: Vec<u8>,
    /// Timestamp when the message was enqueued
    pub timestamp: SystemTime,
    /// Dispatch key used to serialize keyed batches for the same logical target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch_key: Option<String>,
    /// Signature that defines whether adjacent keyed contributions can merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_signature: Option<String>,
}

impl Message {
    /// Create a new message with the given data
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            data,
            timestamp: SystemTime::now(),
            dispatch_key: None,
            merge_signature: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StoredMessage {
    id: String,
    #[serde(deserialize_with = "deserialize_stored_message_data")]
    data: Vec<u8>,
    #[serde(default)]
    timestamp: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dispatch_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    merge_signature: Option<String>,
}

impl StoredMessage {
    pub(super) fn from_message(msg: &Message) -> Self {
        Self {
            id: msg.id.clone(),
            data: msg.data.clone(),
            // Prefer unix seconds for compatibility with older queue.db producers.
            timestamp: Some(serde_json::Value::Number(unix_secs(msg.timestamp).into())),
            dispatch_key: msg.dispatch_key.clone(),
            merge_signature: msg.merge_signature.clone(),
        }
    }

    pub(super) fn into_message(self) -> Message {
        Message {
            id: self.id,
            data: self.data,
            timestamp: parse_stored_timestamp(self.timestamp),
            dispatch_key: self.dispatch_key,
            merge_signature: self.merge_signature,
        }
    }
}

pub const KEYED_BATCH_SCHEMA_VERSION: u8 = 1;
pub const MAX_KEYED_BATCH_CONTRIBUTIONS: usize = 1024;
pub const MAX_KEYED_BATCH_BYTES: usize = 8_388_608;
pub const MAX_DISPATCH_KEY_BYTES: usize = 512;
pub const MAX_MERGE_SIGNATURE_BYTES: usize = 128;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyedEnqueueRequest {
    pub dispatch_key: String,
    pub merge_signature: String,
    pub data: String,
}

impl KeyedEnqueueRequest {
    pub(super) fn validate(&self) -> Result<()> {
        validate_route_field("dispatch_key", &self.dispatch_key, MAX_DISPATCH_KEY_BYTES)?;
        validate_route_field(
            "merge_signature",
            &self.merge_signature,
            MAX_MERGE_SIGNATURE_BYTES,
        )
    }
}

fn validate_route_field(name: &str, value: &str, max_bytes: usize) -> Result<()> {
    if value.is_empty() {
        return Err(Error::InvalidOperation(format!("{name} must not be empty")));
    }
    if value.len() > max_bytes {
        return Err(Error::InvalidOperation(format!(
            "{name} must be at most {max_bytes} UTF-8 bytes"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyedBatchCreateReason {
    NoPending,
    Signature,
    Count,
    Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyedEnqueueOutcome {
    Created {
        message_id: String,
        reason: KeyedBatchCreateReason,
    },
    Merged {
        message_id: String,
        contribution_count: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct KeyedBatchBody {
    pub(super) schema_version: u8,
    pub(super) dispatch_key: String,
    pub(super) merge_signature: String,
    pub(super) contributions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct KeyedBatchEnvelope {
    pub(super) _queuefs_keyed_batch: KeyedBatchBody,
}

fn serialize_keyed_batch(body: &KeyedBatchBody) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&KeyedBatchEnvelope {
        _queuefs_keyed_batch: body.clone(),
    })?)
}

fn deserialize_keyed_batch(data: &[u8]) -> Result<KeyedBatchBody> {
    let body = serde_json::from_slice::<KeyedBatchEnvelope>(data)
        .map_err(|error| Error::Serialization(error.to_string()))?
        ._queuefs_keyed_batch;
    if body.schema_version != KEYED_BATCH_SCHEMA_VERSION {
        return Err(Error::InvalidOperation(format!(
            "unsupported keyed batch schema version {}",
            body.schema_version
        )));
    }
    Ok(body)
}

fn deserialize_stored_message_data<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        // Legacy queue.db rows stored message bytes as a JSON string.
        serde_json::Value::String(data) => Ok(data.into_bytes()),
        // New rows keep arbitrary message bytes lossless as a JSON byte array.
        serde_json::Value::Array(bytes) => bytes
            .into_iter()
            .map(|byte| {
                byte.as_u64()
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| {
                        serde::de::Error::custom(
                            "stored queue message data byte must be an integer in 0..=255",
                        )
                    })
            })
            .collect(),
        other => Err(serde::de::Error::custom(format!(
            "stored queue message data must be a string or byte array, got {other}"
        ))),
    }
}

fn parse_stored_timestamp(raw: Option<serde_json::Value>) -> SystemTime {
    match raw {
        Some(serde_json::Value::String(ts)) => DateTime::parse_from_rfc3339(&ts)
            .map(|dt| {
                let secs = dt.timestamp();
                let secs_u64 = if secs <= 0 { 0 } else { secs as u64 };
                UNIX_EPOCH + Duration::from_secs(secs_u64)
            })
            .unwrap_or_else(|_| SystemTime::now()),
        Some(serde_json::Value::Number(num)) => num
            .as_i64()
            .map(|secs| {
                let secs_u64 = if secs <= 0 { 0 } else { secs as u64 };
                UNIX_EPOCH + Duration::from_secs(secs_u64)
            })
            .unwrap_or_else(SystemTime::now),
        _ => SystemTime::now(),
    }
}

fn unix_secs(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn read_sqlite_text(row: &Row<'_>, idx: usize) -> rusqlite::Result<String> {
    match row.get_ref(idx)? {
        ValueRef::Text(bytes) => Ok(String::from_utf8_lossy(bytes).into_owned()),
        ValueRef::Blob(bytes) => Ok(String::from_utf8_lossy(bytes).into_owned()),
        ValueRef::Null => Ok(String::new()),
        other => Err(rusqlite::Error::InvalidColumnType(
            idx,
            row.as_ref()
                .column_name(idx)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| format!("column_{idx}")),
            other.data_type(),
        )),
    }
}

/// Queue backend trait for pluggable storage implementations
pub trait QueueBackend: Send + Sync {
    /// Create a new queue with the given name
    fn create_queue(&mut self, name: &str) -> Result<()>;

    /// Remove a queue and all its messages
    fn remove_queue(&mut self, name: &str) -> Result<()>;

    /// Check if a queue exists
    fn queue_exists(&self, name: &str) -> bool;

    /// List all queues with the given prefix
    /// If prefix is empty, returns all queues
    fn list_queues(&self, prefix: &str) -> Vec<String>;

    /// Add a message to the queue
    fn enqueue(&mut self, queue_name: &str, msg: Message) -> Result<()>;

    /// Add a contribution to the pending batch for a dispatch key, or create one.
    fn enqueue_keyed(
        &mut self,
        _queue_name: &str,
        _request: KeyedEnqueueRequest,
    ) -> Result<KeyedEnqueueOutcome> {
        Err(Error::InvalidOperation(
            "keyed enqueue is not supported by this queue backend".to_string(),
        ))
    }

    /// Remove and return the first message from the queue
    fn dequeue(&mut self, queue_name: &str) -> Result<Option<Message>>;

    /// View the first message without removing it
    fn peek(&self, queue_name: &str) -> Result<Option<Message>>;

    /// Get the number of messages in the queue
    fn size(&self, queue_name: &str) -> Result<usize>;

    /// List every unacknowledged message without changing queue state
    fn list_unacked(&self, queue_name: &str) -> Result<Vec<Message>>;

    /// Clear all messages from the queue
    fn clear(&mut self, queue_name: &str) -> Result<()>;

    /// Get the last enqueue time for the queue
    fn get_last_enqueue_time(&self, queue_name: &str) -> Result<SystemTime>;

    /// Acknowledge (delete) a message by ID
    fn ack(&mut self, queue_name: &str, msg_id: &str) -> Result<bool>;

    /// Return a processing message to the front of the pending queue.
    fn requeue(&mut self, _queue_name: &str, _msg_id: &str) -> Result<bool> {
        Err(Error::InvalidOperation(
            "requeue is not supported by this queue backend".to_string(),
        ))
    }
}

/// Options for SQLite queue backend.
#[derive(Debug, Clone, Copy)]
pub struct SQLiteQueueOptions {
    pub recover_stale_sec: i64,
    pub busy_timeout_ms: u64,
}

impl Default for SQLiteQueueOptions {
    fn default() -> Self {
        Self {
            recover_stale_sec: 0,
            busy_timeout_ms: 5_000,
        }
    }
}

/// A single queue with its messages
struct Queue {
    pending: VecDeque<String>,
    messages: HashMap<String, Message>,
    processing: HashSet<String>,
    active_keys: HashMap<String, String>,
    pending_tails: HashMap<String, String>,
    last_enqueue_time: SystemTime,
}

impl Queue {
    fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            messages: HashMap::new(),
            processing: HashSet::new(),
            active_keys: HashMap::new(),
            pending_tails: HashMap::new(),
            last_enqueue_time: SystemTime::UNIX_EPOCH,
        }
    }
}

fn create_keyed_batch(
    queue: &mut Queue,
    request: &KeyedEnqueueRequest,
    reason: KeyedBatchCreateReason,
) -> Result<KeyedEnqueueOutcome> {
    let body = KeyedBatchBody {
        schema_version: KEYED_BATCH_SCHEMA_VERSION,
        dispatch_key: request.dispatch_key.clone(),
        merge_signature: request.merge_signature.clone(),
        contributions: vec![request.data.clone()],
    };
    let data = serialize_keyed_batch(&body)?;
    if data.len() > MAX_KEYED_BATCH_BYTES {
        return Err(Error::InvalidOperation(format!(
            "keyed batch payload exceeds {MAX_KEYED_BATCH_BYTES} bytes"
        )));
    }
    let mut message = Message::new(data);
    message.dispatch_key = Some(request.dispatch_key.clone());
    message.merge_signature = Some(request.merge_signature.clone());
    let message_id = message.id.clone();
    queue.messages.insert(message_id.clone(), message);
    queue.pending.push_back(message_id.clone());
    queue
        .pending_tails
        .insert(request.dispatch_key.clone(), message_id.clone());
    Ok(KeyedEnqueueOutcome::Created { message_id, reason })
}

/// In-memory queue backend using HashMap
pub struct MemoryBackend {
    queues: HashMap<String, Queue>,
}

impl MemoryBackend {
    /// Create a new memory backend
    pub fn new() -> Self {
        Self {
            queues: HashMap::new(),
        }
    }
}

impl QueueBackend for MemoryBackend {
    fn create_queue(&mut self, name: &str) -> Result<()> {
        if self.queues.contains_key(name) {
            return Err(Error::AlreadyExists(format!(
                "queue '{}' already exists",
                name
            )));
        }
        self.queues.insert(name.to_string(), Queue::new());
        Ok(())
    }

    fn remove_queue(&mut self, name: &str) -> Result<()> {
        if self.queues.remove(name).is_none() {
            return Err(Error::NotFound(format!("queue '{}' not found", name)));
        }
        Ok(())
    }

    fn queue_exists(&self, name: &str) -> bool {
        self.queues.contains_key(name)
    }

    fn list_queues(&self, prefix: &str) -> Vec<String> {
        if prefix.is_empty() {
            self.queues.keys().cloned().collect()
        } else {
            self.queues
                .keys()
                .filter(|name| name.starts_with(prefix))
                .cloned()
                .collect()
        }
    }

    fn enqueue(&mut self, queue_name: &str, msg: Message) -> Result<()> {
        let queue = self
            .queues
            .get_mut(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        queue.last_enqueue_time = SystemTime::now();
        let msg_id = msg.id.clone();
        queue.messages.insert(msg_id.clone(), msg);
        queue.pending.push_back(msg_id);
        Ok(())
    }

    fn enqueue_keyed(
        &mut self,
        queue_name: &str,
        request: KeyedEnqueueRequest,
    ) -> Result<KeyedEnqueueOutcome> {
        request.validate()?;
        let queue = self
            .queues
            .get_mut(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        let Some(tail_id) = queue.pending_tails.get(&request.dispatch_key).cloned() else {
            let outcome = create_keyed_batch(queue, &request, KeyedBatchCreateReason::NoPending)?;
            queue.last_enqueue_time = SystemTime::now();
            return Ok(outcome);
        };
        let tail = queue.messages.get_mut(&tail_id).ok_or_else(|| {
            Error::internal(format!("keyed pending tail '{}' is missing", tail_id))
        })?;
        let mut body = deserialize_keyed_batch(&tail.data)?;
        let reason = if body.merge_signature != request.merge_signature {
            Some(KeyedBatchCreateReason::Signature)
        } else if body.contributions.len() >= MAX_KEYED_BATCH_CONTRIBUTIONS {
            Some(KeyedBatchCreateReason::Count)
        } else {
            body.contributions.push(request.data.clone());
            let candidate = serialize_keyed_batch(&body)?;
            if candidate.len() > MAX_KEYED_BATCH_BYTES {
                Some(KeyedBatchCreateReason::Bytes)
            } else {
                tail.data = candidate;
                queue.last_enqueue_time = SystemTime::now();
                return Ok(KeyedEnqueueOutcome::Merged {
                    message_id: tail_id,
                    contribution_count: body.contributions.len(),
                });
            }
        };

        let outcome = create_keyed_batch(
            queue,
            &request,
            reason.expect("keyed batch reason is assigned"),
        )?;
        queue.last_enqueue_time = SystemTime::now();
        Ok(outcome)
    }

    fn dequeue(&mut self, queue_name: &str) -> Result<Option<Message>> {
        let queue = self
            .queues
            .get_mut(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        let position = queue.pending.iter().position(|message_id| {
            let message = queue
                .messages
                .get(message_id)
                .expect("pending queue message payload is present");
            message
                .dispatch_key
                .as_ref()
                .is_none_or(|key| !queue.active_keys.contains_key(key))
        });
        let Some(position) = position else {
            return Ok(None);
        };
        let message_id = queue
            .pending
            .remove(position)
            .expect("dispatchable pending queue message is present");
        let message = queue
            .messages
            .get(&message_id)
            .cloned()
            .expect("pending queue message payload is present");
        queue.processing.insert(message_id.clone());
        if let Some(key) = &message.dispatch_key {
            queue.active_keys.insert(key.clone(), message_id.clone());
            if queue.pending_tails.get(key) == Some(&message_id) {
                queue.pending_tails.remove(key);
            }
        }
        Ok(Some(message))
    }

    fn peek(&self, queue_name: &str) -> Result<Option<Message>> {
        let queue = self
            .queues
            .get(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        let Some(message_id) = queue.pending.iter().find(|message_id| {
            let message = queue
                .messages
                .get(*message_id)
                .expect("pending queue message payload is present");
            message
                .dispatch_key
                .as_ref()
                .is_none_or(|key| !queue.active_keys.contains_key(key))
        }) else {
            return Ok(None);
        };
        Ok(queue.messages.get(message_id).cloned())
    }

    fn size(&self, queue_name: &str) -> Result<usize> {
        let queue = self
            .queues
            .get(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        Ok(queue.pending.len())
    }

    fn list_unacked(&self, queue_name: &str) -> Result<Vec<Message>> {
        let queue = self
            .queues
            .get(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;
        Ok(queue
            .pending
            .iter()
            .chain(queue.processing.iter())
            .filter_map(|message_id| queue.messages.get(message_id).cloned())
            .collect())
    }

    fn clear(&mut self, queue_name: &str) -> Result<()> {
        let queue = self
            .queues
            .get_mut(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        queue.pending.clear();
        queue.messages.clear();
        queue.processing.clear();
        queue.active_keys.clear();
        queue.pending_tails.clear();
        Ok(())
    }

    fn get_last_enqueue_time(&self, queue_name: &str) -> Result<SystemTime> {
        let queue = self
            .queues
            .get(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        Ok(queue.last_enqueue_time)
    }

    fn ack(&mut self, queue_name: &str, msg_id: &str) -> Result<bool> {
        let queue = self
            .queues
            .get_mut(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;

        if !queue.processing.contains(msg_id) {
            return Ok(false);
        }
        let message = queue.messages.get(msg_id).cloned().ok_or_else(|| {
            Error::internal(format!("processing queue message '{}' is missing", msg_id))
        })?;
        queue.processing.remove(msg_id);
        queue.messages.remove(msg_id);
        if let Some(key) = message.dispatch_key {
            if queue.active_keys.get(&key) == Some(&message.id) {
                queue.active_keys.remove(&key);
            }
        }
        Ok(true)
    }

    fn requeue(&mut self, queue_name: &str, msg_id: &str) -> Result<bool> {
        let queue = self
            .queues
            .get_mut(queue_name)
            .ok_or_else(|| Error::NotFound(format!("queue '{}' not found", queue_name)))?;
        if !queue.processing.contains(msg_id) {
            return Ok(false);
        }
        let message = queue.messages.get(msg_id).cloned().ok_or_else(|| {
            Error::internal(format!("processing queue message '{}' is missing", msg_id))
        })?;
        queue.processing.remove(msg_id);
        if let Some(key) = message.dispatch_key {
            if queue.active_keys.get(&key) == Some(&message.id) {
                queue.active_keys.remove(&key);
            }
            queue
                .pending_tails
                .entry(key)
                .or_insert_with(|| msg_id.to_string());
        }
        queue.pending.push_front(msg_id.to_string());
        Ok(true)
    }
}

/// SQLite queue backend with at-least-once delivery semantics.
pub struct SQLiteQueueBackend {
    conn: Mutex<Connection>,
}

impl SQLiteQueueBackend {
    pub fn open(db_path: &str, options: SQLiteQueueOptions) -> Result<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| Error::internal(format!("sqlite connection error: {}", e)))?;

        if options.busy_timeout_ms > 0 {
            conn.busy_timeout(Duration::from_millis(options.busy_timeout_ms))
                .map_err(|e| Error::internal(format!("sqlite busy_timeout error: {}", e)))?;
        }

        if Self::is_new_database(&conn)? {
            conn.execute_batch(
                r#"
                PRAGMA auto_vacuum=FULL;
                VACUUM;
                "#,
            )
            .map_err(|e| Error::internal(format!("sqlite auto_vacuum pragma error: {}", e)))?;
        }

        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            "#,
        )
        .map_err(|e| Error::internal(format!("sqlite pragma error: {}", e)))?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS queue_metadata (
                queue_name TEXT PRIMARY KEY,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                last_updated INTEGER DEFAULT (strftime('%s', 'now'))
            );

            CREATE TABLE IF NOT EXISTS queue_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_name TEXT NOT NULL,
                message_id TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                processing_started_at INTEGER,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            );

            "#,
        )
        .map_err(|e| Error::internal(format!("sqlite schema init error: {}", e)))?;

        let backend = Self {
            conn: Mutex::new(conn),
        };
        backend.run_migrations()?;
        backend.recover_stale(options.recover_stale_sec)?;
        Ok(backend)
    }

    fn is_new_database(conn: &Connection) -> Result<bool> {
        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| Error::internal(format!("sqlite schema probe error: {}", e)))?;
        Ok(table_count == 0)
    }

    fn run_migrations(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        for stmt in [
            "ALTER TABLE queue_messages ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'",
            "ALTER TABLE queue_messages ADD COLUMN processing_started_at INTEGER",
            "CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_messages(queue_name, status, id)",
            "CREATE INDEX IF NOT EXISTS idx_queue_message_id ON queue_messages(queue_name, message_id)",
        ] {
            if let Err(err) = conn.execute(stmt, []) {
                let text = err.to_string();
                if !text.contains("duplicate column name") && !text.contains("already exists") {
                    return Err(Error::internal(format!(
                        "sqlite migration failed for '{}': {}",
                        stmt, text
                    )));
                }
            }
        }

        // Backfill queue metadata for legacy databases that only persisted queue_messages.
        conn.execute(
            "INSERT OR IGNORE INTO queue_metadata (queue_name)
             SELECT DISTINCT queue_name
             FROM queue_messages
             WHERE queue_name IS NOT NULL AND queue_name != ''",
            [],
        )
        .map_err(|e| {
            Error::internal(format!(
                "sqlite migration backfill queue metadata error: {}",
                e
            ))
        })?;

        Ok(())
    }

    fn recover_stale(&self, stale_sec: i64) -> Result<usize> {
        let cutoff = Utc::now().timestamp() - stale_sec.max(0);
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        let changed = if stale_sec <= 0 {
            conn.execute(
                "UPDATE queue_messages SET status = 'pending', processing_started_at = NULL WHERE status = 'processing'",
                [],
            )
        } else {
            conn.execute(
                "UPDATE queue_messages SET status = 'pending', processing_started_at = NULL WHERE status = 'processing' AND processing_started_at <= ?1",
                params![cutoff],
            )
        }
        .map_err(|e| Error::internal(format!("sqlite recover stale error: {}", e)))?;

        Ok(changed)
    }

    fn queue_known(conn: &Connection, queue_name: &str) -> Result<bool> {
        let metadata_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM queue_metadata WHERE queue_name = ?1",
                params![queue_name],
                |row| row.get(0),
            )
            .map_err(|e| Error::internal(format!("sqlite queue metadata lookup error: {}", e)))?;

        if metadata_count > 0 {
            return Ok(true);
        }

        let message_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM queue_messages WHERE queue_name = ?1",
                params![queue_name],
                |row| row.get(0),
            )
            .map_err(|e| Error::internal(format!("sqlite queue messages lookup error: {}", e)))?;

        Ok(message_count > 0)
    }

    fn require_queue_exists(conn: &Connection, queue_name: &str) -> Result<()> {
        if Self::queue_known(conn, queue_name)? {
            Ok(())
        } else {
            Err(Error::NotFound(format!("queue '{}' not found", queue_name)))
        }
    }
}

impl QueueBackend for SQLiteQueueBackend {
    fn create_queue(&mut self, name: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        let changed = conn
            .execute(
                "INSERT OR IGNORE INTO queue_metadata (queue_name) VALUES (?1)",
                params![name],
            )
            .map_err(|e| Error::internal(format!("sqlite create queue error: {}", e)))?;
        if changed == 0 {
            return Err(Error::AlreadyExists(format!(
                "queue '{}' already exists",
                name
            )));
        }
        Ok(())
    }

    fn remove_queue(&mut self, name: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        let pattern = format!("{}/%", name);
        let deleted_messages = conn
            .execute(
                "DELETE FROM queue_messages WHERE queue_name = ?1 OR queue_name LIKE ?2",
                params![name, pattern],
            )
            .map_err(|e| Error::internal(format!("sqlite remove messages error: {}", e)))?;
        let deleted_meta = conn
            .execute(
                "DELETE FROM queue_metadata WHERE queue_name = ?1 OR queue_name LIKE ?2",
                params![name, pattern],
            )
            .map_err(|e| Error::internal(format!("sqlite remove metadata error: {}", e)))?;

        if deleted_messages == 0 && deleted_meta == 0 {
            return Err(Error::NotFound(format!("queue '{}' not found", name)));
        }
        Ok(())
    }

    fn queue_exists(&self, name: &str) -> bool {
        let conn = match self.conn.lock() {
            Ok(conn) => conn,
            Err(_) => return false,
        };
        Self::queue_known(&conn, name).unwrap_or(false)
    }

    fn list_queues(&self, prefix: &str) -> Vec<String> {
        let conn = match self.conn.lock() {
            Ok(conn) => conn,
            Err(_) => return Vec::new(),
        };

        if prefix.is_empty() {
            let mut stmt =
                match conn.prepare("SELECT queue_name FROM queue_metadata ORDER BY queue_name") {
                    Ok(stmt) => stmt,
                    Err(_) => return Vec::new(),
                };
            let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
                Ok(rows) => rows,
                Err(_) => return Vec::new(),
            };
            return rows.filter_map(|row| row.ok()).collect();
        }

        let mut stmt = match conn.prepare(
            "SELECT queue_name FROM queue_metadata
             WHERE queue_name = ?1 OR queue_name LIKE ?2
             ORDER BY queue_name",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let pattern = format!("{}/%", prefix);
        let rows = match stmt.query_map(params![prefix, pattern], |row| row.get::<_, String>(0)) {
            Ok(rows) => rows,
            Err(_) => return Vec::new(),
        };
        rows.filter_map(|row| row.ok()).collect()
    }

    fn enqueue(&mut self, queue_name: &str, msg: Message) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;
        let stored = serde_json::to_string(&StoredMessage::from_message(&msg))?;
        conn.execute(
            "INSERT INTO queue_messages (queue_name, message_id, data, timestamp, status)
             VALUES (?1, ?2, ?3, ?4, 'pending')",
            params![queue_name, msg.id, stored, unix_secs(msg.timestamp)],
        )
        .map_err(|e| Error::internal(format!("sqlite enqueue error: {}", e)))?;
        Ok(())
    }

    fn dequeue(&mut self, queue_name: &str) -> Result<Option<Message>> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;

        let tx = conn
            .transaction()
            .map_err(|e| Error::internal(format!("sqlite transaction begin error: {}", e)))?;

        let row = tx.query_row(
            "SELECT id, data FROM queue_messages
             WHERE queue_name = ?1 AND status = 'pending'
             ORDER BY id LIMIT 1",
            params![queue_name],
            |row| Ok((row.get::<_, i64>(0)?, read_sqlite_text(row, 1)?)),
        );

        let (id, raw_data) = match row {
            Ok(row) => row,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => {
                return Err(Error::internal(format!(
                    "sqlite dequeue query error: {}",
                    e
                )))
            }
        };

        tx.execute(
            "UPDATE queue_messages
             SET status = 'processing', processing_started_at = ?1
             WHERE id = ?2",
            params![Utc::now().timestamp(), id],
        )
        .map_err(|e| Error::internal(format!("sqlite mark processing error: {}", e)))?;

        tx.commit()
            .map_err(|e| Error::internal(format!("sqlite transaction commit error: {}", e)))?;

        let stored: StoredMessage =
            serde_json::from_str(&raw_data).map_err(|e| Error::Serialization(e.to_string()))?;
        Ok(Some(stored.into_message()))
    }

    fn peek(&self, queue_name: &str) -> Result<Option<Message>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;

        let raw_data = match conn.query_row(
            "SELECT data FROM queue_messages
             WHERE queue_name = ?1 AND status = 'pending'
             ORDER BY id LIMIT 1",
            params![queue_name],
            |row| read_sqlite_text(row, 0),
        ) {
            Ok(data) => data,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(Error::internal(format!("sqlite peek query error: {}", e))),
        };

        let stored: StoredMessage = serde_json::from_str(&raw_data)?;
        Ok(Some(stored.into_message()))
    }

    fn size(&self, queue_name: &str) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM queue_messages
                 WHERE queue_name = ?1 AND status = 'pending'",
                params![queue_name],
                |row| row.get(0),
            )
            .map_err(|e| Error::internal(format!("sqlite size query error: {}", e)))?;
        Ok(count as usize)
    }

    fn list_unacked(&self, queue_name: &str) -> Result<Vec<Message>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;
        let mut stmt = conn
            .prepare(
                "SELECT data FROM queue_messages
                 WHERE queue_name = ?1 AND status IN ('pending', 'processing')
                 ORDER BY id",
            )
            .map_err(|e| Error::internal(format!("sqlite list unacked prepare error: {}", e)))?;
        let rows = stmt
            .query_map(params![queue_name], |row| read_sqlite_text(row, 0))
            .map_err(|e| Error::internal(format!("sqlite list unacked query error: {}", e)))?;

        let mut messages = Vec::new();
        for row in rows {
            let raw_data =
                row.map_err(|e| Error::internal(format!("sqlite list unacked row error: {}", e)))?;
            let stored: StoredMessage = serde_json::from_str(&raw_data)?;
            messages.push(stored.into_message());
        }
        Ok(messages)
    }

    fn clear(&mut self, queue_name: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;

        conn.execute(
            "DELETE FROM queue_messages WHERE queue_name = ?1",
            params![queue_name],
        )
        .map_err(|e| Error::internal(format!("sqlite clear error: {}", e)))?;
        Ok(())
    }

    fn get_last_enqueue_time(&self, queue_name: &str) -> Result<SystemTime> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;

        let timestamp: Option<i64> = conn
            .query_row(
                "SELECT MAX(timestamp) FROM queue_messages
                 WHERE queue_name = ?1 AND status = 'pending'",
                params![queue_name],
                |row| row.get(0),
            )
            .map_err(|e| Error::internal(format!("sqlite last enqueue query error: {}", e)))?;

        Ok(timestamp
            .map(|secs| {
                let secs_u64 = if secs <= 0 { 0 } else { secs as u64 };
                UNIX_EPOCH + Duration::from_secs(secs_u64)
            })
            .unwrap_or(SystemTime::UNIX_EPOCH))
    }

    fn ack(&mut self, queue_name: &str, msg_id: &str) -> Result<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| Error::internal(format!("sqlite mutex poisoned: {}", e)))?;

        Self::require_queue_exists(&conn, queue_name)?;

        let changed = conn
            .execute(
                "DELETE FROM queue_messages
                 WHERE queue_name = ?1 AND message_id = ?2 AND status = 'processing'",
                params![queue_name, msg_id],
            )
            .map_err(|e| Error::internal(format!("sqlite ack error: {}", e)))?;
        Ok(changed > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::tempdir;
    use tempfile::TempDir;

    fn keyed_request(key: &str, signature: &str, data: &str) -> KeyedEnqueueRequest {
        KeyedEnqueueRequest {
            dispatch_key: key.to_string(),
            merge_signature: signature.to_string(),
            data: data.to_string(),
        }
    }

    fn decode_batch(message: &Message) -> KeyedBatchBody {
        serde_json::from_slice::<KeyedBatchEnvelope>(&message.data)
            .unwrap()
            ._queuefs_keyed_batch
    }

    fn pending_batches_for_key(backend: &MemoryBackend, queue: &str, key: &str) -> usize {
        backend
            .list_unacked(queue)
            .unwrap()
            .into_iter()
            .filter(|message| message.dispatch_key.as_deref() == Some(key))
            .count()
    }

    /// Create an in-memory backend with a test queue.
    fn memory_backend_with_queue(queue: &str) -> MemoryBackend {
        let mut backend = MemoryBackend::new();
        backend.create_queue(queue).unwrap();
        backend
    }

    /// Create a SQLite backend backed by a temporary database file.
    fn sqlite_backend() -> (TempDir, String, SQLiteQueueBackend) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("queue.db");
        let db_path_str = db_path.to_str().unwrap().to_string();
        let backend =
            SQLiteQueueBackend::open(&db_path_str, SQLiteQueueOptions::default()).unwrap();
        (dir, db_path_str, backend)
    }

    #[test]
    fn test_create_queue() {
        let mut backend = MemoryBackend::new();

        backend.create_queue("test").unwrap();
        assert!(backend.queue_exists("test"));

        // Creating duplicate should fail
        let result = backend.create_queue("test");
        assert!(result.is_err());
    }

    #[test]
    fn memory_keyed_enqueue_merges_pending_and_serializes_processing_key() {
        let mut backend = MemoryBackend::new();
        backend.create_queue("Semantic").unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("k", "s", "one"))
            .unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("k", "s", "two"))
            .unwrap();
        assert_eq!(backend.size("Semantic").unwrap(), 1);

        let active = backend.dequeue("Semantic").unwrap().unwrap();
        assert_eq!(decode_batch(&active).contributions, vec!["one", "two"]);
        backend
            .enqueue_keyed("Semantic", keyed_request("k", "s", "three"))
            .unwrap();
        assert!(backend.dequeue("Semantic").unwrap().is_none());

        backend
            .enqueue_keyed("Semantic", keyed_request("other", "s", "four"))
            .unwrap();
        assert_eq!(
            backend
                .dequeue("Semantic")
                .unwrap()
                .unwrap()
                .dispatch_key
                .as_deref(),
            Some("other")
        );
        assert!(backend.ack("Semantic", &active.id).unwrap());
        assert_eq!(
            backend
                .dequeue("Semantic")
                .unwrap()
                .unwrap()
                .dispatch_key
                .as_deref(),
            Some("k")
        );
    }

    #[test]
    fn memory_requeue_preserves_payload_and_precedes_successor() {
        let mut backend = MemoryBackend::new();
        backend.create_queue("Semantic").unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("k", "s", "one"))
            .unwrap();
        let active = backend.dequeue("Semantic").unwrap().unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("k", "s", "two"))
            .unwrap();
        assert!(backend.requeue("Semantic", &active.id).unwrap());
        let retried = backend.dequeue("Semantic").unwrap().unwrap();
        assert_eq!(retried.id, active.id);
        assert_eq!(decode_batch(&retried).contributions, vec!["one"]);
        assert!(backend.dequeue("Semantic").unwrap().is_none());
        backend.ack("Semantic", &retried.id).unwrap();
        assert_eq!(
            decode_batch(&backend.dequeue("Semantic").unwrap().unwrap()).contributions,
            vec!["two"],
        );
    }

    #[test]
    fn memory_keyed_batches_split_on_signature_count_and_bytes() {
        let mut backend = MemoryBackend::new();
        backend.create_queue("Semantic").unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("signature", "s1", "one"))
            .unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("signature", "s2", "two"))
            .unwrap();
        assert_eq!(backend.size("Semantic").unwrap(), 2);

        for index in 0..=MAX_KEYED_BATCH_CONTRIBUTIONS {
            backend
                .enqueue_keyed("Semantic", keyed_request("count", "s", &index.to_string()))
                .unwrap();
        }
        assert_eq!(pending_batches_for_key(&backend, "Semantic", "count"), 2);

        let half_limit = "x".repeat(MAX_KEYED_BATCH_BYTES / 2);
        backend
            .enqueue_keyed("Semantic", keyed_request("bytes", "s", &half_limit))
            .unwrap();
        backend
            .enqueue_keyed("Semantic", keyed_request("bytes", "s", &half_limit))
            .unwrap();
        assert_eq!(pending_batches_for_key(&backend, "Semantic", "bytes"), 2);
    }

    #[test]
    fn memory_list_unacked_contains_pending_and_processing() {
        let mut backend = MemoryBackend::new();
        backend.create_queue("Semantic").unwrap();
        backend
            .enqueue("Semantic", Message::new(b"one".to_vec()))
            .unwrap();
        backend
            .enqueue("Semantic", Message::new(b"two".to_vec()))
            .unwrap();
        let active = backend.dequeue("Semantic").unwrap().unwrap();
        let ids = backend
            .list_unacked("Semantic")
            .unwrap()
            .into_iter()
            .map(|message| message.id)
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&active.id));
    }

    #[test]
    fn test_remove_queue() {
        let mut backend = memory_backend_with_queue("test");
        backend.remove_queue("test").unwrap();
        assert!(!backend.queue_exists("test"));

        // Removing non-existent queue should fail
        let result = backend.remove_queue("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_queues() {
        let mut backend = MemoryBackend::new();

        backend.create_queue("queue1").unwrap();
        backend.create_queue("queue2").unwrap();
        backend.create_queue("logs/errors").unwrap();

        let all = backend.list_queues("");
        assert_eq!(all.len(), 3);

        let logs = backend.list_queues("logs");
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0], "logs/errors");
    }

    #[test]
    fn test_enqueue_dequeue() {
        let mut backend = memory_backend_with_queue("test");

        let msg1 = Message::new(b"message 1".to_vec());
        let msg2 = Message::new(b"message 2".to_vec());

        backend.enqueue("test", msg1.clone()).unwrap();
        backend.enqueue("test", msg2.clone()).unwrap();

        assert_eq!(backend.size("test").unwrap(), 2);

        let dequeued1 = backend.dequeue("test").unwrap().unwrap();
        assert_eq!(dequeued1.data, b"message 1");
        assert!(backend.ack("test", &dequeued1.id).unwrap());

        let dequeued2 = backend.dequeue("test").unwrap().unwrap();
        assert_eq!(dequeued2.data, b"message 2");
        assert!(backend.ack("test", &dequeued2.id).unwrap());

        assert_eq!(backend.size("test").unwrap(), 0);
        assert!(backend.dequeue("test").unwrap().is_none());
    }

    #[test]
    fn test_peek() {
        let mut backend = memory_backend_with_queue("test");

        let msg = Message::new(b"test message".to_vec());
        backend.enqueue("test", msg.clone()).unwrap();

        let peeked1 = backend.peek("test").unwrap().unwrap();
        assert_eq!(peeked1.data, b"test message");

        let peeked2 = backend.peek("test").unwrap().unwrap();
        assert_eq!(peeked2.data, b"test message");

        // Size should still be 1
        assert_eq!(backend.size("test").unwrap(), 1);
    }

    #[test]
    fn test_clear() {
        let mut backend = memory_backend_with_queue("test");

        backend
            .enqueue("test", Message::new(b"msg1".to_vec()))
            .unwrap();
        backend
            .enqueue("test", Message::new(b"msg2".to_vec()))
            .unwrap();

        assert_eq!(backend.size("test").unwrap(), 2);

        backend.clear("test").unwrap();
        assert_eq!(backend.size("test").unwrap(), 0);
    }

    #[test]
    fn test_multi_queue_isolation() {
        let mut backend = MemoryBackend::new();
        backend.create_queue("queue1").unwrap();
        backend.create_queue("queue2").unwrap();

        backend
            .enqueue("queue1", Message::new(b"msg1".to_vec()))
            .unwrap();
        backend
            .enqueue("queue2", Message::new(b"msg2".to_vec()))
            .unwrap();

        assert_eq!(backend.size("queue1").unwrap(), 1);
        assert_eq!(backend.size("queue2").unwrap(), 1);

        let msg1 = backend.dequeue("queue1").unwrap().unwrap();
        assert_eq!(msg1.data, b"msg1");
        assert!(backend.ack("queue1", &msg1.id).unwrap());

        // queue2 should be unaffected
        assert_eq!(backend.size("queue2").unwrap(), 1);
    }

    #[test]
    fn test_operations_on_nonexistent_queue() {
        let mut backend = MemoryBackend::new();

        assert!(backend
            .enqueue("nonexistent", Message::new(b"data".to_vec()))
            .is_err());
        assert!(backend.dequeue("nonexistent").is_err());
        assert!(backend.peek("nonexistent").is_err());
        assert!(backend.size("nonexistent").is_err());
        assert!(backend.clear("nonexistent").is_err());
    }

    #[test]
    fn test_sqlite_backend_basic_flow() {
        let (_dir, _db_path, mut backend) = sqlite_backend();

        backend.create_queue("test").unwrap();
        let msg1 = Message::new(b"message 1".to_vec());
        let msg2 = Message::new(b"message 2".to_vec());
        let msg1_id = msg1.id.clone();

        backend.enqueue("test", msg1).unwrap();
        backend.enqueue("test", msg2).unwrap();

        let first = backend.dequeue("test").unwrap().unwrap();
        assert_eq!(first.data, b"message 1");
        assert_eq!(backend.size("test").unwrap(), 1);
        let unacked = backend.list_unacked("test").unwrap();
        assert_eq!(unacked.len(), 2);
        assert_eq!(unacked[0].id, msg1_id);
        assert!(backend.ack("test", &msg1_id).unwrap());

        let second = backend.dequeue("test").unwrap().unwrap();
        assert_eq!(second.data, b"message 2");
    }

    #[test]
    fn test_sqlite_backend_preserves_non_utf8_payload_bytes() {
        let (_dir, _db_path, mut backend) = sqlite_backend();

        backend.create_queue("test").unwrap();
        let payload = vec![0xff, 0x00, 0x80, b'a'];
        backend
            .enqueue("test", Message::new(payload.clone()))
            .unwrap();

        let peeked = backend.peek("test").unwrap().unwrap();
        assert_eq!(peeked.data, payload);

        let dequeued = backend.dequeue("test").unwrap().unwrap();
        assert_eq!(dequeued.data, payload);
    }

    #[test]
    fn test_sqlite_backend_recover_stale() {
        let (_dir, db_path_str, _backend) = sqlite_backend();

        let msg_id = {
            let mut backend =
                SQLiteQueueBackend::open(&db_path_str, SQLiteQueueOptions::default()).unwrap();
            backend.create_queue("test").unwrap();
            let msg = Message::new(b"recover me".to_vec());
            let msg_id = msg.id.clone();
            backend.enqueue("test", msg).unwrap();
            let dequeued = backend.dequeue("test").unwrap().unwrap();
            assert_eq!(dequeued.id, msg_id);
            msg_id
        };

        let mut reopened =
            SQLiteQueueBackend::open(&db_path_str, SQLiteQueueOptions::default()).unwrap();
        let recovered = reopened.dequeue("test").unwrap().unwrap();
        assert_eq!(recovered.id, msg_id);
        assert_eq!(recovered.data, b"recover me");
    }

    #[test]
    fn test_sqlite_backend_dequeue_legacy_go_row() {
        let (_dir, db_path_str, mut backend) = sqlite_backend();

        backend.create_queue("Semantic").unwrap();
        drop(backend);

        let conn = Connection::open(&db_path_str).unwrap();
        conn.execute(
            "INSERT INTO queue_messages (queue_name, message_id, data, timestamp, status)
             VALUES (?1, ?2, ?3, ?4, 'pending')",
            params![
                "Semantic",
                "legacy-msg-id",
                r#"{"id":"legacy-msg-id","data":"{\"id\":\"semantic-inner\",\"uri\":\"viking://resources/demo\",\"context_type\":\"resource\",\"status\":\"pending\",\"timestamp\":1776411350,\"recursive\":true,\"account_id\":\"default\",\"user_id\":\"default\",\"peer_id\":\"default\",\"role\":\"root\",\"skip_vectorization\":false,\"telemetry_id\":\"tm_demo\",\"target_uri\":null,\"lifecycle_lock_handle_id\":\"lock-demo\",\"is_code_repo\":false,\"changes\":null}","timestamp":"2026-04-17T15:37:39.287855+08:00"}"#,
                1776411459_i64,
            ],
        )
        .unwrap();
        drop(conn);

        let mut reopened =
            SQLiteQueueBackend::open(&db_path_str, SQLiteQueueOptions::default()).unwrap();
        let msg = reopened.dequeue("Semantic").unwrap().unwrap();
        let payload = String::from_utf8(msg.data).unwrap();
        assert!(payload.contains("\"uri\":\"viking://resources/demo\""));
    }

    #[test]
    fn test_sqlite_backend_backfills_queue_metadata_from_legacy_rows() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("legacy.db");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let conn = Connection::open(&db_path_str).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE queue_metadata (
                queue_name TEXT PRIMARY KEY,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                last_updated INTEGER DEFAULT (strftime('%s', 'now'))
            );
            CREATE TABLE queue_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_name TEXT NOT NULL,
                message_id TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            INSERT INTO queue_messages (queue_name, message_id, data, timestamp)
            VALUES ('legacy/semantic', 'legacy-id', '{"id":"legacy-id","data":"payload"}', 1776411459);
            "#,
        )
        .unwrap();
        drop(conn);

        let backend =
            SQLiteQueueBackend::open(&db_path_str, SQLiteQueueOptions::default()).unwrap();
        assert!(backend.queue_exists("legacy/semantic"));
        assert_eq!(backend.list_queues("legacy"), vec!["legacy/semantic"]);
        drop(backend);

        let conn = Connection::open(&db_path_str).unwrap();
        let migrated_indexes = conn
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'index'
                   AND name IN ('idx_queue_message_id', 'idx_queue_status')
                 ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            migrated_indexes,
            vec![
                "idx_queue_message_id".to_string(),
                "idx_queue_status".to_string()
            ]
        );
    }

    #[test]
    fn test_sqlite_backend_enables_full_auto_vacuum_for_new_databases() {
        let (_dir, db_path, _backend) = sqlite_backend();

        let conn = Connection::open(db_path).unwrap();
        let auto_vacuum: i64 = conn.query_row("PRAGMA auto_vacuum", [], |row| row.get(0)).unwrap();
        assert_eq!(auto_vacuum, 1);
    }

    #[test]
    fn test_sqlite_backend_does_not_rewrite_existing_auto_vacuum_mode() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("legacy-auto-vacuum.db");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let conn = Connection::open(&db_path_str).unwrap();
        conn.execute_batch(
            r#"
            PRAGMA auto_vacuum=NONE;
            CREATE TABLE queue_metadata (
                queue_name TEXT PRIMARY KEY,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                last_updated INTEGER DEFAULT (strftime('%s', 'now'))
            );
            CREATE TABLE queue_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_name TEXT NOT NULL,
                message_id TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            "#,
        )
        .unwrap();
        drop(conn);

        let _backend = SQLiteQueueBackend::open(&db_path_str, SQLiteQueueOptions::default()).unwrap();

        let conn = Connection::open(&db_path_str).unwrap();
        let auto_vacuum: i64 = conn.query_row("PRAGMA auto_vacuum", [], |row| row.get(0)).unwrap();
        assert_eq!(auto_vacuum, 0);
    }

    #[test]
    fn test_sqlite_backend_operations_on_nonexistent_queue() {
        let (_dir, _db_path, mut backend) = sqlite_backend();

        assert!(backend
            .enqueue("nonexistent", Message::new(b"data".to_vec()))
            .is_err());
        assert!(backend.dequeue("nonexistent").is_err());
        assert!(backend.peek("nonexistent").is_err());
        assert!(backend.size("nonexistent").is_err());
        assert!(backend.clear("nonexistent").is_err());
        assert!(backend.get_last_enqueue_time("nonexistent").is_err());
        assert!(backend.ack("nonexistent", "msg-id").is_err());
    }
}
