PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE frames (
    frame_index INTEGER PRIMARY KEY,
    timestamp_ns INTEGER NOT NULL,
    pts INTEGER,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    keypoint_count INTEGER NOT NULL,
    keypoints_xy_f32 BLOB NOT NULL,
    scores_f32 BLOB NOT NULL,
    descriptors BLOB NOT NULL,
    descriptor_encoding TEXT NOT NULL CHECK(descriptor_encoding IN ('f16', 'f32', 'none')),
    decode_ms REAL NOT NULL,
    preprocess_ms REAL NOT NULL,
    inference_ms REAL NOT NULL,
    readback_ms REAL NOT NULL,
    landmark_ids_u64 BLOB NOT NULL,
    landmark_similarities_f32 BLOB NOT NULL,
    new_landmarks INTEGER NOT NULL,
    matched_landmarks INTEGER NOT NULL,
    tracking_ms REAL NOT NULL
);

CREATE TABLE landmarks (
    landmark_id INTEGER PRIMARY KEY,
    observation_count INTEGER NOT NULL,
    resultant_f32 BLOB NOT NULL,
    concentration REAL NOT NULL,
    first_frame INTEGER NOT NULL,
    last_frame INTEGER NOT NULL
);

CREATE INDEX frames_timestamp ON frames(timestamp_ns);
