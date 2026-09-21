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
    descriptor_encoding TEXT NOT NULL CHECK(descriptor_encoding IN ('f16', 'f32')),
    decode_ms REAL NOT NULL,
    preprocess_ms REAL NOT NULL,
    inference_ms REAL NOT NULL,
    readback_ms REAL NOT NULL
);

