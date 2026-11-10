-- VeeOhTracker — V1 per-crossing log schema (§5.7)
-- One row per counted tripwire crossing. Flat/denormalized on purpose:
-- the write path is a resource-constrained edge node, the read path is
-- append-only time-series (same shape the V2 MQTT publisher ships and V3 aggregates).
--
-- schema_version is stamped per-row (see the provenance block) so a row's DDL
-- shape stays detectable after it leaves the node and lands in a mixed-shape
-- aggregate (the V2 MQTT store / V3 fleet roll-up, where PRAGMA user_version
-- doesn't travel with the row and a staggered OTA rollout means heterogeneous
-- schemas coexist). user_version is kept as the file-level default for local
-- single-node reads; the per-row column is the source of truth on the wire.
PRAGMA user_version = 1;

CREATE TABLE crossing (
    ----------------------------------------------------------------------
    -- Identity / dedup
    ----------------------------------------------------------------------
    crossing_id                     TEXT    NOT NULL,   -- UUID; one per counted crossing
    session_id                      TEXT    NOT NULL,   -- one deployment session (dawn->dusk)
    node_id                         TEXT    NOT NULL,   -- physical device
    track_id                        INTEGER NOT NULL,   -- tracker-assigned ID within session

    ----------------------------------------------------------------------
    -- Provenance (V3 attribution, §5.6) -- irrecoverable if not stamped now
    ----------------------------------------------------------------------
    schema_version                  TEXT    NOT NULL,   -- crossing-row DDL shape; distinct from config_version (see header)
    model_version                   TEXT    NOT NULL,   -- detector artifact hash/tag
    config_version                  TEXT    NOT NULL,   -- thresholds + tripwire set hash

    ----------------------------------------------------------------------
    -- The count (Tier 1/2, §7)
    ----------------------------------------------------------------------
    rec_ld_ts                       TEXT    NOT NULL,   -- ISO-8601 UTC crossing instant
    direction                       TEXT    NOT NULL,   -- in / out
    committed_direction_confidence  REAL    NOT NULL,   -- 0..1, cleanliness of pre-convergence heading

    ----------------------------------------------------------------------
    -- The class (Tier 3, §7)
    ----------------------------------------------------------------------
    user_type                       TEXT    NOT NULL,   -- walker/runner/cyclist/dog_walker/ambiguous
    ambiguous_reason                TEXT,               -- NULL unless user_type='ambiguous'
    association                     TEXT    NOT NULL DEFAULT 'none',  -- none/bicycle/dog
    association_frame_count         INTEGER NOT NULL DEFAULT 0,       -- approach frames carrying bike/dog (§5.4.1)

    ----------------------------------------------------------------------
    -- Kinematics (tuning + audit)
    ----------------------------------------------------------------------
    speed_mps                       REAL,               -- windowed ground speed; NULL if gated out pre-speed
    speed_fit_residual              REAL,               -- robust-fit residual (§5.3); foot-point-jitter proxy

    ----------------------------------------------------------------------
    -- Track quality (Rung 0 + V3 drift, §5.2/§5.6)
    ----------------------------------------------------------------------
    track_length_frames             INTEGER NOT NULL,
    track_gap_count                 INTEGER NOT NULL DEFAULT 0,
    track_instability_flag          INTEGER NOT NULL DEFAULT 0,       -- 0/1

    ----------------------------------------------------------------------
    -- Detector health (V3 drift, §5.6)
    ----------------------------------------------------------------------
    detection_confidence            REAL,               -- crossing-moment confidence; raw material for PSI

    ----------------------------------------------------------------------
    -- Constraints
    ----------------------------------------------------------------------
    PRIMARY KEY (crossing_id),

    CHECK (direction IN ('in', 'out')),
    CHECK (user_type IN ('walker', 'runner', 'cyclist', 'dog_walker', 'ambiguous')),
    CHECK (association IN ('none', 'bicycle', 'dog')),
    CHECK (ambiguous_reason IS NULL OR ambiguous_reason IN
           ('rung0_track_quality', 'rung2_speed_band', 'speed_ceiling', 'no_signal')),
    CHECK (track_instability_flag IN (0, 1)),
    CHECK (committed_direction_confidence BETWEEN 0.0 AND 1.0),
    CHECK (association_frame_count >= 0),
    CHECK (track_length_frames >= 0),
    CHECK (track_gap_count >= 0),

    -- Invariant: a reason is present iff the label is 'ambiguous'
    CHECK ((user_type = 'ambiguous') = (ambiguous_reason IS NOT NULL)),
    -- Ladder invariants: a committed cyclist/dog_walker must carry the matching association
    CHECK (user_type <> 'cyclist'     OR association = 'bicycle'),
    CHECK (user_type <> 'dog_walker'  OR association = 'dog')
);

-- Read-path indexes: time-series scans, per-session reconciliation, drift-by-model segmentation.
CREATE INDEX idx_crossing_rec_ld_ts     ON crossing (rec_ld_ts);
CREATE INDEX idx_crossing_session_ts    ON crossing (session_id, rec_ld_ts);
CREATE INDEX idx_crossing_model_ts      ON crossing (model_version, rec_ld_ts);
