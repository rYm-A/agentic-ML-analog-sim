-- Analog ML Accelerator Simulation Proposals and Evaluation Database Schema

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS proposals (
    candidate_id TEXT PRIMARY KEY,
    iteration INT,
    solver TEXT NOT NULL,
    noise_model TEXT NOT NULL,
    sparsity_config TEXT DEFAULT '{}',
    solver_params TEXT,
    noise_params TEXT,
    status TEXT CHECK(status IN ('PENDING', 'EVALUATING', 'COMPLETED', 'FAILED', 'REJECTED')),
    is_reference BOOLEAN DEFAULT 0,
    is_active_evaluation BOOLEAN DEFAULT 0,
    accuracy_fid REAL,
    relative_error REAL,
    latency_ms REAL,
    wall_clock_s REAL,
    worker_utilization REAL,
    pareto_optimal BOOLEAN DEFAULT 0,
    selection_reason TEXT,
    phase_1_prompt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    comment TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(candidate_id) REFERENCES proposals(candidate_id) ON DELETE CASCADE
);

-- Indexes on status, solver+noise_model, candidate_id
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_solver_noise ON proposals(solver, noise_model);
CREATE INDEX IF NOT EXISTS idx_proposals_candidate_id ON proposals(candidate_id);
CREATE INDEX IF NOT EXISTS idx_comments_candidate_id ON comments(candidate_id);

-- Trigger to update updated_at timestamp automatically on proposal modification
CREATE TRIGGER IF NOT EXISTS trg_proposals_updated_at
AFTER UPDATE ON proposals
FOR EACH ROW
BEGIN
    UPDATE proposals SET updated_at = CURRENT_TIMESTAMP WHERE candidate_id = OLD.candidate_id;
END;
