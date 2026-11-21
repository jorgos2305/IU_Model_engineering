-- sqlite schema
DROP TABLE IF EXISTS prediction;
DROP TABLE IF EXISTS observation;
DROP TABLE IF EXISTS patient;

-- observation table
CREATE TABLE observation (
    id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL,
    date_diagnosis TEXT NOT NULL,
    radius_mean REAL NOT NULL,
    texture_mean REAL NOT NULL,
    perimeter_mean REAL NOT NULL,
    area_mean REAL NOT NULL,
    smoothness_mean REAL NOT NULL,
    compactness_mean REAL NOT NULL,
    concavity_mean REAL NOT NULL,
    concave_points_mean REAL NOT NULL,
    symmetry_mean REAL NOT NULL,
    fractal_dimension_mean REAL NOT NULL,
    radius_se REAL NOT NULL,
    texture_se REAL NOT NULL,
    perimeter_se REAL NOT NULL,
    area_se REAL NOT NULL,
    smoothness_se REAL NOT NULL,
    compactness_se REAL NOT NULL,
    concavity_se REAL NOT NULL,
    concave_points_se REAL NOT NULL,
    symmetry_se REAL NOT NULL,
    fractal_dimension_se REAL NOT NULL,
    radius_worst REAL NOT NULL,
    texture_worst REAL NOT NULL,
    perimeter_worst REAL NOT NULL,
    area_worst REAL NOT NULL,
    smoothness_worst REAL NOT NULL,
    compactness_worst REAL NOT NULL,
    concavity_worst REAL NOT NULL,
    concave_points_worst REAL NOT NULL,
    symmetry_worst REAL NOT NULL,
    fractal_dimension_worst REAL NOT NULL
);

-- prediction table
CREATE TABLE prediction (
    id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    diagnosis TEXT NOT NULL
        CHECK (diagnosis IN ('B', 'M')),
    probability_malignant REAL NOT NULL
        CHECK (probability_malignant >= 0 AND probability_malignant <= 1),
    model_name TEXT NOT NULL,
    model_version INTEGER NOT NULL,
    FOREIGN KEY (observation_id)
        REFERENCES observation(id)
);
