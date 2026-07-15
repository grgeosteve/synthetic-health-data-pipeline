# Data Dictionary: Stroke Prediction Dataset

| Variable | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `id` | Identifier | Identifier | Administrative primary key (Dropped in preprocessing). |
| `gender` | Categorical | Quasi-Identifier | Patient gender. |
| `age` | Continuous | Quasi-Identifier | Patient age. |
| `hypertension` | Binary | Sensitive | Patient hypertension history. |
| `heart_disease` | Binary | Sensitive | Patient heart disease history. |
| `ever_married` | Categorical | Quasi-Identifier | Patient married status. |
| `work_type` | Categorical | Quasi-Identifier | Patient work type. |
| `Residence_type` | Categorical | Quasi-Identifier | Patient residence type. |
| `avg_glucose_level` | Continuous | Sensitive | Patient average glucose levels. |
| `bmi` | Continuous | Sensitive | Body Mass Index (Target of non-random missingness). |
| `smoking_status` | Categorical | Sensitive | Patient smoking status. |
| `stroke` | Binary | Target | Clinical stroke outcome (4.9% prevalence). |

**Roles.** Quasi-Identifier: externally knowable, used in combination in the
singling-out check (`age` banded). Sensitive: a clinical or behavioural
attribute not assumed to be externally knowable. Target: the outcome
variable.

`smoking_status` is treated as sensitive rather than a quasi-identifier, since smoking
history is not externally observable.
