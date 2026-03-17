# Single source of truth for all DynamoDB key patterns.

# ---------------------------------------------------------------------------
# PK prefixes
# ---------------------------------------------------------------------------
PK_EXPERIMENT_PREFIX = "EXP#"
PK_MODEL_PREFIX = "RM#"
PK_WORKSPACE_PREFIX = "WORKSPACE#"
PK_USER_PREFIX = "USER#"
PK_CONFIG = "CONFIG"

# ---------------------------------------------------------------------------
# SK prefixes
# ---------------------------------------------------------------------------
SK_EXPERIMENT_META = "E#META"
SK_EXPERIMENT_TAG_PREFIX = "E#TAG#"
SK_EXPERIMENT_NAME_REV = "E#NAME_REV"
SK_RUN_PREFIX = "R#"
SK_TRACE_PREFIX = "T#"
SK_MODEL_META = "M#META"
SK_MODEL_TAG_PREFIX = "M#TAG#"
SK_MODEL_ALIAS_PREFIX = "M#ALIAS#"
SK_MODEL_NAME_REV = "M#NAME_REV"
SK_VERSION_PREFIX = "V#"
SK_VERSION_TAG_SUFFIX = "#TAG#"
SK_WORKSPACE_META = "META"
SK_USER_META = "U#META"
SK_USER_PERM_PREFIX = "U#PERM#"
# Within run: R#<ulid>#METRIC#<key>
SK_METRIC_PREFIX = "#METRIC#"
# Within run: R#<ulid>#MHIST#...
SK_METRIC_HISTORY_PREFIX = "#MHIST#"
SK_PARAM_PREFIX = "#PARAM#"
SK_TAG_PREFIX = "#TAG#"
SK_INPUT_PREFIX = "#INPUT#"
SK_INPUT_TAG_SUFFIX = "#ITAG#"
SK_LOGGED_MODEL_PREFIX = "#LM#"
# V3 experiment-scoped logged models (distinct from V2 SK_LOGGED_MODEL_PREFIX)
SK_LM_PREFIX = "LM#"
SK_LM_TAG_PREFIX = "#TAG#"
SK_LM_PARAM_PREFIX = "#PARAM#"
SK_LM_METRIC_PREFIX = "#METRIC#"
SK_RANK_LM_PREFIX = "RANK#lm#"  # RANK items for logged model metrics (global)
SK_RANK_LMD_PREFIX = "RANK#lmd#"  # RANK items for logged model metrics (dataset-scoped)
SK_DATASET_PREFIX = "D#"
SK_DLINK_PREFIX = "DLINK#"
SK_RANK_PREFIX = "RANK#"
SK_FTS_PREFIX = "FTS#"
SK_FTS_REV_PREFIX = "FTS_REV#"

# ---------------------------------------------------------------------------
# GSI names
# ---------------------------------------------------------------------------
GSI1_NAME = "gsi1"
GSI2_NAME = "gsi2"
GSI3_NAME = "gsi3"
GSI4_NAME = "gsi4"
GSI5_NAME = "gsi5"

# ---------------------------------------------------------------------------
# GSI attribute names
# ---------------------------------------------------------------------------
GSI1_PK = "gsi1pk"
GSI1_SK = "gsi1sk"
GSI2_PK = "gsi2pk"
GSI2_SK = "gsi2sk"
GSI3_PK = "gsi3pk"
GSI3_SK = "gsi3sk"
GSI4_PK = "gsi4pk"
GSI4_SK = "gsi4sk"
GSI5_PK = "gsi5pk"
GSI5_SK = "gsi5sk"

# ---------------------------------------------------------------------------
# LSI attribute names
# ---------------------------------------------------------------------------
LSI1_SK = "lsi1sk"
LSI2_SK = "lsi2sk"
LSI3_SK = "lsi3sk"
LSI4_SK = "lsi4sk"
LSI5_SK = "lsi5sk"

# ---------------------------------------------------------------------------
# GSI PK value prefixes
# ---------------------------------------------------------------------------
GSI1_RUN_PREFIX = "RUN#"
GSI1_TRACE_PREFIX = "TRACE#"
GSI1_CLIENT_PREFIX = "CLIENT#"
GSI1_LM_PREFIX = "LM#"
GSI1_DS_PREFIX = "DS#"
GSI2_EXPERIMENTS_PREFIX = "EXPERIMENTS#"
GSI2_MODELS_PREFIX = "MODELS#"
GSI2_AUTH_USERS = "AUTH_USERS"
GSI2_WORKSPACES = "WORKSPACES"
GSI2_FTS_NAMES_PREFIX = "FTS_NAMES#"
GSI3_EXP_NAME_PREFIX = "EXP_NAME#"
GSI3_MODEL_NAME_PREFIX = "MODEL_NAME#"
GSI3_ALIAS_PREFIX = "ALIAS#"
GSI4_PERM_PREFIX = "PERM#"
GSI5_EXP_NAMES_PREFIX = "EXP_NAMES#"
GSI5_MODEL_NAMES_PREFIX = "MODEL_NAMES#"

# ---------------------------------------------------------------------------
# CONFIG SK constants
# ---------------------------------------------------------------------------
CONFIG_DENORMALIZE_TAGS = "DENORMALIZE_TAGS"
CONFIG_TTL_POLICY = "TTL_POLICY"
CONFIG_FTS_TRIGRAM_FIELDS = "FTS_TRIGRAM_FIELDS"

# ---------------------------------------------------------------------------
# Evaluation Dataset partition (DS#<dataset_id>)
# ---------------------------------------------------------------------------
PK_DATASET_PREFIX = "DS#"
SK_DATASET_META = "DS#META"
SK_DATASET_TAG_PREFIX = "DS#TAG#"
SK_DATASET_RECORD_PREFIX = "DS#REC#"
SK_DATASET_EXP_PREFIX = "DS#EXP#"

# GSI prefixes for evaluation datasets
# Note: GSI1_DS_PREFIX = "DS#" already exists for legacy V2 dataset items.
# The new DS_EXP# prefix is for experiment-dataset associations (distinct purpose).
GSI1_DS_EXP_PREFIX = "DS_EXP#"
GSI2_DS_LIST_PREFIX = "DS_LIST#"
GSI3_DS_NAME_PREFIX = "DS_NAME#"

# ---------------------------------------------------------------------------
# Scorer sort keys (within EXP# partition)
# ---------------------------------------------------------------------------
SK_SCORER_PREFIX = "SCOR#"
SK_SCORER_OSCFG_SUFFIX = "#OSCFG"

# ---------------------------------------------------------------------------
# GSI PK prefixes for scorers
# ---------------------------------------------------------------------------
GSI1_SCOR_PREFIX = "SCOR#"
GSI2_ACTIVE_SCORERS_PREFIX = "ACTIVE_SCORERS#"
GSI3_SCOR_NAME_PREFIX = "SCOR_NAME#"
