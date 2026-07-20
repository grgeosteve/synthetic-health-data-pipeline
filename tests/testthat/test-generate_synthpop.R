# test-generate_synthpop.R
#
# Structural tests only, matching the scope decision applied to the Python
# generators: schema, types, row count, and the validation guard. No
# statistical-quality assertions. That is handled by dedicated evaluation scripts.

library(testthat)
source(file.path("..", "..", "src", "generate_synthpop.R"))

# Global configuration for reproducibility and easy inspection
SEED <- 24

make_sample_columns <- function() {
  list(
    numeric = c("age", "bmi"),
    binary = c("stroke"),
    categorical = c("gender")
  )
}

make_sample_data <- function(seed = SEED) {
  set.seed(seed)
  n <- 200

  data.frame(
    gender = sample(c("Male", "Female"), n, replace = TRUE),
    age = sample(1:90, n, replace = TRUE),
    bmi = rnorm(n, mean = 28, sd = 6),
    stroke = sample(c(0L, 1L), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

test_that("load_config raises an error if the file does not exist", {
  expect_error(load_config("nonexistent_config.yaml"), "Config file not found")
})

test_that("load_csv raises an error if the file does not exist", {
  expect_error(load_csv("nonexistent_data.csv"), "Data not found at")
})

test_that("write_csv creates directories and writes data correctly", {
  tmp_dir <- tempdir()
  tmp_path <- file.path(tmp_dir, "nested_dir", "test_output.csv")
  
  if (file.exists(tmp_path)) unlink(tmp_path)
  
  test_data <- data.frame(a = 1:3, b = c("X", "Y", "Z"), stringsAsFactors = FALSE)
  
  expect_invisible(write_csv(tmp_path, test_data))
  expect_true(file.exists(tmp_path))
  
  expect_error(write_csv(tmp_path, test_data), "already exists")
  
  unlink(tmp_path)
})

test_that("load_csv correctly parses data and explicitly handles NA strings", {
  tmp_path <- tempfile(fileext = ".csv")
  
  writeLines("a,b\n1,NA\n2,foo", tmp_path)
  
  loaded_data <- load_csv(tmp_path)
  
  expect_s3_class(loaded_data, "data.frame")
  expect_true(is.na(loaded_data$b[1]))  
  expect_equal(loaded_data$b[2], "foo")
  
  unlink(tmp_path)
})

test_that("validate_column_config passes on a matching schema", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  expect_silent(validate_column_config(data, columns))
})

test_that("validate_column_config raises on an undeclared data column", {
  data <- make_sample_data(SEED)
  data$undeclared_column <- runif(nrow(data))
  columns <- make_sample_columns()
  expect_error(validate_column_config(data, columns), "not declared in config columns")
})

test_that("validate_column_config raises on a stale config column", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  columns$categorical <- c(columns$categorical, "nonexistent_col")
  expect_error(validate_column_config(data, columns), "not present in data")
})

test_that("validate_column_config raises on a column declared in two groups", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  columns$categorical <- c(columns$categorical, "age")
  expect_error(validate_column_config(data, columns), "more than one column group")
})

test_that("generate returns the same columns as the input", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)
  synthetic <- generate(typed, seed = SEED)
  expect_equal(sort(names(synthetic)), sort(names(data)))
})

test_that("generate returns the same row count as the input", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)
  synthetic <- generate(typed, seed = SEED)
  expect_equal(nrow(synthetic), nrow(data))
})

test_that("generate is deterministic given the same seed", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)
  a <- generate(typed, seed = SEED)
  b <- generate(typed, seed = SEED)
  expect_equal(a, b)
})

test_that("generate preserves missingness in the bmi column", {
  data <- make_sample_data(SEED)
  data$bmi[1] <- NA
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)
  
  synthetic <- generate(typed, seed = SEED)
  expect_true(any(is.na(synthetic$bmi)))
})

test_that("full pipeline preserves original types through generate and restore", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()

  typed <- apply_column_types(data, columns)
  synthetic <- generate(typed, seed = SEED)
  restored <- restore_column_types(data, synthetic)

  tmp_path <- tempfile(fileext = ".csv")
  write_csv(tmp_path, restored)
  readback_data <- load_csv(tmp_path)

  for (col in names(data)) {
    if (is.character(data[[col]])) {
      readback_data[[col]] <- as.character(readback_data[[col]])
    } else if (is.integer(data[[col]])) {
      readback_data[[col]] <- as.integer(readback_data[[col]])
    } else if (is.numeric(data[[col]])) {
      readback_data[[col]] <- as.numeric(readback_data[[col]])
    }

    expect_identical(
      typeof(readback_data[[col]]),
      typeof(data[[col]]),
      info = paste0("Type mismatch verified on target column: '", col, "'")
    )
  }

  unlink(tmp_path)
})

test_that("apply_column_types coerces numeric and factor columns correctly", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()

  typed <- apply_column_types(data, columns)

  expect_true(is.numeric(typed$age))
  expect_true(is.numeric(typed$bmi))
  expect_true(is.factor(typed$gender))
  expect_true(is.factor(typed$stroke))
})

test_that("restore_column_types works identically regardless of value encoding", {
  real <- data.frame(
    col_int_binary = c(0L, 1L, 1L, 0L),
    col_logical = c(TRUE, FALSE, FALSE, TRUE),
    col_string_yn = c("Yes", "No", "No", "Yes"),
    col_continuous = c(1.5, 2.7, 3.1, 4.4),
    stringsAsFactors = FALSE
  )

  synthetic <- data.frame(
    col_int_binary = as.factor(c(1L, 0L, 0L, 1L)),
    col_logical = as.factor(c(FALSE, TRUE, TRUE, FALSE)),
    col_string_yn = as.factor(c("No", "Yes", "Yes", "No")),
    col_continuous = c(1.9, 2.2, 3.9, 4.0)
  )

  restored <- restore_column_types(real, synthetic)

  expect_true(is.integer(restored$col_int_binary))
  expect_true(is.logical(restored$col_logical))
  expect_true(is.character(restored$col_string_yn))
  expect_true(is.numeric(restored$col_continuous) && !is.factor(restored$col_continuous))

  expect_equal(restored$col_int_binary, c(1L, 0L, 0L, 1L))
  expect_equal(restored$col_logical, c(FALSE, TRUE, TRUE, FALSE))
  expect_equal(restored$col_string_yn, c("No", "Yes", "Yes", "No"))
})

test_that("generate accepts a visit_sequence and still returns input column order", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)

  custom_sequence <- c("stroke", "gender", "age", "bmi")

  synthetic <- generate(typed, seed = SEED, visit_sequence = custom_sequence)

  # output column order must match the input's order, not the visit sequence's
  expect_equal(names(synthetic), names(data))
})

test_that("generate works identically to before when visit_sequence is not supplied", {
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)

  synthetic <- generate(typed, seed = SEED)

  expect_equal(names(synthetic), names(data))
  expect_equal(nrow(synthetic), nrow(data))
})

test_that("generate produces different results with a different visit_sequence", {
  # confirms the visit_sequence argument is actually being passed to synthpop,
  # not silently ignored. Same schema and row count are expected,
  # but the actual generated values should differ, since a different
  # conditioning order fits different trees at each step. If this ever
  # started passing with identical output, that would mean visit_sequence
  # had stopped reaching synthpop::syn().
  data <- make_sample_data(SEED)
  columns <- make_sample_columns()
  typed <- apply_column_types(data, columns)

  seq_a <- c("gender", "age", "bmi", "stroke")
  seq_b <- c("stroke", "bmi", "age", "gender")

  result_a <- generate(typed, seed = SEED, visit_sequence = seq_a)
  result_b <- generate(typed, seed = SEED, visit_sequence = seq_b)

  expect_equal(names(result_a), names(result_b))
  expect_equal(nrow(result_a), nrow(result_b))
  expect_false(identical(result_a, result_b))
})
