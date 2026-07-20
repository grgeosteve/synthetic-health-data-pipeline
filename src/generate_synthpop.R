library(synthpop)
library(yaml)

#' Parse command-line arguments.
#'
#' @return A named list containing the configuration file path. 
parse_arguments <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    config_path <- "configs/config.yaml"

    flag_index <- which(args == "--config-path")
    if (length(flag_index) > 0 && length(args) >= flag_index + 1) {
        config_path <- args[flag_index + 1]
    }

    list(config_path = config_path)
}

#' Load the configuration file.
#' 
#' Reads and parses a YAML configuration file from the specified path.
#'
#' @param path The path to the configuration file.
#' 
#' @return A named list containing the configuration file contents. 
#' 
#' @section Errors:
#' Throws a `stop()` error if the config file is missing.
load_config <- function(path) {
    if (!file.exists(path)) {
        stop(paste0("Config file not found: '", path, "'"))
    }

    yaml::read_yaml(path)
}

#' Load a CSV file.
#'
#' @param path    The path to the CSV file.
#' 
#' @return A `data.frame` containing the raw CSV data. 
#' 
#' @section Errors:
#' Throws a `stop()` error if the file does not exist at the provided path.
load_csv <- function(path) {
    if (!file.exists(path)) {
        stop(paste0("Data not found at: ", path))
    }

    read.csv(path, na.strings = "NA", stringsAsFactors = FALSE)
}

#' Write data to a CSV file.
#' 
#' Writes a data frame to disk. Creates a parent directory if needed.
#' Guards against silent overwrite of an existing file.
#' Missing values are explicitly written as the literal token "NA".
#'
#' @param path The destination file path. 
#' @param data The data to write in the CSV file.
#' 
#' @return Returns `NULL` invisibly.
#' 
#' @section Errors:
#' Throws a `stop()` error if the file already exists.
write_csv <- function(path, data) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

    if (file.exists(path)) {
        stop(paste0(
            "File ", path, " already exists. ",
            "Please delete or rename it before proceeding."
        ))
    }

    write.csv(data, path, row.names = FALSE, na = "NA")
    invisible(NULL)
}

#' Validate that the config's column groupings exactly match the data.
#'
#' Every column in the data must be declared in exactly one of numeric,
#' binary, or categorical, and every declared column must be
#' present in the data.
#'
#' @param data A `data.frame` to check.
#' @param columns A named `list` containing the character vectors for `numeric`,
#'   `binary`, and `categorical` column groupings from the config.
#'
#' @return Returns `NULL` invisibly if validation passes.
#'
#' @section Errors:
#' Throws a `stop()` error if any data column is undeclared, if any
#' declared column is missing from the data, or if any column is declared in
#' more than one group.
validate_column_config <- function(data, columns) {
    all_declared <- c(columns$numeric, columns$binary, columns$categorical)
    declared <- unique(all_declared)
    actual <- names(data)

    undeclared <- setdiff(actual, declared)
    stale <- setdiff(declared, actual)
    duplicated <- unique(all_declared[duplicated(all_declared)])

    errors <- c()
    if (length(undeclared) > 0) {
        errors <- c(errors, paste0(
            "present in data but not declared in config columns: ",
            paste(undeclared, collapse = ", ")
        ))
    }
    if (length(stale) > 0) {
        errors <- c(errors, paste0(
            "present in config columns but not present in data: ",
            paste(stale, collapse = ", ")
        ))
    }
    if (length(duplicated) > 0) {
        errors <- c(errors, paste0(
            "declared in more than one column group: ",
            paste(duplicated, collapse = ", ")
        ))
    }

    if (length(errors) > 0) {
        stop(paste0(
            "Column configuration does not match the data. ",
            paste(errors, collapse = " ")
        ))
    }
    invisible(NULL)
}

#' Apply the config's declared column types to the data.
#'
#' Numeric columns are cast to numeric, binary and categorical columns
#' are cast to factor, so synthpop treats them as classification
#' targets rather than continuous variables.
#'
#' @param data A `data.frame` to type.
#' @param columns A named `list` containing the character vectors for
#'   `numeric`, `binary`, and `categorical` column groupings from config.
#'
#' @return The same `data.frame`, with types applied.
apply_column_types <- function(data, columns) {
    for (col in columns$numeric) {
        data[[col]] <- as.numeric(data[[col]])
    }
    for (col in c(columns$binary, columns$categorical)) {
        data[[col]] <- as.factor(data[[col]])
    }
    data
}

#' Restore each synthetic column's type to match the real data.
#'
#' Driven entirely by inspecting each real column's actual R type.
#' Converts character first when reversing a factor, never directly, since
#' as.integer() or as.logical() on a factor returns the internal level
#' code, not the label's value.
#'
#' @param real The real data.frame, before any factor conversion.
#' @param synthetic The synthesised data.frame to restore.
#'
#' @return The synthetic data.frame, with every column's type matching
#'   the corresponding column in the real data.
restore_column_types <- function(real, synthetic) {
    for (col in names(real)) {
        real_col <- real[[col]]
        syn_col <- synthetic[[col]]
        syn_as_char <- if (is.factor(syn_col)) as.character(syn_col) else syn_col

        synthetic[[col]] <- if (is.logical(real_col)) {
            as.logical(syn_as_char)
        } else if (is.integer(real_col)) {
            as.integer(round(as.numeric(syn_as_char)))
        } else if (is.numeric(real_col)) {
            as.numeric(syn_as_char)
        } else if (is.character(real_col)) {
            as.character(syn_as_char)
        } else {
            syn_col
        }
    }
    synthetic
}

#' Generate a synthetic dataset using synthpop.
#' 
#' Generates a multivariate synthetic dataset using Classification and 
#' Regression Trees (CART) via the synthpop package. If a visit sequence is
#' supplied, synthpop uses it to order conditioning between columns.
#' Output column order always matches the input, regardless of whether
#' a visit sequence was supplied.
#' 
#' @param data The baseline `data.frame` containing real data.
#' @param seed An integer random seed for reproducibility.
#' @param visit_sequence Optional character vector naming the generation order
#'  for synthpop. If `NULL`, synthpop's own default order is used.
#' 
#' @return A `data.frame` containing the generated synthetic data.
generate <- function(data, seed, visit_sequence = NULL) {
    cat("Generating synthpop (L3 - multivariate) synthetic data...\n")
    set.seed(seed)

    syn_args <- list(
        data,
        method = "cart",
        seed = seed,
        print.flag = FALSE
    )
    if (!is.null(visit_sequence)) {
        cat(paste0("Using custom visit sequence: ", paste(visit_sequence, collapse = ", "), "\n"))
        syn_args$visit.sequence <- visit_sequence
    }

    result <- do.call(synthpop::syn, syn_args)
    synthetic <- result$syn[, names(data)]

    cat("Generation completed successfully.\n")
    synthetic
}

#' Main execution flow
main <- function() {
    args <- parse_arguments()
    cat("Generating synthpop synthetic dataset...\n")

    config <- load_config(args$config_path)

    train_path <- file.path(config$paths$processed_dir, "train.csv")

    data <- load_csv(train_path)
    validate_column_config(data, config$columns)

    # Cast data according to column config for synthpop to use
    typed_data <- apply_column_types(data, config$columns)

    # Generate synthetic data
    synthetic <- generate(typed_data, config$seed, config$generation$visit_sequence)

    # Restore types as found in the original data
    synthetic <- restore_column_types(data, synthetic)

    output_path <- file.path(config$paths$synthetic_dir, "synthpop.csv")
    write_csv(output_path, synthetic)

    cat("Synthpop synthetic dataset generation completed successfully.\n")
}

if (sys.nframe() == 0) {
    main()
}
