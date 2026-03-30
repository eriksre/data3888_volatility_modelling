#!/usr/bin/env Rscript
# Convert every CSV in individual_book_train to Parquet in individual_book_train_parquet.

required_pkgs <- c("arrow", "dplyr")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Install missing packages with: install.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
    "))",
    call. = FALSE
  )
}

library(arrow)

args      <- commandArgs(trailingOnly = FALSE)
file_arg  <- grep("--file=", args, value = TRUE)
root_dir  <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("--file=", "", file_arg)))
} else {
  getwd()
}
input_dir   <- file.path(root_dir, "individual_book_train")
output_dir  <- file.path(root_dir, "individual_book_train_parquet")

if (!dir.exists(input_dir)) {
  stop("Not a directory: ", input_dir, call. = FALSE)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

csv_files <- sort(list.files(input_dir, pattern = "\\.csv$", full.names = TRUE))
if (length(csv_files) == 0) {
  stop("No CSV files found in ", input_dir, call. = FALSE)
}

for (csv_path in csv_files) {
  base_name    <- tools::file_path_sans_ext(basename(csv_path))
  parquet_path <- file.path(output_dir, paste0(base_name, ".parquet"))
  message("Converting ", basename(csv_path), " ...")
  df <- arrow::read_csv_arrow(csv_path)
  arrow::write_parquet(df, parquet_path)
}

message("Done. Parquet files written to: ", output_dir)
