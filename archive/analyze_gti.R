#!/usr/bin/env Rscript
# =============================================================================
# GTI Statistical Analysis v2.0 - Publication-Quality Graphics & IR/LIS Metrics
# =============================================================================
#
# Usage:
#   Rscript analyze_gti_v2.R gti_results_XXXXXX.json
#   Rscript analyze_gti_v2.R gti_results_XXXXXX.json --output figures/
#
# Outputs:
#   - 6 publication-ready figures (PNG + PDF)
#   - Comprehensive JSON results
#   - LaTeX table snippets
#
# =============================================================================

# -----------------------------------------------------------------------------
# Setup and Dependencies
# -----------------------------------------------------------------------------

required_packages <- c("ggplot2", "dplyr", "tidyr", "jsonlite", "scales", "gridExtra", "viridis")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript analyze_gti_v2.R <results.json> [--output <dir>]\n")
  cat("\nLooking for most recent gti_results_*.json...\n")
  
  results_files <- list.files(pattern = "gti_results.*\\.json$", full.names = TRUE)
  if (length(results_files) == 0) {
    stop("No GTI results file found. Run gti_validator_v2.py first.")
  }
  input_file <- results_files[length(results_files)]
} else {
  input_file <- args[1]
}

output_dir <- "."
if ("--output" %in% args) {
  idx <- which(args == "--output")
  if (idx < length(args)) {
    output_dir <- args[idx + 1]
  }
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

cat("\n")
cat("=============================================================================\n")
cat("       GTI STATISTICAL ANALYSIS v2.0 - IR/LIS Metrics & Graphics\n")
cat("=============================================================================\n\n")

cat("Loading:", input_file, "\n")

if (!file.exists(input_file)) {
  stop(paste("ERROR: File not found:", input_file))
}

data <- fromJSON(input_file)

# Extract components
results_df <- as.data.frame(data$results)
metrics_9 <- data$metrics_9type
metrics_8 <- data$metrics_8type
per_type_9 <- data$per_type_9
efficiency <- data$efficiency
metadata <- data$metadata

n_total <- nrow(results_df)
cat("Total classifications:", n_total, "\n")
cat("Model:", metadata$model, "\n")
cat("Timestamp:", metadata$timestamp, "\n\n")

# -----------------------------------------------------------------------------
# Data Transformation
# -----------------------------------------------------------------------------

# Create 8-type transformation
results_df <- results_df %>%
  mutate(
    expected_8 = ifelse(expected %in% c("Stag_Hunt", "Assurance_Game"), "Trust_Game", expected),
    predicted_8 = ifelse(predicted %in% c("Stag_Hunt", "Assurance_Game"), "Trust_Game", predicted),
    correct_8 = expected_8 == predicted_8
  )

# Per-type dataframe for plotting
type_stats <- data.frame(
  type = names(per_type_9),
  precision = sapply(per_type_9, function(x) x$precision),
  recall = sapply(per_type_9, function(x) x$recall),
  f1 = sapply(per_type_9, function(x) x$f1),
  support = sapply(per_type_9, function(x) x$support),
  stringsAsFactors = FALSE
)

# -----------------------------------------------------------------------------
# Print Summary
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("                           SUMMARY STATISTICS\n")
cat("=============================================================================\n\n")

cat(sprintf("%-20s %8s %8s %8s %8s\n", "Taxonomy", "Macro F1", "Macro P", "Macro R", "Accuracy"))
cat(paste(rep("-", 60), collapse = ""), "\n")
cat(sprintf("%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", 
            "9-type (R-G)", 
            metrics_9$macro_f1 * 100,
            metrics_9$macro_precision * 100,
            metrics_9$macro_recall * 100,
            metrics_9$accuracy * 100))
cat(sprintf("%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", 
            "8-type (strategic)", 
            metrics_8$macro_f1 * 100,
            metrics_8$macro_precision * 100,
            metrics_8$macro_recall * 100,
            metrics_8$accuracy * 100))
cat("\n")

cat(sprintf("Cohen's Kappa: %.3f\n", metrics_9$cohens_kappa))
cat(sprintf("Cohen's h vs chance: %.3f\n", metrics_9$cohens_h))
cat(sprintf("p-value vs chance: %.2e\n", metrics_9$p_value))
cat("\n")

# -----------------------------------------------------------------------------
# Theme for Publication
# -----------------------------------------------------------------------------

theme_gti <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 11, face = "bold")
  )

# Color palette
type_colors <- c(
  "Trust_Game" = "#2E86AB",
  "Stag_Hunt" = "#2E86AB",
  "Assurance_Game" = "#5DADE2",
  "Coordination_Game" = "#27AE60",
  "Battle_of_the_Sexes" = "#8E44AD",
  "Hero" = "#E74C3C",
  "Compromise" = "#F39C12",
  "Chicken" = "#E67E22",
  "Deadlock" = "#95A5A6",
  "Prisoners_Dilemma" = "#34495E",
  "Harmony" = "#1ABC9C",
  "Peace" = "#3498DB",
  "Concord" = "#9B59B6"
)

cat("=============================================================================\n")
cat("                        GENERATING FIGURES\n")
cat("=============================================================================\n\n")

# -----------------------------------------------------------------------------
# FIGURE 1: Per-Type F1 Bar Chart (Horizontal)
# -----------------------------------------------------------------------------

cat("Creating Figure 1: Per-Type F1 Scores...\n")

fig1 <- ggplot(type_stats, aes(x = reorder(type, f1), y = f1, fill = type)) +
  geom_bar(stat = "identity", alpha = 0.85, width = 0.7) +
  geom_errorbar(aes(ymin = f1 - sqrt(f1*(1-f1)/support) * 1.96,
                    ymax = pmin(1, f1 + sqrt(f1*(1-f1)/support) * 1.96)),
                width = 0.3, color = "gray40") +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "#E74C3C", alpha = 0.7) +
  geom_hline(yintercept = 1/9, linetype = "dotted", color = "gray50") +
  geom_text(aes(label = sprintf("%.0f%%", f1 * 100)), 
            hjust = -0.2, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = type_colors, guide = "none") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.15),
                     breaks = seq(0, 1, 0.2)) +
  coord_flip() +
  labs(
    title = "GTI Classification Performance by Game Type",
    subtitle = sprintf("Macro F1 = %.1f%% | n = %d", metrics_9$macro_f1 * 100, n_total),
    x = NULL,
    y = "F1 Score",
    caption = "Error bars: 95% CI | Dashed: 80% threshold | Dotted: chance (11%)"
  ) +
  theme_gti

ggsave(file.path(output_dir, "fig1_f1_by_type.png"), fig1, width = 10, height = 6, dpi = 300)
ggsave(file.path(output_dir, "fig1_f1_by_type.pdf"), fig1, width = 10, height = 6)
cat("  Saved: fig1_f1_by_type.png/pdf\n")

# -----------------------------------------------------------------------------
# FIGURE 2: Precision vs Recall Scatter
# -----------------------------------------------------------------------------

cat("Creating Figure 2: Precision-Recall Scatter...\n")

fig2 <- ggplot(type_stats, aes(x = recall, y = precision, color = type, size = support)) +
  geom_point(alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50", alpha = 0.5) +
  geom_text(aes(label = type), hjust = -0.1, vjust = -0.5, size = 3, show.legend = FALSE) +
  scale_color_manual(values = type_colors, guide = "none") +
  scale_size_continuous(range = c(4, 12), name = "Support (n)") +
  scale_x_continuous(labels = percent_format(), limits = c(0, 1.1)) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.1)) +
  labs(
    title = "Precision vs Recall by Game Type",
    subtitle = sprintf("Macro P = %.1f%% | Macro R = %.1f%%", 
                       metrics_9$macro_precision * 100, 
                       metrics_9$macro_recall * 100),
    x = "Recall (Sensitivity)",
    y = "Precision (PPV)",
    caption = "Diagonal = balanced P/R | Point size = sample size"
  ) +
  theme_gti +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "fig2_precision_recall.png"), fig2, width = 10, height = 8, dpi = 300)
ggsave(file.path(output_dir, "fig2_precision_recall.pdf"), fig2, width = 10, height = 8)
cat("  Saved: fig2_precision_recall.png/pdf\n")

# -----------------------------------------------------------------------------
# FIGURE 3: Confusion Matrix Heatmap
# -----------------------------------------------------------------------------

cat("Creating Figure 3: Confusion Matrix...\n")

confusion_df <- results_df %>%
  count(expected, predicted) %>%
  group_by(expected) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

# Ensure all type combinations exist
all_types <- unique(c(results_df$expected, results_df$predicted))
full_grid <- expand.grid(expected = all_types, predicted = all_types, stringsAsFactors = FALSE)
confusion_full <- full_grid %>%
  left_join(confusion_df, by = c("expected", "predicted")) %>%
  mutate(
    n = ifelse(is.na(n), 0, n),
    prop = ifelse(is.na(prop), 0, prop)
  )

fig3 <- ggplot(confusion_full, aes(x = predicted, y = expected, fill = prop)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = ifelse(n > 0, n, "")), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "white", mid = "#FEE08B", high = "#1A9850", 
                       midpoint = 0.5, limits = c(0, 1),
                       labels = percent_format(), name = "Row %") +
  labs(
    title = "Classification Confusion Matrix",
    subtitle = "Numbers show counts; color shows row proportion (recall)",
    x = "Predicted Type",
    y = "True Type"
  ) +
  theme_gti +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "right")

ggsave(file.path(output_dir, "fig3_confusion_matrix.png"), fig3, width = 10, height = 8, dpi = 300)
ggsave(file.path(output_dir, "fig3_confusion_matrix.pdf"), fig3, width = 10, height = 8)
cat("  Saved: fig3_confusion_matrix.png/pdf\n")

# -----------------------------------------------------------------------------
# FIGURE 4: 8-Type vs 9-Type Comparison
# -----------------------------------------------------------------------------

cat("Creating Figure 4: Taxonomy Comparison...\n")

taxonomy_df <- data.frame(
  taxonomy = c("9-type\n(R-G Core)", "8-type\n(Strategic)"),
  metric = rep(c("Macro F1", "Precision", "Recall", "Accuracy"), each = 2),
  value = c(
    metrics_9$macro_f1, metrics_8$macro_f1,
    metrics_9$macro_precision, metrics_8$macro_precision,
    metrics_9$macro_recall, metrics_8$macro_recall,
    metrics_9$accuracy, metrics_8$accuracy
  )
)

fig4 <- ggplot(taxonomy_df, aes(x = metric, y = value, fill = taxonomy)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", value * 100)), 
            position = position_dodge(width = 0.8), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("9-type\n(R-G Core)" = "#E74C3C", "8-type\n(Strategic)" = "#2E86AB")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.1)) +
  labs(
    title = "Performance: 9-Type vs 8-Type Taxonomy",
    subtitle = "8-type merges strategically equivalent Stag_Hunt + Assurance_Game",
    x = NULL,
    y = "Score",
    fill = "Taxonomy"
  ) +
  theme_gti

ggsave(file.path(output_dir, "fig4_taxonomy_comparison.png"), fig4, width = 10, height = 6, dpi = 300)
ggsave(file.path(output_dir, "fig4_taxonomy_comparison.pdf"), fig4, width = 10, height = 6)
cat("  Saved: fig4_taxonomy_comparison.png/pdf\n")

# -----------------------------------------------------------------------------
# FIGURE 5: Statistical Significance Plot
# -----------------------------------------------------------------------------

cat("Creating Figure 5: Statistical Significance...\n")

# Simulate accuracy distribution under null hypothesis
set.seed(42)
n_sim <- 10000
chance_level <- 1/9
null_dist <- rbinom(n_sim, n_total, chance_level) / n_total
observed_acc <- metrics_9$accuracy

sig_df <- data.frame(accuracy = null_dist)

fig5 <- ggplot(sig_df, aes(x = accuracy)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, fill = "gray70", color = "gray50", alpha = 0.7) +
  geom_vline(xintercept = observed_acc, color = "#E74C3C", linewidth = 1.5) +
  geom_vline(xintercept = chance_level, color = "gray40", linetype = "dashed", linewidth = 1) +
  annotate("text", x = observed_acc + 0.02, y = Inf, vjust = 2, hjust = 0,
           label = sprintf("Observed\n%.1f%%", observed_acc * 100), 
           color = "#E74C3C", fontface = "bold", size = 4) +
  annotate("text", x = chance_level, y = Inf, vjust = 2, hjust = 1.1,
           label = sprintf("Chance\n%.1f%%", chance_level * 100), 
           color = "gray40", size = 4) +
  scale_x_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(
    title = "Statistical Significance: Observed vs Chance",
    subtitle = sprintf("p < %.0e | Cohen's h = %.2f | n = %d", 
                       metrics_9$p_value, metrics_9$cohens_h, n_total),
    x = "Accuracy",
    y = "Density",
    caption = "Gray: null distribution under random guessing | Red: observed accuracy"
  ) +
  theme_gti

ggsave(file.path(output_dir, "fig5_significance.png"), fig5, width = 10, height = 6, dpi = 300)
ggsave(file.path(output_dir, "fig5_significance.pdf"), fig5, width = 10, height = 6)
cat("  Saved: fig5_significance.png/pdf\n")

# -----------------------------------------------------------------------------
# FIGURE 6: Latency Distribution
# -----------------------------------------------------------------------------

cat("Creating Figure 6: Latency Distribution...\n")

latency_df <- results_df %>% 
  filter(!is.na(latency_ms) & latency_ms > 0) %>%
  select(latency_ms, correct)

if (nrow(latency_df) > 0) {
  fig6 <- ggplot(latency_df, aes(x = latency_ms)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#3498DB", color = "white", alpha = 0.7) +
    geom_density(color = "#2C3E50", linewidth = 1) +
    geom_vline(xintercept = efficiency$mean_latency_ms, color = "#E74C3C", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = efficiency$p95_latency_ms, color = "#F39C12", linetype = "dotted", linewidth = 1) +
    annotate("text", x = efficiency$mean_latency_ms, y = Inf, vjust = 2, hjust = -0.1,
             label = sprintf("Mean: %.0fms", efficiency$mean_latency_ms), color = "#E74C3C", size = 3.5) +
    annotate("text", x = efficiency$p95_latency_ms, y = Inf, vjust = 4, hjust = -0.1,
             label = sprintf("P95: %.0fms", efficiency$p95_latency_ms), color = "#F39C12", size = 3.5) +
    labs(
      title = "Classification Latency Distribution",
      subtitle = sprintf("Median: %.0fms | Throughput: %.2f QPS | Tokens/query: %.0f",
                         efficiency$median_latency_ms, efficiency$throughput_qps, efficiency$tokens_per_query),
      x = "Latency (ms)",
      y = "Density"
    ) +
    theme_gti
  
  ggsave(file.path(output_dir, "fig6_latency.png"), fig6, width = 10, height = 6, dpi = 300)
  ggsave(file.path(output_dir, "fig6_latency.pdf"), fig6, width = 10, height = 6)
  cat("  Saved: fig6_latency.png/pdf\n")
} else {
  cat("  Skipped: No latency data available\n")
}

# -----------------------------------------------------------------------------
# Generate LaTeX Tables
# -----------------------------------------------------------------------------

cat("\nGenerating LaTeX tables...\n")

latex_file <- file.path(output_dir, "gti_tables.tex")

latex_content <- sprintf('
%% GTI Validation Results - LaTeX Tables
%% Generated: %s

%% Table 1: Summary Metrics
\\begin{table}[htbp]
\\centering
\\caption{GTI Classification Performance}
\\label{tab:gti_summary}
\\begin{tabular}{lcccc}
\\toprule
Taxonomy & Macro F1 & Precision & Recall & Accuracy \\\\
\\midrule
9-type (R-G Core) & %.1f\\%% & %.1f\\%% & %.1f\\%% & %.1f\\%% \\\\
8-type (Strategic) & %.1f\\%% & %.1f\\%% & %.1f\\%% & %.1f\\%% \\\\
\\bottomrule
\\end{tabular}
\\end{table}

%% Table 2: Per-Type Metrics
\\begin{table}[htbp]
\\centering
\\caption{Per-Type Classification Metrics (9-type)}
\\label{tab:gti_per_type}
\\begin{tabular}{lcccc}
\\toprule
Game Type & Precision & Recall & F1 & Support \\\\
\\midrule
%s
\\midrule
Macro Average & %.1f\\%% & %.1f\\%% & %.1f\\%% & --- \\\\
\\bottomrule
\\end{tabular}
\\end{table}

%% Table 3: Statistical Tests
\\begin{table}[htbp]
\\centering
\\caption{Statistical Significance Tests}
\\label{tab:gti_stats}
\\begin{tabular}{lc}
\\toprule
Metric & Value \\\\
\\midrule
Sample size ($n$) & %d \\\\
Cohen\'s $\\kappa$ & %.3f \\\\
Cohen\'s $h$ (vs. chance) & %.3f \\\\
$p$-value (vs. chance) & $<$ %.0e \\\\
95\\%% CI (Accuracy) & [%.1f\\%%, %.1f\\%%] \\\\
\\bottomrule
\\end{tabular}
\\end{table}
',
  Sys.time(),
  metrics_9$macro_f1 * 100, metrics_9$macro_precision * 100, metrics_9$macro_recall * 100, metrics_9$accuracy * 100,
  metrics_8$macro_f1 * 100, metrics_8$macro_precision * 100, metrics_8$macro_recall * 100, metrics_8$accuracy * 100,
  paste(sapply(names(per_type_9), function(t) {
    sprintf("%s & %.1f\\%% & %.1f\\%% & %.1f\\%% & %d \\\\",
            gsub("_", "\\\\_", t),
            per_type_9[[t]]$precision * 100,
            per_type_9[[t]]$recall * 100,
            per_type_9[[t]]$f1 * 100,
            per_type_9[[t]]$support)
  }), collapse = "\n"),
  metrics_9$macro_precision * 100, metrics_9$macro_recall * 100, metrics_9$macro_f1 * 100,
  n_total,
  metrics_9$cohens_kappa,
  metrics_9$cohens_h,
  metrics_9$p_value,
  (metrics_9$accuracy - 1.96 * sqrt(metrics_9$accuracy * (1 - metrics_9$accuracy) / n_total)) * 100,
  (metrics_9$accuracy + 1.96 * sqrt(metrics_9$accuracy * (1 - metrics_9$accuracy) / n_total)) * 100
)

writeLines(latex_content, latex_file)
cat("  Saved:", latex_file, "\n")

# -----------------------------------------------------------------------------
# Save Comprehensive Results JSON
# -----------------------------------------------------------------------------

cat("\nSaving comprehensive results...\n")

comprehensive_results <- list(
  metadata = list(
    analysis_timestamp = as.character(Sys.time()),
    input_file = input_file,
    n_samples = n_total,
    r_version = R.version.string
  ),
  metrics_9type = list(
    accuracy = metrics_9$accuracy,
    macro_precision = metrics_9$macro_precision,
    macro_recall = metrics_9$macro_recall,
    macro_f1 = metrics_9$macro_f1,
    cohens_kappa = metrics_9$cohens_kappa,
    cohens_h = metrics_9$cohens_h,
    p_value = metrics_9$p_value,
    ci_95_lower = metrics_9$accuracy - 1.96 * sqrt(metrics_9$accuracy * (1 - metrics_9$accuracy) / n_total),
    ci_95_upper = metrics_9$accuracy + 1.96 * sqrt(metrics_9$accuracy * (1 - metrics_9$accuracy) / n_total)
  ),
  metrics_8type = list(
    accuracy = metrics_8$accuracy,
    macro_precision = metrics_8$macro_precision,
    macro_recall = metrics_8$macro_recall,
    macro_f1 = metrics_8$macro_f1
  ),
  per_type = as.list(type_stats %>% 
    select(type, precision, recall, f1, support) %>%
    split(.$type) %>%
    lapply(function(x) as.list(x[1, -1])))
)

results_json <- file.path(output_dir, "gti_analysis_results.json")
write_json(comprehensive_results, results_json, pretty = TRUE, auto_unbox = TRUE)
cat("  Saved:", results_json, "\n")

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

cat("\n")
cat("=============================================================================\n")
cat("                           ANALYSIS COMPLETE\n")
cat("=============================================================================\n\n")

cat("OUTPUT FILES:\n")
cat(sprintf("  • %s\n", file.path(output_dir, "fig1_f1_by_type.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "fig2_precision_recall.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "fig3_confusion_matrix.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "fig4_taxonomy_comparison.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "fig5_significance.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "fig6_latency.png")))
cat(sprintf("  • %s\n", latex_file))
cat(sprintf("  • %s\n", results_json))

cat("\nKEY FINDINGS:\n")
cat(sprintf("  • Macro F1:       %.1f%% (9-type) / %.1f%% (8-type)\n", 
            metrics_9$macro_f1 * 100, metrics_8$macro_f1 * 100))
cat(sprintf("  • Precision:      %.1f%%\n", metrics_9$macro_precision * 100))
cat(sprintf("  • Recall:         %.1f%%\n", metrics_9$macro_recall * 100))
cat(sprintf("  • Cohen's Kappa:  %.3f (%s)\n", metrics_9$cohens_kappa,
            ifelse(metrics_9$cohens_kappa > 0.8, "almost perfect",
              ifelse(metrics_9$cohens_kappa > 0.6, "substantial",
                ifelse(metrics_9$cohens_kappa > 0.4, "moderate", "fair")))))
cat(sprintf("  • Significance:   p < %.0e, h = %.2f\n", metrics_9$p_value, metrics_9$cohens_h))

cat("\n=============================================================================\n")
