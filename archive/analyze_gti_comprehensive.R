#!/usr/bin/env Rscript
# =============================================================================
# Game Theory Index (GTI) - Comprehensive Statistical Analysis
# Adapted from Caprazli Orthogonality Index analysis framework
# =============================================================================
#
# Key differences from COI:
#   - COI: Continuous score predicting binary outcome
#   - GTI: Categorical classifier (multi-class accuracy)
#
# Adapted metrics:
#   - Cohen's d → Cohen's h (for proportions)
#   - ROC/AUC → Multi-class macro-averaged metrics
#   - t-test → Binomial/Chi-square tests
#   - Logistic regression → Multinomial analysis (optional)
#
# =============================================================================

# -----------------------------------------------------------------------------
# Setup and Dependencies
# -----------------------------------------------------------------------------

required_packages <- c("ggplot2", "dplyr", "tidyr", "jsonlite", "scales")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Command line argument for input file
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  input_file <- args[1]
} else {
  # Default: find most recent results file
  results_files <- list.files(pattern = "gti_results.*\\.json$", full.names = TRUE)
  if (length(results_files) == 0) {
    stop("No GTI results file found. Run gti_stats_v2.py first or specify file as argument.")
  }
  input_file <- results_files[length(results_files)]
}

output_dir <- dirname(input_file)
if (output_dir == ".") output_dir <- getwd()

cat("\n")
cat("=============================================================================\n")
cat("       GAME THEORY INDEX - COMPREHENSIVE STATISTICAL ANALYSIS\n")
cat("=============================================================================\n\n")

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

cat("Loading data from:", input_file, "\n\n")

if (!file.exists(input_file)) {
  stop("ERROR: Input file not found: ", input_file)
}

data <- fromJSON(input_file)
results_df <- as.data.frame(data$results)
n_total <- nrow(results_df)

cat("Total classifications:", n_total, "\n")
cat("Timestamp:", data$timestamp, "\n\n")

# -----------------------------------------------------------------------------
# Taxonomy Transformations
# -----------------------------------------------------------------------------

# 8-type: Merge Stag_Hunt + Assurance_Game -> Trust_Game
results_df <- results_df %>%
  mutate(
    expected_8 = ifelse(expected %in% c("Stag_Hunt", "Assurance_Game"), "Trust_Game", expected),
    predicted_8 = ifelse(predicted %in% c("Stag_Hunt", "Assurance_Game"), "Trust_Game", predicted),
    correct_8 = expected_8 == predicted_8
  )

# Strategic families
family_map <- c(
  "Prisoners_Dilemma" = "Conflict",
  "Chicken" = "Conflict", 
  "Deadlock" = "Conflict",
  "Battle_of_the_Sexes" = "Coordination",
  "Coordination_Game" = "Coordination",
  "Stag_Hunt" = "Trust",
  "Assurance_Game" = "Trust",
  "Trust_Game" = "Trust",
  "Hero" = "Sacrifice",
  "Compromise" = "Negotiation",
  "Harmony" = "Cooperative",
  "Peace" = "Cooperative",
  "Concord" = "Cooperative"
)

results_df <- results_df %>%
  mutate(
    family_expected = family_map[expected],
    family_predicted = family_map[predicted],
    correct_family = family_expected == family_predicted
  )

# -----------------------------------------------------------------------------
# Compute Per-Type Statistics
# -----------------------------------------------------------------------------

compute_type_stats <- function(df, expected_col, predicted_col, correct_col) {
  df %>%
    group_by(!!sym(expected_col)) %>%
    summarise(
      n = n(),
      correct = sum(!!sym(correct_col)),
      accuracy = mean(!!sym(correct_col)),
      se = sqrt(accuracy * (1 - accuracy) / n),
      ci_lower = accuracy - 1.96 * se,
      ci_upper = accuracy + 1.96 * se,
      .groups = "drop"
    ) %>%
    rename(type = !!sym(expected_col))
}

stats_9type <- compute_type_stats(results_df, "expected", "predicted", "correct")
stats_8type <- compute_type_stats(results_df, "expected_8", "predicted_8", "correct_8")
stats_family <- compute_type_stats(results_df, "family_expected", "family_predicted", "correct_family")

# Overall accuracies
acc_9type <- mean(results_df$correct)
acc_8type <- mean(results_df$correct_8)
acc_family <- mean(results_df$correct_family, na.rm = TRUE)

# -----------------------------------------------------------------------------
# Precision, Recall, F1 (Per-Type)
# -----------------------------------------------------------------------------

compute_prf <- function(df, expected_col, predicted_col) {
  types <- unique(c(df[[expected_col]], df[[predicted_col]]))
  
  prf_list <- lapply(types, function(t) {
    tp <- sum(df[[expected_col]] == t & df[[predicted_col]] == t)
    fp <- sum(df[[expected_col]] != t & df[[predicted_col]] == t)
    fn <- sum(df[[expected_col]] == t & df[[predicted_col]] != t)
    
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    
    data.frame(type = t, precision = precision, recall = recall, f1 = f1, 
               tp = tp, fp = fp, fn = fn)
  })
  
  bind_rows(prf_list)
}

prf_9type <- compute_prf(results_df, "expected", "predicted")
prf_8type <- compute_prf(results_df, "expected_8", "predicted_8")

# Macro-averaged metrics
macro_precision_9 <- mean(prf_9type$precision)
macro_recall_9 <- mean(prf_9type$recall)
macro_f1_9 <- mean(prf_9type$f1)

macro_precision_8 <- mean(prf_8type$precision)
macro_recall_8 <- mean(prf_8type$recall)
macro_f1_8 <- mean(prf_8type$f1)

# -----------------------------------------------------------------------------
# Summary Statistics
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("                        SUMMARY STATISTICS\n")
cat("=============================================================================\n\n")

cat("TAXONOMY COMPARISON:\n")
cat(sprintf("  %-20s %7s %10s %10s %10s\n", "Taxonomy", "Acc", "Precision", "Recall", "F1"))
cat(sprintf("  %-20s %7s %10s %10s %10s\n", "--------", "---", "---------", "------", "--"))
cat(sprintf("  %-20s %6.1f%% %9.1f%% %9.1f%% %9.1f%%\n", 
            "8-type", acc_8type * 100, macro_precision_8 * 100, macro_recall_8 * 100, macro_f1_8 * 100))
cat(sprintf("  %-20s %6.1f%% %9.1f%% %9.1f%% %9.1f%%\n", 
            "9-type", acc_9type * 100, macro_precision_9 * 100, macro_recall_9 * 100, macro_f1_9 * 100))
cat(sprintf("  %-20s %6.1f%% %9s %9s %9s\n", 
            "Family (5)", acc_family * 100, "—", "—", "—"))
cat("\n")

cat("PER-TYPE ACCURACY (8-type):\n")
print(stats_8type %>% 
        left_join(prf_8type, by = "type") %>%
        select(type, n, accuracy, precision, recall, f1) %>%
        arrange(desc(n)), n = 20)
cat("\n")

# -----------------------------------------------------------------------------
# Statistical Tests
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("                        STATISTICAL TESTS\n")
cat("=============================================================================\n\n")

# Test 1: Accuracy vs Chance
cat("--- Test 1: Accuracy vs Chance (Binomial Exact Test) ---\n\n")

test_vs_chance <- function(k, n, n_types, name) {
  p_chance <- 1 / n_types
  p_observed <- k / n
  
  binom_result <- binom.test(k, n, p = p_chance, alternative = "greater")
  
  # Cohen's h
  h <- 2 * (asin(sqrt(p_observed)) - asin(sqrt(p_chance)))
  
  cat(sprintf("%s:\n", name))
  cat(sprintf("  Chance:       %5.1f%% (1/%d)\n", p_chance * 100, n_types))
  cat(sprintf("  Observed:     %5.1f%% (%d/%d)\n", p_observed * 100, k, n))
  cat(sprintf("  p-value:      %.2e %s\n", binom_result$p.value,
              ifelse(binom_result$p.value < 0.001, "***",
                ifelse(binom_result$p.value < 0.01, "**",
                  ifelse(binom_result$p.value < 0.05, "*", "")))))
  cat(sprintf("  Cohen's h:    %.3f (%s)\n\n", h,
              ifelse(abs(h) < 0.2, "negligible",
                ifelse(abs(h) < 0.5, "small",
                  ifelse(abs(h) < 0.8, "medium", "LARGE")))))
  
  return(list(p = binom_result$p.value, h = h))
}

test_8 <- test_vs_chance(sum(results_df$correct_8), n_total, 8, "8-type taxonomy")
test_9 <- test_vs_chance(sum(results_df$correct), n_total, 9, "9-type taxonomy")

# Test 2: 8-type vs 9-type Comparison
cat("--- Test 2: 8-type vs 9-type Comparison (McNemar + z-test) ---\n\n")

# McNemar's test
b <- sum(results_df$correct_8 & !results_df$correct)  # 8 correct, 9 wrong
c <- sum(!results_df$correct_8 & results_df$correct)  # 8 wrong, 9 correct

cat(sprintf("  Discordant pairs:\n"))
cat(sprintf("    8✓ 9✗: %d\n", b))
cat(sprintf("    8✗ 9✓: %d\n", c))

if (b + c > 0) {
  mcnemar_stat <- (abs(b - c) - 1)^2 / (b + c)
  mcnemar_p <- 1 - pchisq(mcnemar_stat, df = 1)
  cat(sprintf("  McNemar's χ²: %.2f (p = %.4f)\n\n", mcnemar_stat, mcnemar_p))
} else {
  cat("  No discordant pairs\n\n")
}

# Two-proportion z-test
diff_acc <- acc_8type - acc_9type
se_diff <- sqrt(acc_8type * (1 - acc_8type) / n_total + acc_9type * (1 - acc_9type) / n_total)
z_stat <- diff_acc / se_diff
z_p <- 2 * (1 - pnorm(abs(z_stat)))

# Cohen's h for 8 vs 9
h_8v9 <- 2 * (asin(sqrt(acc_8type)) - asin(sqrt(acc_9type)))

cat(sprintf("  Two-proportion z-test:\n"))
cat(sprintf("    Δ accuracy:  %+.1f pp\n", diff_acc * 100))
cat(sprintf("    z-statistic: %.3f\n", z_stat))
cat(sprintf("    p-value:     %.4f\n", z_p))
cat(sprintf("    Cohen's h:   %.3f\n", h_8v9))
cat(sprintf("    95%% CI:      [%.1f%%, %.1f%%]\n\n", 
            (diff_acc - 1.96 * se_diff) * 100,
            (diff_acc + 1.96 * se_diff) * 100))

# Test 3: Stag_Hunt vs Assurance_Game
cat("--- Test 3: Stag_Hunt vs Assurance_Game (Fisher's Exact) ---\n\n")

sh_data <- results_df %>% filter(expected == "Stag_Hunt")
ag_data <- results_df %>% filter(expected == "Assurance_Game")

if (nrow(sh_data) > 0 && nrow(ag_data) > 0) {
  sh_correct <- sum(sh_data$correct)
  ag_correct <- sum(ag_data$correct)
  
  fisher_table <- matrix(c(
    sh_correct, nrow(sh_data) - sh_correct,
    ag_correct, nrow(ag_data) - ag_correct
  ), nrow = 2, byrow = TRUE)
  
  fisher_result <- fisher.test(fisher_table)
  
  cat(sprintf("  Stag_Hunt:      %d/%d (%.1f%%)\n", sh_correct, nrow(sh_data), 
              sh_correct / nrow(sh_data) * 100))
  cat(sprintf("  Assurance_Game: %d/%d (%.1f%%)\n", ag_correct, nrow(ag_data),
              ag_correct / nrow(ag_data) * 100))
  cat(sprintf("  Fisher's p:     %.4f %s\n", fisher_result$p.value,
              ifelse(fisher_result$p.value < 0.001, "***",
                ifelse(fisher_result$p.value < 0.01, "**",
                  ifelse(fisher_result$p.value < 0.05, "*", "")))))
  cat(sprintf("  Odds Ratio:     %.2f\n\n", fisher_result$estimate))
  
  if (fisher_result$p.value < 0.05) {
    cat("  ➜ SIGNIFICANT: Supports merging into Trust_Game\n\n")
  }
}

# Test 4: Homogeneity (Chi-square)
cat("--- Test 4: Homogeneity Across Types (Chi-square) ---\n\n")

contingency <- stats_9type %>%
  mutate(incorrect = n - correct) %>%
  select(type, correct, incorrect)

chi_matrix <- as.matrix(contingency[, c("correct", "incorrect")])
chi_result <- chisq.test(chi_matrix)
cramers_v <- sqrt(chi_result$statistic / (sum(chi_matrix) * (min(dim(chi_matrix)) - 1)))

cat(sprintf("  χ²(%d) = %.2f, p = %.4f\n", chi_result$parameter, 
            chi_result$statistic, chi_result$p.value))
cat(sprintf("  Cramér's V: %.3f (%s)\n\n", cramers_v,
            ifelse(cramers_v < 0.1, "negligible",
              ifelse(cramers_v < 0.3, "small",
                ifelse(cramers_v < 0.5, "medium", "LARGE")))))

# Test 5: Cohen's Kappa (Inter-rater reliability analog)
cat("--- Test 5: Cohen's Kappa (Agreement Beyond Chance) ---\n\n")

# For multi-class, compute simple kappa
p_o <- acc_9type  # Observed agreement
# Expected agreement by chance
type_dist_exp <- table(results_df$expected) / n_total
type_dist_pred <- table(results_df$predicted) / n_total
common_types <- intersect(names(type_dist_exp), names(type_dist_pred))
p_e <- sum(type_dist_exp[common_types] * type_dist_pred[common_types])

kappa <- (p_o - p_e) / (1 - p_e)

cat(sprintf("  Observed agreement (p_o): %.3f\n", p_o))
cat(sprintf("  Expected by chance (p_e): %.3f\n", p_e))
cat(sprintf("  Cohen's κ:                %.3f\n", kappa))
cat(sprintf("  Interpretation: %s\n\n",
            ifelse(kappa < 0.2, "Poor",
              ifelse(kappa < 0.4, "Fair",
                ifelse(kappa < 0.6, "Moderate",
                  ifelse(kappa < 0.8, "Substantial", "Almost Perfect"))))))

# -----------------------------------------------------------------------------
# Pre-registered Hypothesis Test
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("                   PRE-REGISTERED HYPOTHESIS TEST\n")
cat("=============================================================================\n\n")

cat("HYPOTHESIS: The GTI classifier reliably identifies game-theoretic\n")
cat("            structures from narrative descriptions.\n\n")

cat("VALIDATION CRITERIA:\n")
cat("  1. 8-type accuracy > 80%\n")
cat("  2. All individual types > 50% accuracy\n")
cat("  3. Significantly above chance (p < 0.001)\n")
cat("  4. Large effect size (Cohen's h > 0.8)\n")
cat("  5. Substantial agreement (κ > 0.6)\n\n")

c1 <- acc_8type > 0.80
c2 <- all(stats_8type$accuracy > 0.50)
c3 <- test_8$p < 0.001
c4 <- test_8$h > 0.8
c5 <- kappa > 0.6

cat("RESULTS:\n")
cat(sprintf("  1. Accuracy > 80%%:     %s (%.1f%%)\n", ifelse(c1, "PASS ✓", "FAIL ✗"), acc_8type * 100))
cat(sprintf("  2. All types > 50%%:    %s (min: %.1f%%)\n", ifelse(c2, "PASS ✓", "FAIL ✗"), min(stats_8type$accuracy) * 100))
cat(sprintf("  3. p < 0.001:          %s (p = %.2e)\n", ifelse(c3, "PASS ✓", "FAIL ✗"), test_8$p))
cat(sprintf("  4. Cohen's h > 0.8:    %s (h = %.2f)\n", ifelse(c4, "PASS ✓", "FAIL ✗"), test_8$h))
cat(sprintf("  5. Kappa > 0.6:        %s (κ = %.2f)\n", ifelse(c5, "PASS ✓", "FAIL ✗"), kappa))
cat("\n")

n_pass <- sum(c(c1, c2, c3, c4, c5))

cat("-----------------------------------------------------------------------------\n")
if (n_pass == 5) {
  verdict <- "HYPOTHESIS FULLY SUPPORTED"
  cat(sprintf("VERDICT: ★★★ %s ★★★\n\n", verdict))
  cat("The GTI classifier demonstrates excellent performance across all criteria.\n")
} else if (n_pass >= 4) {
  verdict <- "HYPOTHESIS STRONGLY SUPPORTED"
  cat(sprintf("VERDICT: ★★ %s ★★\n\n", verdict))
  cat("The GTI classifier demonstrates robust performance with minor limitations.\n")
} else if (n_pass >= 3) {
  verdict <- "HYPOTHESIS PARTIALLY SUPPORTED"
  cat(sprintf("VERDICT: ★ %s ★\n\n", verdict))
  cat("The GTI classifier shows promise but requires improvement.\n")
} else {
  verdict <- "HYPOTHESIS NOT SUPPORTED"
  cat(sprintf("VERDICT: %s\n\n", verdict))
  cat("The GTI classifier does not meet validation criteria.\n")
}
cat("-----------------------------------------------------------------------------\n\n")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("                        GENERATING VISUALIZATIONS\n")
cat("=============================================================================\n\n")

theme_gti <- theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# Color palette (8 types)
type_colors <- c(
  "Trust_Game" = "#2E86AB",
  "Coordination_Game" = "#27AE60",
  "Battle_of_the_Sexes" = "#8E44AD",
  "Hero" = "#E74C3C",
  "Compromise" = "#F39C12",
  "Chicken" = "#E67E22",
  "Deadlock" = "#95A5A6",
  "Prisoners_Dilemma" = "#34495E"
)

# Figure 1: Accuracy Bar Plot (equivalent to COI boxplot)
cat("Creating accuracy barplot...\n")

fig1_data <- stats_8type %>%
  left_join(prf_8type, by = "type") %>%
  arrange(desc(accuracy))

fig1 <- ggplot(fig1_data, aes(x = reorder(type, accuracy), y = accuracy, fill = type)) +
  geom_bar(stat = "identity", alpha = 0.85) +
  geom_errorbar(aes(ymin = ci_lower, ymax = pmin(ci_upper, 1)), width = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#E74C3C", alpha = 0.7) +
  geom_hline(yintercept = 1/8, linetype = "dotted", color = "gray50") +
  geom_text(aes(label = sprintf("%.0f%%", accuracy * 100)), 
            hjust = -0.2, size = 3.5) +
  scale_fill_manual(values = type_colors) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.15)) +
  coord_flip() +
  labs(
    title = "GTI Classification Accuracy by Game Type",
    subtitle = sprintf("Overall: %.1f%% | Macro F1: %.1f%% | κ = %.2f", 
                       acc_8type * 100, macro_f1_8 * 100, kappa),
    x = NULL,
    y = "Accuracy",
    caption = "Error bars: 95% CI | Dashed: 50% | Dotted: chance (12.5%)"
  ) +
  theme_gti +
  theme(legend.position = "none")

ggsave(file.path(output_dir, "gti_fig1_accuracy.png"), fig1, width = 10, height = 6, dpi = 300)
ggsave(file.path(output_dir, "gti_fig1_accuracy.pdf"), fig1, width = 10, height = 6)
cat("  Saved: gti_fig1_accuracy.png/pdf\n")

# Figure 2: Confusion Matrix (equivalent to density plot showing distributions)
cat("Creating confusion matrix heatmap...\n")

confusion_df <- results_df %>%
  count(expected, predicted) %>%
  group_by(expected) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

fig2 <- ggplot(confusion_df, aes(x = predicted, y = expected, fill = prop)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = ifelse(n > 0, n, "")), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "white", mid = "#FEE08B", high = "#1A9850", 
                       midpoint = 0.5, limits = c(0, 1),
                       labels = percent_format()) +
  labs(
    title = "GTI Classification Confusion Matrix",
    subtitle = "Numbers show counts; color shows row proportion (recall)",
    x = "Predicted Type",
    y = "Expected (True) Type",
    fill = "Proportion"
  ) +
  theme_gti +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "gti_fig2_confusion.png"), fig2, width = 10, height = 8, dpi = 300)
cat("  Saved: gti_fig2_confusion.png\n")

# Figure 3: Strategic Family Breakdown (equivalent to mechanism breakdown)
cat("Creating strategic family breakdown...\n")

family_stats <- results_df %>%
  filter(!is.na(family_expected)) %>%
  group_by(family_expected) %>%
  summarise(
    n = n(),
    accuracy = mean(correct_family),
    .groups = "drop"
  )

family_colors <- c(
  "Conflict" = "#E74C3C",
  "Coordination" = "#27AE60",
  "Trust" = "#2E86AB",
  "Sacrifice" = "#8E44AD",
  "Negotiation" = "#F39C12"
)

fig3 <- ggplot(family_stats, aes(x = reorder(family_expected, -accuracy), 
                                  y = accuracy, fill = family_expected)) +
  geom_bar(stat = "identity", alpha = 0.85) +
  geom_text(aes(label = sprintf("%.0f%%\n(n=%d)", accuracy * 100, n)), 
            vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = family_colors) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.15)) +
  labs(
    title = "GTI Accuracy by Strategic Family",
    subtitle = sprintf("Family-level accuracy: %.1f%%", acc_family * 100),
    x = "Strategic Family",
    y = "Accuracy",
    caption = "Families group games with similar strategic structure"
  ) +
  theme_gti +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave(file.path(output_dir, "gti_fig3_families.png"), fig3, width = 8, height = 6, dpi = 300)
cat("  Saved: gti_fig3_families.png\n")

# Figure 4: Precision-Recall by Type
cat("Creating precision-recall plot...\n")

fig4 <- ggplot(prf_8type, aes(x = recall, y = precision, color = type, size = tp + fp + fn)) +
  geom_point(alpha = 0.8) +
  geom_text(aes(label = type), hjust = -0.1, vjust = -0.5, size = 3, show.legend = FALSE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50", alpha = 0.5) +
  scale_color_manual(values = type_colors) +
  scale_size_continuous(range = c(3, 10), name = "Support") +
  scale_x_continuous(labels = percent_format(), limits = c(0, 1.1)) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.1)) +
  labs(
    title = "Precision vs Recall by Game Type",
    subtitle = sprintf("Macro P: %.1f%% | Macro R: %.1f%% | Macro F1: %.1f%%",
                       macro_precision_8 * 100, macro_recall_8 * 100, macro_f1_8 * 100),
    x = "Recall (Sensitivity)",
    y = "Precision (PPV)"
  ) +
  theme_gti +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "gti_fig4_precision_recall.png"), fig4, width = 10, height = 8, dpi = 300)
cat("  Saved: gti_fig4_precision_recall.png\n")

# Figure 5: Taxonomy Comparison (8 vs 9 vs Family)
cat("Creating taxonomy comparison...\n")

taxonomy_comp <- data.frame(
  taxonomy = c("8-type\n(Strategic)", "9-type\n(R-G Core)", "5-Family\n(Aggregate)"),
  accuracy = c(acc_8type, acc_9type, acc_family),
  n_classes = c(8, 9, 5)
) %>%
  mutate(
    se = sqrt(accuracy * (1 - accuracy) / n_total),
    ci_lower = accuracy - 1.96 * se,
    ci_upper = accuracy + 1.96 * se,
    chance = 1 / n_classes
  )

fig5 <- ggplot(taxonomy_comp, aes(x = taxonomy, y = accuracy, fill = taxonomy)) +
  geom_bar(stat = "identity", alpha = 0.8, width = 0.6) +
  geom_errorbar(aes(ymin = ci_lower, ymax = pmin(ci_upper, 1)), width = 0.2) +
  geom_point(aes(y = chance), shape = 4, size = 4, stroke = 2) +
  geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)), vjust = -0.8, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("#2E86AB", "#E74C3C", "#27AE60")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.15)) +
  labs(
    title = "GTI Accuracy Across Taxonomy Granularities",
    subtitle = "X marks chance level for each taxonomy",
    x = NULL,
    y = "Accuracy",
    caption = "Error bars: 95% CI"
  ) +
  theme_gti +
  theme(legend.position = "none")

ggsave(file.path(output_dir, "gti_fig5_taxonomy.png"), fig5, width = 8, height = 6, dpi = 300)
cat("  Saved: gti_fig5_taxonomy.png\n")

cat("\n")

# -----------------------------------------------------------------------------
# Export Results
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("                           ANALYSIS COMPLETE\n")
cat("=============================================================================\n\n")

# Comprehensive results object
results_export <- list(
  metadata = list(
    timestamp = as.character(Sys.time()),
    input_file = input_file,
    n_total = n_total
  ),
  accuracy = list(
    acc_8type = acc_8type,
    acc_9type = acc_9type,
    acc_family = acc_family
  ),
  macro_metrics = list(
    precision_8 = macro_precision_8,
    recall_8 = macro_recall_8,
    f1_8 = macro_f1_8,
    precision_9 = macro_precision_9,
    recall_9 = macro_recall_9,
    f1_9 = macro_f1_9
  ),
  hypothesis_tests = list(
    vs_chance_8_p = test_8$p,
    vs_chance_8_h = test_8$h,
    vs_chance_9_p = test_9$p,
    vs_chance_9_h = test_9$h,
    z_8v9 = z_stat,
    p_8v9 = z_p,
    h_8v9 = h_8v9,
    chi2_homogeneity = as.numeric(chi_result$statistic),
    p_homogeneity = chi_result$p.value,
    cramers_v = as.numeric(cramers_v),
    kappa = kappa
  ),
  validation = list(
    criterion_1_acc80 = c1,
    criterion_2_all50 = c2,
    criterion_3_p001 = c3,
    criterion_4_h08 = c4,
    criterion_5_kappa06 = c5,
    n_criteria_passed = n_pass,
    verdict = verdict
  ),
  per_type_8 = setNames(
    lapply(1:nrow(stats_8type), function(i) {
      list(
        n = stats_8type$n[i],
        accuracy = stats_8type$accuracy[i],
        precision = prf_8type$precision[prf_8type$type == stats_8type$type[i]],
        recall = prf_8type$recall[prf_8type$type == stats_8type$type[i]],
        f1 = prf_8type$f1[prf_8type$type == stats_8type$type[i]]
      )
    }),
    stats_8type$type
  )
)

results_path <- file.path(output_dir, "gti_comprehensive_results.json")
write_json(results_export, results_path, pretty = TRUE, auto_unbox = TRUE)
cat("Results saved to:", results_path, "\n\n")

cat("KEY FINDINGS:\n")
cat(sprintf("  • Total classifications: %d\n", n_total))
cat(sprintf("  • 8-type accuracy: %.1f%% (F1: %.1f%%)\n", acc_8type * 100, macro_f1_8 * 100))
cat(sprintf("  • 9-type accuracy: %.1f%% (F1: %.1f%%)\n", acc_9type * 100, macro_f1_9 * 100))
cat(sprintf("  • Cohen's Kappa: %.2f (%s agreement)\n", kappa,
            ifelse(kappa < 0.4, "Fair", ifelse(kappa < 0.6, "Moderate", 
              ifelse(kappa < 0.8, "Substantial", "Almost Perfect")))))
cat(sprintf("  • Verdict: %s (%d/5 criteria)\n\n", verdict, n_pass))

cat("OUTPUT FILES:\n")
cat(sprintf("  • %s\n", file.path(output_dir, "gti_fig1_accuracy.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "gti_fig2_confusion.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "gti_fig3_families.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "gti_fig4_precision_recall.png")))
cat(sprintf("  • %s\n", file.path(output_dir, "gti_fig5_taxonomy.png")))
cat(sprintf("  • %s\n", results_path))
cat("\n")

cat("=============================================================================\n")
cat("                              END OF ANALYSIS\n")
cat("=============================================================================\n")
