# System requirements (R 4.1 w/ cairo, Python 3 w/ iTRADE, perl) ---------------
stopifnot(version$major >= 4)
stopifnot(version$minor >= 1)
stopifnot(capabilities("cairo"))
stopifnot(length(system("python --help", intern = TRUE)) > 0)
stopifnot(length(system("perl --help", intern = TRUE)) > 0)

# Warnings and errors ----------------------------------------------------------
options(
  warn = 2,
  showWarnCalls = TRUE,
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE,
  rlang_backtrace_on_error = "branch"
)

# Reproducibility --------------------------------------------------------------
set.seed(2301)

# Required packages ------------------------------------------------------------
required_packages <- c(
  "tidyverse",
  # > grep -v -E "^#" itrade/util/plots.R | grep -oP "[\w\.]+(?=::)" | sort -u \
  # >   | sed 's/^.*$/"\0",/' | tr "\n" " " | sed 's/, $/\n/' | fold -w 78 -s
  "arrow", "base64enc", "circlize", "ComplexHeatmap", "cowplot", "farver",
  "ggbeeswarm", "gghighlight", "ggpubr", "glue", "grid", "openxlsx", "plyr",
  "R.devices", "rjson", "scales", "tools", "withr", "xml2"
)
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
install.packages(missing_packages)

options(tidyverse.quiet = TRUE)
library(tidyverse)

# CNN constants ----------------------------------------------------------------

cnn_epochs <- 200

cnn_modes <- c(
  "train" = "train",
  "val" = "validation"
)

cnn_metrics <- c(
  "loss" = "epoch_loss",
  "acc" = "epoch_categorical_accuracy",
  "auroc" = "epoch_auc"
)

# Material ---------------------------------------------------------------------

# Cell lines (compare Table S-II)
cell_line_expected_hits <- list(
  "INF_R_153" = c(
    "Entrectinib", "Larotrectinib", "Merestinib"
  ),
  "BT-40" = c(
    "Vemurafenib", "Dabrafenib",
    "Trametinib", "Cobimetinib", "Selumetinib"
  ),
  "HD-MB03" = c(
    "Entinostat", "Vorinostat", "Panobinostat"
  ),
  "NCI-H3122" = c(
    "Lorlatinib"
  ),
  "SJ-GBM2" = c(
    "Cabozantinib", "Foretinib",
    "Entinostat", "Vorinostat", "Panobinostat"
  ),
  "SMS-KCNR" = c(
    "Lorlatinib",
    "Venetoclax"
  )
)

cell_lines <- names(cell_line_expected_hits)

# Screens IDs
cell_line_sids <- c(
  "INF_R_153_V2_DS1",
  "INF_R_153_V3_DS1",
  "BT-40_V2_DS1",
  "BT-40_V3_DS1",
  "BT-40_V3_DS2",
  "HD-MB03_V1_DS1",
  "HD-MB03_V1_DS2",
  "HD-MB03_V2_DS1",
  "HD-MB03_V2_DS2",
  "NCI-H3122_V2_DS1",
  "SJ-GBM2_V2_DS1",
  "SMS-KCNR_V1_DS1",
  "SMS-KCNR_V2_DS1",
  "SMS-KCNR_V2_DS2"
)

patient_sids <- c(
  "INF_R_1021_relapse1_V1_DS2",
  "INF_R_1025_primary_V2_DS1",
  "INF_R_1123_primary_V1_DS1"
)

is_patient_sid <- function(sid) {
  return(any(patient_sids == sid))
}

sids <- c(cell_line_sids, patient_sids)

transfer_epochs <- str_pad(seq(20, 200, 20), 3, pad = "0")
transfer_sid_prefix <- "INF_R_153_V2_DS1-"
transfer_sids <- paste0(
  transfer_sid_prefix,
  c(
    "NoPretraining",
    "OnlyImageNet",
    paste0("Phase1Epoch-model_checkpoint_e", transfer_epochs, "_b190", ".tf")
  )
)

# Modality abbreviations as used by iTRADE and iTReX
.itrade_modality_labels <- c(
  "Phase2" = "CNN (II)",
  "Phase3" = "CNN (III)"
)
.itrex_modality_labels <- c(
  "Met" = "Metabolic",
  "MoS" = "Mean of Stack",
  "iT2" = "CNN (II)",
  "iT3" = "CNN (III)"
)

.modality_labels <- c(.itrade_modality_labels, .itrex_modality_labels)

get_modality_labels <- function(modality) {
  modality_labels <- factor(
    unname(.modality_labels[as.character(modality)]),
    levels = unique(.itrex_modality_labels)
  )
  assert(!any(is.na(levels(modality_labels))))

  # Remove unused levels
  unused_levels <- setdiff(levels(modality_labels), modality_labels)
  for (unused_level in unused_levels) {
    levels(modality_labels) <- gsub(
      unused_level, "", levels(modality_labels),
      fixed = TRUE
    )
  }

  modality_labels
}

itrade_modalities <- names(.itrade_modality_labels)
itrex_modalities <- names(.itrex_modality_labels)

itrade_to_itrex <- \(modality) str_replace(modality, "Phase", "iT")

# Helper functions -------------------------------------------------------------

assert <- function(expr, ...) {
  if (!expr) stop(..., call. = FALSE)
}

glue <- function(...) {
  do.call(file.path, lapply(list(...), glue::glue, .envir = parent.frame()))
}

copy_file <- function(in_file, out_file) {
  assert(file.copy(in_file, out_file, overwrite = TRUE))
}

# Plots ------------------------------------------------------------------------

# Save ggplot as pdf and png
pdf_png <- function(filestem, plot,
                    width = 6, height = 4,
                    scale = 1, unit = "in",
                    dpi = 600,
                    rm_pagegroup = TRUE,
                    rm_dates = TRUE) {
  pdf_file <- glue("{filestem}.pdf")
  png_file <- glue("{filestem}.png")

  save_args <- list(
    plot = plot,
    scale = scale, width = width, height = height, units = unit, dpi = dpi
  )
  # Use Cairo for
  # - Unicode characters ('='), https://stackoverflow.com/a/12775087
  # - anti-aliasing, https://stackoverflow.com/a/34759363
  do.call(ggsave, c(pdf_file, device = cairo_pdf, save_args))
  do.call(ggsave, c(png_file, type = "cairo", save_args))

  # test using pdf_png("tmp", 0)
  if (rm_pagegroup) {
    lines <- c(
      "   /Group <<",
      "      /Type /Group",
      "      /S /Transparency",
      "      /I true",
      "      /CS /DeviceRGB",
      "   >>"
    )
    from <- paste0(lines, collapse = "\\n")
    to <- strrep(" ", nchar(from) - length(lines) + 1)
    pdf_sed(pdf_file, from, to)
  }

  if (rm_dates) {
    froms <- c()
    tos <- c()
    field <- "/(Creation|Mod)Date"
    for (is_cairo in c(FALSE, TRUE)) {
      if (is_cairo) {
        # Cairo PDF files do not have the trailing ', see
        # https://github.com/freedesktop/cairo/blob/3a03c1ba4bc3
        # /src/cairo-pdf-interchange.c#L1440
        value <- "\\(D:[0-9]{14}.[0-9]{2}.[0-9]{2}(.?)\\)"
        to <- "\\1(D:19700101000000+00'00\\3)"
      } else {
        # Non-Cairo PDF files do not have the time zone
        value <- "\\(D:[0-9]{14}\\)"
        to <- "\\1(D:19700101000000)"
      }
      froms <- append(froms, glue("({field}\\s*){value}"))
      tos <- append(tos, to)
    }
    pdf_sed(pdf_file, froms, tos)
  }
}

pdf_sed <- function(pdf_file, froms, tos) {
  size_before <- file.info(pdf_file)$size
  md5_before <- tools::md5sum(pdf_file)
  for (from_to in Map(c, froms, tos)) {
    system(glue('perl -0pe "s#{from_to[1]}#{from_to[2]}#" -i "{pdf_file}"'))
  }
  size_after <- file.info(pdf_file)$size
  md5_after <- tools::md5sum(pdf_file)
  assert(size_before == size_after, "PDF file size has changed.")
  assert(md5_before != md5_after, "PDF file has not changed.")
  return(invisible())
}

# Dimensions
# \textwidth         43pc   % 2 x 21pc + 1pc = 43pc (ieeecolor.cls)
# In TeX one pica is ​12/72.27 of an inch.
# https://en.wikipedia.org/wiki/Pica_(typography)
ieee_full_width <- 43 * 12 / 72.27
ieee_half_width <- 21 * 12 / 72.27

# Plot elements ----------------------------------------------------------------
diagonal <- geom_abline(
  slope = 1,
  color = "lightgrey",
)

dss_10_x <- geom_vline(
  xintercept = 10,
  color = "red",
  linetype = "dashed",
)

dss_10_y <- geom_hline(
  yintercept = 10,
  color = "red",
  linetype = "dashed",
)

blue_scatter_point <- geom_point(
  size = 0.5,
  color = "#21698D",
)

white_violin <- geom_violin(
  color = "white",
  draw_quantiles = 0.5,
)

modality_swarm <- function(cex, size) {
  ggbeeswarm::geom_beeswarm(
    priority = "random",
    cex = cex,
    size = size,
  )
}

white_mean_diamond <- stat_summary(
  fun = "mean",
  geom = "point",
  color = "white",
  shape = 23,
)

r_rho <- function(lower, upper, padding = 0, magic = 0) {
  # magic == 0: middle-top/right-middle alignment
  # magic != 0: bottom alignment
  #             valign top due to depth of \rho
  #             magic to push text down
  mid <- (lower + upper) / 2
  list(
    ggpubr::stat_cor(
      aes(label = ..r.label..),
      size = 2.5,
      hjust = if (magic) "left" else "center",
      vjust = "top",
      label.x = if (magic) lower - padding else mid,
      label.y = if (magic) lower - magic else upper + padding,
    ),
    ggpubr::stat_cor(
      aes(label = ..r.label..),
      size = 2.5,
      method = "spearman",
      cor.coef.name = "rho",
      hjust = if (magic) "right" else "center",
      vjust = "top",
      label.x = upper + padding,
      label.y = if (magic) lower - magic else mid,
      angle = if (magic) 0 else -90,
    )
  )
}

theme <- list(
  theme_bw(),
  theme(
    panel.spacing.x = unit(0, "mm"),
    panel.spacing.y = unit(0, "mm"),
    plot.margin = unit(c(0, 0, 0, 0), "mm")
  )
)

small_facet <- theme(
  strip.text = element_text(size = 7.5),
)

# Plot arrangements ------------------------------------------------------------

margin_cell_lines_sid <- "(all cell lines)"
margin_patients_sid <- "(all patients)"

modality_sample_grid <- facet_grid(
  cols = vars(get_modality_labels(modality)),
  rows = vars(sid),
)

get_margin_distributions <- function(data, add = FALSE) {
  data_margin <- data
  data_margin$sid <- factor(ifelse(
    Vectorize(is_patient_sid)(data$sid),
    margin_patients_sid,
    margin_cell_lines_sid
  ))
  if (!add) {
    return(data_margin)
  }
  return(rbind(data_margin, data))
}

plot_pcc <- function(plots_dir, df, col1, col2, plot_desc, plot_name) {
  num_pcc <- by(df, df[c("modality", "sid")], \(df) {
    cor(df[, col1], df[, col2], method = "pearson")
  })
  df_pcc <- as.data.frame.table(num_pcc, responseName = "cc_pearson")

  df_pcc$modality_label <- get_modality_labels(df_pcc$modality)

  p <- ggplot(
    df_pcc,
    aes(
      x = modality_label,
      y = cc_pearson,
      fill = modality_label,
    )
  ) +
    white_violin +
    modality_swarm(
      cex = 3,
      size = 3 / 4
    ) +
    white_mean_diamond +
    coord_cartesian(
      ylim = c(0, 1),
    ) +
    scale_x_discrete(
      drop = TRUE,
    ) +
    scale_fill_discrete(
      drop = FALSE,
    ) +
    labs(
      x = "Modality",
      y = glue("Pearson's R ({plot_desc})"),
    ) +
    guides(
      fill = "none",
    ) +
    theme +
    NULL

  pdf_png(glue(plots_dir, plot_name), p, ieee_half_width, 1.5, 1.25)
}

################################################################################

# [all] Unzip iTReX results ----------------------------------------------------
unzip_itrex_results_files <- function(itrex_dir, force = FALSE) {
  folders <- list.dirs(itrex_dir, recursive = FALSE)
  if (length(folders) > 0 && !force) {
    return()
  }

  zip_files <- list.files(
    itrex_dir, "^iTReX-Results.*\\.zip$",
    full.names = TRUE
  )
  sapply(zip_files, unzip, exdir = itrex_dir)
}


# [all] Read layout file -------------------------------------------------------
read_layout <- function(data_dir) {
  layout_file <- glue(data_dir, "Layouts", "Layout_DS.csv")
  df_layout <- read.csv(layout_file)

  return(df_layout)
}


# [all] Read drug annotations --------------------------------------------------
read_drugs <- function(data_dir) {
  drugs_file <- glue(data_dir, "Layouts", "Drugs.csv")
  df_drugs <- read.csv(drugs_file)
  df_drugs$GroupAbbr <- str_replace(df_drugs$GroupAbbr, "--", "−")
  df_drugs$GroupName <- str_replace(df_drugs$GroupName, "/", "/\n")
  df_drugs$TopGroupAbbr <- str_match(df_drugs$GroupAbbr, "^[^-]+")

  return(df_drugs)
}


# [cnn2] Search and export outliers in raw viabilities -------------------------
read_viabilities <- function(df_layout, itrade_dir, modality = "Phase2") {
  df_viab <- data.frame()
  for (sid in sids) {
    txt_files <- get_data_txt_files(itrade_dir, sid, modality)
    for (file in txt_files) {
      contents <- read_file(file)
      df <- read.csv(
        text = contents,
        sep = "\t",
        check.names = FALSE,
        row.names = 1,
      )
      df <- pivot_longer(
        rownames_to_column(df, "row"),
        cols = colnames(df),
        names_to = "col"
      )
      df <- na.omit(df)

      df$plate <- strtoi(str_match(file, "(?<=_P)[123](?=\\.txt$)"))
      df$sid <- sid
      df$pid <- gsub("_DS[0-9]$", "", sid)

      df_viab <- rbind(df_viab, df)
    }
  }

  assert(nrow(df_viab) == length(sids) * 3 * 14 * 22)

  df_viab$col <- as.numeric(df_viab$col)
  df_viab$plate <- paste0("P", df_viab$plate)

  df_viab <- merge(
    df_viab, df_layout,
    by.x = c("plate", "row", "col"), by.y = c("Plate", "Row", "Column"),
  )

  df_viab$is_control <- df_viab$WellType %in% c("neg", "pos")

  n_controls <- sum(df_viab$is_control)
  n_controls_sid <- sum(df_layout$WellType %in% c("neg", "pos"))
  assert(n_controls == length(sids) * n_controls_sid)

  return(df_viab)
}

get_data_txt_files <- function(itrade_dir, sid, modality) {
  sid_data_dir <- glue(itrade_dir, "{modality}-{sid}", "predict")
  plate_file_pattern <- glue("{sid}_P\\d\\.txt")
  txt_files <- list.files(sid_data_dir, plate_file_pattern, full.names = TRUE)
  assert(length(txt_files) > 0, "No data for ", sid)
  assert(length(txt_files) == 3, "Incomplete data for ", sid)
  txt_files <- sort(txt_files)

  return(txt_files)
}


# [cnn2] Plot raw data histogram -----------------------------------------------
plot_viabilities <- function(df_viab, plots_dir) {
  df_control <- df_viab[df_viab$is_control, ]
  p <- ggplot(
    df_viab,
    aes(
      x = value / 1000,
      fill = ..x..,
    )
  ) +
    geom_histogram(
      bins = 51,
    ) +
    geom_histogram(
      bins = 51,
      data = df_control,
      color = "black",
    ) +
    scale_fill_gradient2(
      low = "red",
      mid = "grey75",
      high = "blue",
      midpoint = 0.5,
    ) +
    labs(
      x = "Raw cell viability",
      y = "Frequency",
    ) +
    guides(
      fill = "none",
    ) +
    theme +
    theme(
      plot.margin = margin(t = 1, l = 1),
    ) +
    NULL

  pdf_png(glue(plots_dir, "raw_histogram"), p, ieee_half_width, 1.5, 1.25)
}


# [cnn2] Viability outliers for TeX --------------------------------------------
plot_viability_outliers <- function(df, data_dir, plots_dir, threshold = 110) {
  df <- df[order(+df$value), ]

  df$compare <- df$value
  last <- ""
  for (mode in c("min", "max")) {
    if (mode == "max") {
      df$compare <- -df$compare
      threshold <- threshold - 1000
      last <- "1"
    }
    wells <- df[df$compare < threshold, ]
    wells$well <- paste0(wells$row, formatC(wells$col, width = 2, flag = "0"))
    wells$value_str <- formatC(wells$value / 1000, digits = 2, format = "f")
    wells$last <- ifelse(seq_len(nrow(wells)) == nrow(wells), last, "")
    wells$percent <- ifelse(seq_len(nrow(wells)) %% 4, "%", "")
    wells$cmd <- glue(paste0(
      "\\minMax{{{mode}}}{{{wells$sid}}}{{{wells$plate}}}{{{wells$well}}}",
      "{{{wells$value}}}{{{wells$value_str}}}{{{wells$last}}}",
      "{{\\SI{{{wells$Concentration}}}{{\\nano\\Molar}} {wells$Treatment}}}",
      "{wells$percent}"
    ))
    tex_commands <- paste(wells$cmd, collapse = "\n")
    cat(tex_commands, file = glue(plots_dir, "outlier_plot_{mode}.tex"))

    wells$tiff <- glue(
      # https://github.com/tidyverse/glue/issues/14
      data_dir, "Images", "{wells$sid}",
      "{wells$sid}_{wells$plate}_{wells$well}_proj_224x224.tif"
    )
    wells$png <- glue(
      plots_dir, "outlier_{mode}",
      "{wells$sid}_{wells$plate}_{wells$well}_{wells$value}.png"
    )
    wells$scale <- ifelse(wells$sid == "HD-MB03_V1_DS2", 1, 5)
    for (i in seq_len(nrow(wells))) {
      w <- wells[i, ]
      tiff2png_call <- glue(
        'python -m itrade.util.images "{w$tiff}" "{w$png}" {w$scale}'
      )
      system(tiff2png_call, intern = TRUE)
    }
  }
}


# [all] Scatter plots of replicate viabilities ---------------------------------
plot_replicate_viabilities <- function(itrex_dir, plots_dir) { # nolint(cyclocomp_linter)
  data <- data.frame()
  for (sid in sids) {
    for (modality in itrex_modalities) {
      modality_sid <- glue("{modality}-{sid}")
      rds_file <- glue(
        itrex_dir, modality_sid, "pre_process", "{modality_sid}_mono.rds"
      )
      rds_data <- readRDS(rds_file)

      screen_data <- do.call("rbind", rds_data$splitlist)
      assert(all(startsWith(screen_data$PlateDisplayName, modality_sid)))

      screen_data <- merge(
        screen_data[screen_data$Replicate == 1, ],
        screen_data[screen_data$Replicate == 2, ],
        by = c("Treatment", "dose"), suffixes = c("", "_rep")
      )

      screen_data <- screen_data[c("viability", "viability_rep")]
      screen_data$sid <- sid
      screen_data$modality <- modality

      data <- rbind(data, screen_data)
    }
  }
  data$sid <- factor(data$sid, levels = sids)

  plot_pcc(
    plots_dir, data, "viability", "viability_rep", "viability",
    "PlateScatPearson"
  )

  for (plot_margin in FALSE:TRUE) {
    if (plot_margin) {
      data <- get_margin_distributions(data)
    }

    padding <- 0.5

    p <- ggplot(
      data,
      aes(
        x = viability,
        y = viability_rep,
      )
    ) +
      diagonal +
      blue_scatter_point +
      modality_sample_grid +
      r_rho(
        0, 1, padding,
        magic = 0.25
      ) +
      coord_fixed(
        ratio = 1,
        xlim = c(0 - padding, 1 + padding),
        ylim = c(0 - padding, 1 + padding),
      ) +
      scale_x_continuous(
        labels = scales::percent,
        breaks = (if (plot_margin) c(0, 1) else c(0, 0.5, 1)),
      ) +
      scale_y_continuous(
        labels = scales::percent,
        breaks = c(0, 0.5, 1),
      ) +
      labs(
        x = "Percentage cell viability (replicate 1)",
        y = "Percentage cell viability (replicate 2)",
      ) +
      theme +
      (if (plot_margin) NULL else small_facet) +
      NULL

    f <- glue(plots_dir, "PlateScatter{plot_margin}")
    if (plot_margin) {
      pdf_png(f, p, ieee_half_width, 2, 1.25)
    } else {
      pdf_png(f, p, ieee_half_width, 8.26772, scale = 1 / 0.3)
    }
  }
}


# [all] PNG files of treatment response controls -------------------------------
plot_treatment_resp_controls <- function(itrex_dir, plots_dir) {
  trc_dir <- glue(plots_dir, "trc")
  if (!dir.exists(trc_dir)) {
    dir.create(trc_dir, recursive = TRUE)
  }
  for (sid in sids) {
    for (modality in itrex_modalities) {
      modality_sid <- glue("{modality}-{sid}")
      html <- glue(itrex_dir, modality_sid, "{modality_sid}_QC.html")
      png <- glue(trc_dir, "{modality_sid}.png")
      html_to_png_helper(html, png)
    }
  }
}

html_to_png_helper <- function(html_file, png_file) {
  xpath <- "//./div[@id='therapy-response-control-curve']/p/img"
  attr <- "src"
  prefix <- "data:image/png;base64,"

  html <- xml2::read_html(html_file)
  img <- xml2::xml_find_first(html, xpath)
  src <- xml2::xml_attr(img, attr)
  png <- str_remove(src, prefix)
  raw <- base64enc::base64decode(png)
  writeBin(raw, png_file)
}


# [all] Read Z' values from QC files -------------------------------------------
read_zprime <- function(itrex_dir) { # nolint(cyclocomp_linter)
  # Read Z' values
  df_zprime <- data.frame()
  for (sid in sids) {
    for (modality in itrex_modalities) {
      zprime_r <- read_zprime_helper(itrex_dir, sid, modality)
      assert(all((as.numeric(zprime_r) != 0) | (zprime_r == "NaN")))
      zprime_r <- as.numeric(zprime_r)

      df <- data.frame(
        sid = sid,
        modality = modality,
        plate = 1:3,
        zprime_r = zprime_r
      )
      df_zprime <- rbind(df_zprime, df)
    }
  }

  return(df_zprime)
}

read_zprime_helper <- function(itrex_dir, sid, modality) {
  modality_sid <- glue("{modality}-{sid}")
  qc_file <- glue(itrex_dir, modality_sid, "{modality_sid}_QC.html")
  lines <- readLines(qc_file)

  # check SID line
  sid_line <- lines[grepl(glue("<h2>{modality_sid}</h2>"), lines)]
  assert(length(sid_line) > 0)

  # remove everything up to table body
  lines <- tail(lines, -which(grepl("<thead>", lines)))
  lines <- tail(lines, -which(grepl("<tbody>", lines)))

  # search for table cells holding row names
  idxs <- which(grepl("[123]_zprime_r", lines))
  assert(length(idxs) == 3)
  assert(prod(idxs) > 0)

  # jump to adjacent table cell by skipping </td> and <td> lines
  zprime_r <- lines[idxs + 3]
  return(zprime_r)
}


# [all] Violin plot of Z' values -----------------------------------------------
plot_zprime <- function(df_zprime, plots_dir) {
  min_z <- -1
  arr_len <- 0.1
  # https://stackoverflow.com/a/29463136/880783
  outliers <- df_zprime %>%
    filter(zprime_r < min_z) %>%
    group_by(modality) %>%
    summarise(label = paste(round(zprime_r, 1), collapse = ", "))

  n_outliers <- table(df_zprime$modality[df_zprime$zprime_r < min_z])
  n_outliers <- n_outliers[n_outliers > 0]
  outlier_summaries <- paste(n_outliers[n_outliers > 2], "outliers")
  outliers$label[n_outliers > 2] <- outlier_summaries

  df_zprime$zprime_r[df_zprime$zprime_r < min_z] <- min_z - 1
  # https://github.com/eclarke/ggbeeswarm/issues/65
  df_zprime$zprime_r[!is.finite(df_zprime$zprime_r)] <- min_z - 1

  p <- ggplot(
    df_zprime,
    aes(
      x = get_modality_labels(modality),
      y = zprime_r,
      fill = as.factor(..x..),
    )
  ) +
    white_violin +
    modality_swarm(
      cex = 2,
      size = 1 / 2
    ) +
    white_mean_diamond +
    geom_text(
      data = outliers,
      aes(y = (1 - 1.5 * arr_len) * min_z, label = label),
      size = 3,
      vjust = "bottom",
      hjust = "center",
    ) +
    geom_segment(
      data = outliers,
      aes(xend = ..x.., y = (1 - arr_len) * min_z, yend = min_z),
      arrow = arrow(length = unit(0.03, "npc")),
    ) +
    coord_cartesian(
      ylim = c(min_z, 1),
    ) +
    labs(
      x = "Modality",
      y = "Plate Z'",
    ) +
    guides(
      fill = "none",
    ) +
    theme +
    NULL

  # collapsing to unique 'x' values
  withr::with_options(
    list(warn = 0),
    pdf_png(glue(plots_dir, "zprime"), p, ieee_half_width, 1.5, 1.25)
  )
}


# [cnn2] Read TensorBoard metrics ----------------------------------------------
read_metrics <- function(itrade_dir) {
  read_metrics_helper(
    itrade_dir,
    modality = "Phase2", sids = sids, epochs_exp = cnn_epochs
  )
}

read_metrics_helper <- function(itrade_dir, modality, sids, epochs_exp) {
  epochs_read <- if (epochs_exp > 0) 1 else cnn_epochs

  # Collect file names of v2 files
  all_v2 <- list()
  tb_dir <- glue(itrade_dir, "Board")
  for (sid in sids) {
    for (mode in cnn_modes) {
      sid_pattern <- glue("^{modality}-{sid}.*")
      tb_sid_dir <- list.files(tb_dir, sid_pattern, full.names = TRUE)
      assert(length(tb_sid_dir) > 0, "tb_sid_dir does not exist")
      assert(length(tb_sid_dir) == 1, "tb_sid_dir is not unique")

      tb_mode_dir <- list.files(tb_sid_dir, mode, full.names = TRUE)
      assert(length(tb_mode_dir) > 0, "tb_mode_dir does not exist")
      assert(length(tb_mode_dir) == 1, "tb_mode_dir is not unique")

      v2 <- list.files(tb_mode_dir, "\\.v2$", full.names = TRUE)
      v2 <- str_sort(v2, numeric = TRUE)
      v2 <- tail(v2, 1)
      all_v2[glue("{sid}_{mode}")] <- glue("'{v2}'")
    }
  }

  # Get metrics for all v2 files
  v2s <- paste(all_v2, collapse = " ")
  tb_call <- glue("python -m itrade.util.read_tb {epochs_exp} {v2s}")
  metrics_json <- system(tb_call, intern = TRUE)
  all_metrics <- rjson::fromJSON(metrics_json)
  names(all_metrics) <- names(all_v2)

  # Collect metrics in data frame
  df_metrics <- data.frame(matrix(
    nrow = length(sids) * epochs_read,
    ncol = length(cnn_modes) * length(cnn_metrics) + (epochs_read > 1)
  ))

  col_names <- unlist(outer(
    names(cnn_metrics), names(cnn_modes),
    FUN = \(metric, mode) paste(mode, metric, sep = "_")
  ))
  if (epochs_read > 1) {
    col_names <- c("step", col_names)
  }
  colnames(df_metrics) <- col_names
  df_metrics$sid <- rep(sids, each = epochs_read)

  for (sid in sids) {
    for (mode_name in names(cnn_modes)) {
      sid_mode <- glue("{sid}_{cnn_modes[mode_name]}")
      metrics <- all_metrics[[sid_mode]][cnn_metrics]
      assert(all(cnn_metrics %in% names(metrics)), "Incorrect metrics")
      col_names <- paste(mode_name, names(cnn_metrics), sep = "_")
      if (epochs_read > 1) {
        col_names <- c(col_names, "step")
        metrics$step <- 1:epochs_read
      }
      df_metrics[df_metrics$sid == sid, col_names] <- metrics
    }
  }

  return(df_metrics)
}


# [cnn2/met] TeX table of TensorBoard metrics and Z' values --------------------
tabulate_metrics_and_zprime <- function(df_metrics, df_zprime, plots_dir) {
  df_zprime <- pivot_wider(
    df_zprime,
    id_cols = c("sid", "modality"), names_from = "plate",
    names_prefix = "zprime_r", values_from = "zprime_r",
  )
  df_zprime <- full_join(
    subset(df_zprime, modality == "iT2", select = -modality),
    subset(df_zprime, modality == "Met", select = -modality),
    by = "sid",
    suffix = c("_cnn2", "_met"),
  )

  train_qc <- full_join(df_metrics, df_zprime, by = "sid")
  train_qc <- select(train_qc, "sid", everything())

  add_color <- function(var, cmp, th, col) {
    f <- if (cmp == ">") -1 else 1
    var <- sym(var)
    train_qc <- mutate(train_qc, !!var := ifelse(
      ((f * suppressWarnings(as.numeric(!!var))) < (f * th)) %in% TRUE,
      paste0("\\color{", col, "} ", !!var),
      !!var
    ))
    assign("train_qc", train_qc, parent.frame())
  }

  add_color("train_loss", ">", 0.545, "bad")
  add_color("train_loss", "<", 0.525, "good")
  add_color("train_acc", "<", 0.955, "bad")
  add_color("train_acc", ">", 0.985, "good")
  add_color("train_auroc", "<", 0.955, "bad")
  add_color("train_auroc", ">", 0.985, "good")

  add_color("val_loss", ">", 0.585, "bad")
  add_color("val_loss", "<", 0.535, "good")
  add_color("val_acc", "<", 0.7, "bad")
  add_color("val_acc", ">", 0.9, "good")
  add_color("val_auroc", "<", 0.7, "bad")
  add_color("val_auroc", ">", 0.9, "good")

  for (colname in setdiff(colnames(df_zprime), "sid")) {
    add_color(colname, "<", 0.0, "bad")
    add_color(colname, ">", 0.5, "good")
  }

  train_qc_str <- paste0(
    do.call(paste, c(train_qc, sep = " & ")),
    " \\\\\n",
    collapse = ""
  )
  cat(train_qc_str, file = glue(plots_dir, "train_qc_table.tex"))
}


# [cnn2] Curve plots for transfer learning experiments -------------------------
read_transfer_metrics <- function(itrade_dir) {
  read_metrics_helper(
    itrade_dir,
    modality = "TransferStudy-Phase2",
    sids = transfer_sids,
    epochs_exp = 0
  )
}

plot_transfer_metrics <- function(df_metrics, plots_dir) {
  df_metrics$epoch <- df_metrics$sid
  epoch_values <- unique(df_metrics$epoch)
  epoch_labels <- epoch_values
  epoch_num <- as.numeric(str_match(epoch_values, "_e(\\d+)_")[, 2])

  epoch_labels[!is.na(epoch_num)] <- paste(
    "ImageNet + ", epoch_num[!is.na(epoch_num)], "Epochs"
  )
  epoch_labels <- sub(transfer_sid_prefix, "", epoch_labels)
  epoch_labels <- sub("^No", "No ", epoch_labels)
  epoch_labels <- sub("^Only", "Only ", epoch_labels)
  names(epoch_labels) <- epoch_values

  df_metrics$epoch <- recode_factor(df_metrics$epoch, !!!epoch_labels)

  p <- ggplot(
    df_metrics,
    aes(x = step, y = val_loss, group = sid, color = epoch),
  ) +
    geom_line() +
    labs(
      color = "Pretraining/Phase-I Epochs",
      x = "Phase-II Epochs",
      y = "Phase-II Validation Loss",
    ) +
    coord_cartesian(
      ylim = c(0.53, 0.71),
      expand = FALSE,
    ) +
    theme +
    theme(
      legend.title = element_text(size = rel(0.8)),
      legend.text = element_text(size = rel(0.8)),
    ) +
    NULL

  pdf_png(glue(plots_dir, "TransferMetrics"), p, ieee_half_width, 2.75, 1.25)
}


# [all] Scatter plots of drug sensitivity scores -------------------------------
plot_scores <- function(itrex_dir, plots_dir) {
  data_all <- data.frame()
  data_rep <- data.frame()

  for (sid in sids) {
    for (modality in itrex_modalities) {
      data_all <- rbind_xls(itrex_dir, data_all, sid, modality)
      data_rep <- rbind_xls(itrex_dir, data_rep, sid, modality, "_rep")
    }
  }

  data_all$sid <- factor(data_all$sid, levels = sids)
  data_rep$sid <- factor(data_rep$sid, levels = sids)

  data_return <- data_all

  # Data for expected hits
  data_exp <- data_all
  data_exp <- subset(data_exp, modality %in% c("Met", "iT2"))
  data_exp <- subset(data_exp, !Vectorize(is_patient_sid)(sid))
  data_exp$is_exp <- with(data_exp, Vectorize(which_expected)(Drug.Name, sid))
  # Disable for all non-expected
  data_exp <- subset(data_exp, as.logical(is_exp))

  # Expected hits (scores and ranks)
  data_exp_table <- full_join(
    subset(data_exp, modality == "iT2"),
    subset(data_exp, modality == "Met"),
    by = c("Drug.Name", "sid"),
    suffix = c("_cnn2", "_met"),
  )
  data_exp_table$cell_line_index <- sid_to_cell_line_index(data_exp_table$sid)
  data_exp_table <- data_exp_table[order(
    data_exp_table$cell_line_index,
    data_exp_table$is_exp_met,
    data_exp_table$sid
  ), ]
  data_exp_table$label <- paste0(
    data_exp_table$Drug.Name,
    " (", str_sub(data_exp_table$sid, -6), ")"
  )
  data_exp_table$drug_line <- paste0(
    "-- ", data_exp_table$label,
    " & ", data_exp_table$DSS_asym_cnn2,
    " & ", data_exp_table$DSS_asym_met,
    " & ", data_exp_table$dss_rank_cnn2,
    " & ", data_exp_table$dss_rank_met,
    " \\\\"
  )
  eol <- ""
  data_exp_table <- plyr::ddply(
    data_exp_table,
    plyr::.(cell_line_index),
    plyr::summarize,
    cell_line = paste0(
      cell_lines[unique(cell_line_index)],
      " & & & & \\\\\n",
      paste(drug_line, collapse = "\n")
    )
  )
  cat(
    data_exp_table$cell_line,
    sep = "\n & & & & \\\\[-0.5\\normalbaselineskip]\n",
    file = glue(plots_dir, "data_exp_table.tex")
  )

  message("Vinorelbine scores ------------------------------------------------")
  print(subset(data_all,
    select = c("sid", "Drug.Name", "DSS_asym", "modality"),
    Drug.Name == "Vinorelbine" &
      sid %in% c("INF_R_153_V2_DS1", "INF_R_153_V3_DS1") &
      modality %in% c("Met", "iT2")
  ))

  # Prepare data_rep
  data_rep <- separate(
    data = data_rep,
    col = Drug.Name,
    into = c("drug_name", "rep"),
    sep = "_rep",
    fill = "left",
  )
  data_rep_wider0 <- full_join(
    subset(data_rep, rep == "1"),
    subset(data_rep, rep == "2"),
    by = c("drug_name", "sid", "modality"),
    suffix = c("_1", "_2"),
  )
  assert(nrow(data_rep) == 2 * nrow(data_rep_wider0))
  data_rep <- data_rep_wider0

  # Remove some NA - could do before join, but then the cross-check fails
  data_rep <- subset(data_rep, !is.na(Drug.ID_1) & !is.na(Drug.ID_2))
  data_rep <- subset(data_rep, !is.na(DSS_asym_1) & !is.na(DSS_asym_2))

  # Prepare data_all
  data_all_wider0 <- full_join(
    subset(data_all, modality != "Met"),
    subset(data_all, modality == "Met"),
    by = c("Drug.Name", "sid"),
    suffix = c("", "_met"),
  )
  n_mod <- length(itrex_modalities)
  assert(nrow(data_all_wider0) == nrow(data_all) * (n_mod - 1) / n_mod)
  data_all <- data_all_wider0

  # Remove NA data (concerns only MoS data)
  assert(all(data_all$modality[is.na(data_all$DSS_asym)] == "MoS"))
  data_all <- data_all[complete.cases(data_all), ]

  # Remove some NA - could do before join, but then the cross-check fails
  data_all <- subset(data_all, !is.na(Drug.ID))
  data_all <- subset(data_all, modality != "Met")

  plot_pcc(
    plots_dir, data_rep, "DSS_asym_1", "DSS_asym_2", "DSS", "ScoresRepPearson"
  )
  plot_pcc(
    plots_dir, data_all, "DSS_asym_met", "DSS_asym", "DSS", "ScoresPearson"
  )

  vowels <- c("a", "e", "i", "o", "u")
  shorter <- substr(data_exp$Drug.Name, 5, 5) %in% vowels
  length <- ifelse(shorter, 4, 5)
  data_exp$drug <- paste0(substr(data_exp$Drug.Name, 1, length), ".")

  message("Hit mismatch statistics (vs. metabolics) --------------------------")
  data_all$hit_mismatch <-
    (data_all$DSS_asym > 10) != (data_all$DSS_asym_met > 10)
  mismatch_modality_stats <- function(x) {
    c(sum = sum(x), len = length(x), ratio = sum(x) / length(x))
  }
  mismatch_stats <- aggregate(
    data_all$hit_mismatch,
    by = list(modality = data_all$modality),
    FUN = mismatch_modality_stats
  )
  print(mismatch_stats)

  p <- ggplot(
    data_exp,
    aes(
      x = get_modality_labels(modality),
      y = DSS_asym,
      color = drug,
    )
  ) +
    geom_violin(
      fill = "grey",
      color = FALSE,
    ) +
    dss_10_y +
    # Enable for all non-expected:
    # facet_wrap("is_exp") +
    # guides(color = "none") +
    geom_point(
      size = 1,
    ) +
    geom_line(
      aes(
        group = interaction(sid, drug),
      )
    ) +
    labs(
      x = "Modality",
      y = expression(DSS[asym] ~ of ~ expected ~ hits),
      color = NULL,
    ) +
    guides(
      color = guide_legend(ncol = 2),
    ) +
    theme +
    NULL

  pdf_png(glue(plots_dir, "ScoresExpected"), p, ieee_half_width, 1.7, 1.25)

  for (plot_margin in FALSE:TRUE) {
    if (plot_margin) {
      data_rep <- get_margin_distributions(data_rep)
      data_all <- get_margin_distributions(data_all)
    }

    p <- ggplot(data_rep, aes(x = DSS_asym_1, y = DSS_asym_2)) +
      diagonal +
      dss_10_x +
      dss_10_y +
      blue_scatter_point +
      modality_sample_grid +
      r_rho(
        0, 75
      ) +
      coord_fixed(
        ratio = 1,
        xlim = c(0, 75),
        ylim = c(0, 75)
      ) +
      labs(
        x = "DSS (replicate 1)",
        y = "DSS (replicate 2)",
      ) +
      theme +
      (if (plot_margin) NULL else small_facet) +
      NULL
    f <- glue(plots_dir, "ScoresRep{plot_margin}")
    if (plot_margin) {
      pdf_png(f, p, ieee_half_width, 2, 1.25)
    } else {
      pdf_png(f, p, ieee_half_width, 8.26772, scale = 1 / 0.3)
    }

    p <- ggplot(
      data_all,
      aes(
        x = DSS_asym_met,
        y = DSS_asym,
      )
    ) +
      diagonal +
      dss_10_x +
      dss_10_y +
      blue_scatter_point +
      modality_sample_grid +
      r_rho(
        0, 75
      ) +
      coord_fixed(
        ratio = 1,
        xlim = c(0, 75),
        ylim = c(0, 75),
      ) +
      labs(
        x = glue("DSS ({tolower(get_modality_labels('Met'))})"),
        y = "DSS (image-based)",
      ) +
      theme +
      (if (plot_margin) NULL else small_facet) +
      NULL

    f <- glue(plots_dir, "Scores{plot_margin}")
    if (plot_margin) {
      pdf_png(f, p, ieee_half_width, 2, 1.25)
    } else {
      pdf_png(f, p, ieee_half_width, 8.26772, scale = 1 / 0.3)
    }
  }

  return(data_return)
}

rbind_xls <- function(itrex_dir, data, sid, modality, suff = "") {
  modality_sid <- glue("{modality}-{sid}")
  xls_file <- glue(itrex_dir, modality_sid, "{modality_sid}_mono{suff}.xlsx")
  file_data <- openxlsx::read.xlsx(xls_file)

  file_data <- file_data[, c("Drug.ID", "Drug.Name", "DSS_asym")]

  file_data$sid <- sid
  file_data$modality <- modality
  file_data$dss_rank <- rank(-file_data$DSS_asym)

  data <- rbind(data, file_data)

  return(data)
}

which_expected <- function(drug, sid) {
  which(
    drug == cell_line_expected_hits[[sid_to_cell_line_name(sid)]]
  )[1]
}

sid_to_cell_line_name <- function(sids) {
  cell_lines[sid_to_cell_line_index(sids)]
}

sid_to_cell_line_index <- function(sids) {
  sapply(sids, function(sid) which(str_detect(sid, cell_lines)))
}


# [all] Plot heatmaps of drug scores -------------------------------------------
plot_heatmap <- function(df_scores, df_drugs, plots_dir) {
  df_drugs <- df_drugs[, c("DrugName", "DisplayDrugName")]
  df_scores <- merge(
    df_scores, df_drugs,
    by.x = "Drug.Name", by.y = "DrugName"
  )
  df_scores <- subset(df_scores, select = -Drug.Name)

  df_scores <- pivot_wider(
    df_scores,
    id_cols = c("sid", "modality"),
    names_from = "DisplayDrugName",
    values_from = "DSS_asym",
  )

  heatmaps <- ComplexHeatmap::HeatmapList(direction = "vertical")
  for (modality in itrex_modalities) {
    .modality <- modality
    mod_scores <- subset(df_scores, modality == .modality, select = -modality)
    mod_scores <- column_to_rownames(mod_scores, "sid")
    h <- ComplexHeatmap::Heatmap(
      as.matrix(mod_scores),
      name = modality,
      row_title = get_modality_labels(modality),
      col = circlize::colorRamp2(
        c(0, 10, 50), c("steelblue4", "white", "tomato3"),
      ),
      row_names_gp = grid::gpar(fontsize = 5),
      column_names_gp = grid::gpar(fontsize = 5),
      heatmap_legend_param = list(title = "DSS")
    )
    grab_heatmaps <- function(heatmaps, ...) {
      R.devices::suppressGraphics(
        grid::grid.grabExpr(ComplexHeatmap::draw(heatmaps, ...))
      )
    }
    p <- grab_heatmaps(h)
    f <- glue(plots_dir, "heatmap_{modality}")
    pdf_png(f, p, ieee_full_width, 2.5)

    heatmaps <- ComplexHeatmap::add_heatmap(heatmaps, h, "vertical")
  }

  # Plot combined heatmap
  for (i in seq_along(heatmaps)) {
    heatmaps@ht_list[[i]]@matrix_legend_param$direction <- "horizontal"
    if (i == length(heatmaps)) {
      break
    }
    # Show only a single legend by disabling all but the las
    heatmaps@ht_list[[i]]@heatmap_param$show_heatmap_legend <- FALSE
  }

  p <- grab_heatmaps(heatmaps, heatmap_legend_side = "bottom")
  f <- glue(plots_dir, "heatmap")
  pdf_png(f, p, ieee_full_width, 9)
}


# [cnn] Plot t-SNE embeddings of CNN features ----------------------------------
plot_tsne <- function(df_scores, df_drugs, itrade_dir, plots_dir) { # nolint(cyclocomp_linter)

  tsne_dir <- glue(plots_dir, "tsne")
  if (!dir.exists(tsne_dir)) {
    dir.create(tsne_dir, recursive = TRUE)
  }

  # Configure group levels for different plots
  groups <- c("top", "sub", "sub_hits")

  # Prefer the more contrastive HSL palette used by plotnine:
  encode_colour <- farver::encode_colour
  # https://github.com/thomasp85/farver/blob/master/R/encode.R#L42
  encode_colour_hcl2hsl <- function(...) {
    args <- list(...)
    if (length(args$from) && args$from == "hcl") args$from <- "hsl"
    do.call(encode_colour, args)
  }
  assignInNamespace("encode_colour", encode_colour_hcl2hsl, "farver")

  for (sid in sids) {
    for (modality in itrade_modalities) {
      itrex_modality <- itrade_to_itrex(modality)
      tsne_file <- glue(
        itrade_dir, "{modality}-{sid}", "predict", "{sid}.pca_tsne.feather"
      )
      df <- arrow::read_feather(tsne_file)
      df$DrugName <- ifelse(
        !is.na(df$Layout_Treatment), df$Layout_Treatment, df$Layout_WellType
      )

      # Check and merge drugs
      tsne_drugs <- sort(unique(df$Layout_Treatment))
      all_drugs <- sort(unique(df_drugs$DrugName))
      assert(length(setdiff(tsne_drugs, all_drugs)) == 0)
      df <- merge(df, df_drugs)

      n_pc <- sum(grepl("^PCA_\\d+$", colnames(df)))

      # Mark outliers (not more than five)
      outliers <- (abs(df$tSNE_0) > 100 | abs(df$tSNE_1) > 100)
      n_out <- sum(outliers)
      if (n_out) {
        message(glue("{n_out} outliers found for {sid}/{modality}."))
      }
      assert(n_out <= 5, "More than 5 outliers found.")
      out <- if (n_out) glue("_{n_out}out") else ""

      desc <- glue("{n_pc}pc{out}")

      for (group in groups) {
        if (group == "top") {
          color_name <- "Drug Class"
          color_var <- sym("TopGroupAbbr")
          shape_var <- sym("TopGroupAbbr")
          label_var <- NA

          geom_element <- geom_point(size = 2, stroke = 3)
          scale_color <- scale_color_manual(values = c(
            #' from matplotlib import cm, colors as cl
            #' list(map(cl.rgb2hex, (m:=cm.tab10)(range(m.N))))
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
          ))
          scale_shape <- scale_shape_manual(values = c(
            "+",
            "−", # Unicode Minus (U+2212)
            # https://git.io/JOPjb
            "■", # 0, # 's', square
            "◆", # 5, # 'D', Diamond
            "▲", # 2, # '^', triangle up
            "▼", # 6, # 'v', triangle down
            "◀", #    # '<', triangle left # nolint
            "▶", #    # '>', triangle right # nolint
            "●" # 1, # 'o', circle
          ))
        } else {
          color_name <- "Drug Subclass"
          color_var <- sym("GroupAbbr")
          shape_var <- NA
          label_var <- sym("DrugAbbr")

          geom_element <- geom_text(size = 2)
          scale_color <- scale_color_hue(
            h = 3.6 + c(0, 360 * 21 / 22), l = 40,
          )
          scale_shape <- NULL
        }

        if (endsWith(group, "hits")) {
          hits <- df_scores$Drug.Name[
            df_scores$sid == sid &
              df_scores$modality == itrex_modality &
              df_scores$dss_rank <= 10
          ]
          df$hit_drug <- df$DrugName %in% as.vector(hits)
          assert(sum(df$hit_drug) == 2 * length(hits), "Hit error.")

          tsne_controls <- c("DMSO", "STS", "BzCl")
          df$hit <- df$hit_drug | c(df$DrugAbbr %in% tsne_controls)

          highlight_hits <- gghighlight::gghighlight(
            hit,
            keep_scales = TRUE, use_group_by = FALSE,
          )
        } else {
          highlight_hits <- NULL
        }

        p <- ggplot(
          df[!outliers, ],
          aes(
            x = tSNE_0,
            y = tSNE_1,
            color = !!color_var,
            shape = !!shape_var,
            label = !!label_var
          ),
        ) +
          geom_element +
          scale_color +
          scale_shape +
          labs(
            x = "t-SNE x",
            y = "t-SNE y",
            color = color_name,
            shape = "Drug Class",
          ) +
          theme +
          NULL

        # `across()` in `filter()` is deprecated, use `if_any()` or `if_all()`
        p <- withr::with_options(list(warn = 0), p + highlight_hits)

        f <- glue(tsne_dir, "{modality}_{sid}_{desc}_{group}")
        pdf_png(f, p)
        if (group == "top") {
          p1 <- p +
            labs(x = NULL) + theme(legend.justification = "left")
        } else if (group == "sub") {
          p2 <- p
        }
      }
      if (all(c("top", "sub") %in% groups)) {
        f <- glue(tsne_dir, "{modality}_{sid}_{desc}")
        cowplot::set_null_device("cairo")
        p <- cowplot::plot_grid(p1, p2, ncol = 1, align = "v")
        pdf_png(f, p, ieee_full_width, 8.8)
      }
    }
  }

  # Restore HCL for all following plots
  assignInNamespace("encode_colour", encode_colour, "farver")
}


plots <- function(data_dir = "itrade-data",
                  itrade_dir = "itrade-results",
                  itrex_dir = "itrex-results",
                  plots_dir = "plot-results") {
  unzip_itrex_results_files(itrex_dir)
  if (!dir.exists(plots_dir)) {
    dir.create(plots_dir, recursive = TRUE)
  }

  df_layout <- read_layout(data_dir)

  df_drugs <- read_drugs(data_dir)

  df_viab <- read_viabilities(df_layout, itrade_dir)
  plot_viabilities(df_viab, plots_dir) # histogram
  plot_viability_outliers(df_viab, data_dir, plots_dir) # tex figure

  plot_replicate_viabilities(itrex_dir, plots_dir) # scatter and pcc-violins

  plot_treatment_resp_controls(itrex_dir, plots_dir) # png files

  df_zprime <- read_zprime(itrex_dir)
  plot_zprime(df_zprime, plots_dir) # violins
  df_metrics <- read_metrics(itrade_dir)
  tabulate_metrics_and_zprime(df_metrics, df_zprime, plots_dir) # tex table

  df_transfer_metrics <- read_transfer_metrics(itrade_dir)
  plot_transfer_metrics(df_transfer_metrics, plots_dir) # curves

  df_scores <- plot_scores(itrex_dir, plots_dir) # scatter and pcc-violins

  plot_heatmap(df_scores, df_drugs, plots_dir) # heatmap

  plot_tsne(df_scores, df_drugs, itrade_dir, plots_dir) # tsne

  message("Done.")
  warnings()
}

# Run main function if called from Rscript or vscDebugger
if (sys.nframe() == 0 || exists(".vsc.getSession")) {
  plots()
}
