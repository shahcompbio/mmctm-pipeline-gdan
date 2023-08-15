#!/usr/bin/env Rscript

library(argparse)
library(copynumber)
library(dplyr)

set.seed(230498230)

get_args <- function() {
    p <- ArgumentParser(description="Identify clusterd SVs")

    p$add_argument("svs", help="SV tsv file")
    p$add_argument("out", help="output tsv file")

    return(p$parse_args())
}

open_file <- function(fn) {
    if (fn %in% c("-", "/dev/stdin")) {
        return(file("stdin", open="r"))
    } else if (grepl("^/dev/fd/", fn)) {
        return(fifo(fn, open="r"))
    } else {
        return(file(fn, open="r"))
    }
}

read_tsv <- function(fn, ...) {
    con <- open_file(fn)
    df <- read.delim(con, check.names=FALSE, stringsAsFactors=FALSE, ...)
    close(con)
    return(df)
}

get_breakpoints <- function(svs) {
    brk1 <- svs[, c("id", "chrom_1", "brk_1")]
    colnames(brk1) <- c("id", "chrom", "brk")

    brk2 <- svs[, c("id", "chrom_2", "brk_2")]
    colnames(brk2) <- c("id", "chrom", "brk")

    brk <- rbind(brk1, brk2)
    return(brk)
}

# from Nik-Zainal et al. Methods section:
# For each sample, both breakpoints of each rearrangement were considered individually and all breakpoints were ordered by chromosomal position. The inter-rearrangement distance, defined as the number of base pairs from one rearrangement breakpoint to the one immediately preceding it in the reference genome, was calculated. Putative regions of clustered rearrangements were identified as having an average inter-rearrangement distance that was at least 10 times greater (note: less) than the whole-genome average for the individual sample. Piecewise constant fitting parameters used were γ = 25 and kmin = 10, with γ as the parameter that controls smoothness of segmentation, and kmin the minimum number of breakpoints in a segment.
identify_clustered_svs <- function(svs) {
    svs$id <- 1:nrow(svs)
    brk <- get_breakpoints(svs)

    brk_diff <- brk %>% 
        arrange(chrom, brk) %>%
        group_by(chrom) %>% 
        mutate(brk_diff=brk - lag(brk)) %>%
        ungroup() %>%
        mutate(brk=1:n())
    brk_diff <- as.data.frame(brk_diff)

    mean_diff <- mean(brk_diff$brk_diff, na.rm=TRUE)

    cl_ids <- c()
    for (c in unique(brk_diff$chrom)) {
        chrom_brk_diff <- brk_diff[brk_diff$chrom==c, ]
        if (nrow(chrom_brk_diff) < 10) next

        segs <- pcfPlain(
            chrom_brk_diff[, c("brk", "brk_diff")],
            kmin=10, gamma=25, normalize=FALSE, verbose=FALSE
        )
        #print(segs)
        cl_segs <- segs[mean_diff/segs$mean >= 10, ]

        if (nrow(cl_segs) == 0) next

        # print(cl_segs)

        for (s in 1:nrow(cl_segs)) {
            start_mask <- chrom_brk_diff$brk >= cl_segs[s, "start.pos"]
            end_mask <- chrom_brk_diff$brk <= cl_segs[s, "end.pos"]
            seg_brk <- chrom_brk_diff[start_mask & end_mask, ]
            cl_ids <- c(cl_ids, seg_brk$id)
        }
    }
    cl_ids <- unique(cl_ids)

    svs$cl <- 0
    svs[svs$id %in% cl_ids, "cl"] <- 1
    return(svs)
}

main <- function() {
    argv <- get_args()

    svs <- read_tsv(argv$svs)
    
    if (nrow(svs) == 0) {
        svs$cl <- numeric(0)
    } else {
        svs <- identify_clustered_svs(svs)
    }

    write.table(svs, argv$out, quote=FALSE, sep="\t", row.names=FALSE)
}

main()