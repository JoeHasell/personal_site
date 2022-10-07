library(fs)
library(stringr)

fp<- "PhD_pages"

rmd_filedirs <- dir_ls(path = paste0(fp,"/rmarkdown"), glob = "*.Rmd")

qmd_filedirs <- str_replace(string = rmd_filedirs,
                         pattern = "Rmd",
                         replacement = "qmd")


qmd_filedirs <- str_replace(string = qmd_filedirs,
                            pattern = "/rmarkdown",
                            replacement = "/quarto")

file_copy(path = rmd_filedirs,
          new_path = qmd_filedirs,
          overwrite = TRUE)


