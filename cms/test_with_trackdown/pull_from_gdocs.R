

library(trackdown)
library(tidyverse)



fp<- "PhD_pages"

rmd_filedirs <- dir_ls(path = paste0(fp,"/rmarkdown"), glob = "*.Rmd")

# rmd_filedirs <- gsub(".Rmd","-markdown.Rmd", group$group)


download_file(gpath = paste0("joehasell.netlify.app/",fp), file = rmd_filedirs)