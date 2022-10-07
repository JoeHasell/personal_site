
library(trackdown)


fp<- "PhD_pages"

rmd_filedirs <- dir_ls(path = paste0(fp,"/rmarkdown"), glob = "*.Rmd")


update_file(gpath = paste0("joehasell.netlify.app/",fp), file = rmd_filedirs, hide_code = TRUE)


