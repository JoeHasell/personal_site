library(tidyverse)


# Country name standardization
map_names <- function(df, 
                      df_mapping, 
                      old_country_varname_df, 
                      old_country_varname_df_mapping,
                      new_country_varname_df_mapping,
                      new_country_varname_final) {  
  
  df<- left_join(df, df_mapping, 
                 by=setNames(old_country_varname_df_mapping,  old_country_varname_df))
  

  df<- df %>%
    select(-c(all_of(old_country_varname_df)))
  
  names(df)[names(df) == new_country_varname_df_mapping] <- new_country_varname_final
  
  return(df)
}




