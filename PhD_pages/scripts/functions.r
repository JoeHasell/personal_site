
library(tidyverse)

# A function to drop duplicates. Because the API sometimes returns the same headcount share for different requested percentiles we get duplicate headcount/poverty line observations. Later we use this function to drop these... 

drop_dupes<- function(df, id_vars, duplicate_data_var){
    
    dedupe_vars<- c(id_vars, duplicate_data_var)

    df<- df %>%
        distinct(across(all_of(dedupe_vars)), .keep_all = TRUE)
    
return(df)

}


monotonicity_check<- function(df, id_vars, order_var, data_var, action){
  
  
  # Group, order and lag
  lag_data_var<- paste0("lag_", data_var)
  
  data_var <- sym(data_var)
  
  # df<- df %>%
  #    group_by_at(id_vars) %>% 
  #    arrange_at(order_var, .by_group = TRUE) %>%
  #    mutate({{lag_data_var}} := dplyr::lag(!!data_var, n = 1, default = NA)) %>%
  #    arrange_at(c(id_vars,order_var))
  
  df<- df %>%
    group_by_at(id_vars) %>% 
    arrange_at(order_var, .by_group = TRUE) %>%
    mutate({{lag_data_var}} := dplyr::lag(!!data_var, n = 1, default = NA)) %>%
    arrange_at(c(id_vars,order_var))
  
  lag_data_var <- sym(lag_data_var)
  
  
  if(action == "browse"){
    df<- df %>% filter(!!lag_data_var>=!!data_var)
  } else if (action == "drop"){
    df<- df %>% filter(!!lag_data_var<!!data_var)  
  } else {
    print("Action parameter must be set to 'browse' or 'drop'.")
  }
  
  df<- df %>% ungroup()
  return(df)
  
}



keep_closest<- function(df, id_vars, distance_varname, targets, 
                        tie_break_dist_up_or_down, tie_break_varname = NULL, 
                        tie_break_value = NULL){
  
#The function assumes that distance_var is unique within the id_vars or, if there
  #is a tie_break_var uniques within id_vars and tie_break_var taken together.
  
  # Example values
  # df<- pip_main_vars
  # targets<- c(1990, 2010)
  # t<- 2010
  # id_vars<- c('Entity', 'reporting_level')
  # distance_varname<- 'Year'
  # tie_break_dist_up_or_down<- "down"
  # tie_break_varname<- 'welfare_type'
  # tie_break_value<- 'income'
  
  distance_var <- sym(distance_varname)
  
  matched_to_target<- list()
  
  for(t in targets){
    
    df_target<- df %>% 
      mutate(target = t) %>%
      mutate(distance = !!distance_var-target) %>%
      mutate(abs_distance = abs(distance)) %>%
      group_by_at(id_vars) %>%
      mutate(min_distance = min(abs_distance))%>%
      filter(abs_distance == min_distance) 
    
    #Break ties on a particular var. e.g. 'Prefer income to consumption'
    

    if(!is.null(tie_break_varname)){ 
      
      
      tie_break_var<- sym(tie_break_varname) 
      
      df_target<- df_target %>%
        group_by_at(id_vars) %>%
        mutate(unique_tie_break_var_values = n_distinct(!!tie_break_var))
      
      df_target<- df_target %>% 
        filter(unique_tie_break_var_values==1 | !!tie_break_var == tie_break_var_value) %>%
        select(-unique_tie_break_var_values)
      
    }

    
    #Break ties with +ve and -ve distance. e,g, 1988 and 1992 matching to 1990
    
    
    df_target<- df_target %>%
      group_by_at(id_vars) %>%
      mutate(unique_distances = n_distinct(distance))
    
    if(tie_break_dist_up_or_down=="up"){
      
      df_target<- df_target %>% 
        filter(unique_distances==1 | distance>0)
      
    } else if (tie_break_up_or_down=="down"){
      
      df_target<- df_target %>% 
        filter(unique_distances==1 | distance<0)
      
    }
    
    df_target<- df_target %>%
      select(-unique_distances)
    
    # Here I should add a warning if there are non-unique results - i.e. if match_count>1
    # df_target<- df_target %>%
    #   group_by_at(id_vars) %>%
    #   mutate(match_count = n())
    
    
    matched_to_target[[as.character(t)]]<- df_target
    
  }
  
  # Unpack list into a single df 
  df_all_targets<- bind_rows(matched_to_target) %>%
    select(-c(distance, min_distance))
  
  return(df_all_targets)
  
}

