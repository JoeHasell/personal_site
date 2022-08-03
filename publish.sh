#!/bin/zsh

# For documentation of how to create and publish shinylive apps, see https://shiny.rstudio.com/py/docs/shinylive.html#sharing-shinylive-applications

# To create a new app run: shiny create <PATH>, where for me PATH = shiny_apps/<APP_NAME>

# Create the dirstribution for each 

apps_folder_name="shiny_apps"
site_folder="site_apps"

for d in ./$apps_folder_name/*/; do
    

    folder=${d#"./"}
    folder=${folder%"/"}
    app=${folder#$apps_folder_name"/"}

    echo "$d"
    echo "$folder"
    echo "$app"

    #shiny static $folder"/script" $folder"/site"
    shiny static $folder $site_folder --subdir $app

done


netlify deploy --prod    


quarto publish netlify