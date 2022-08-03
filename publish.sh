#!/bin/zsh

# For documentation of how to create and publish shinylive apps, see https://shiny.rstudio.com/py/docs/shinylive.html#sharing-shinylive-applications

# To create a new app run: shiny create <PATH>, where for me PATH = shiny_apps/<APP_NAME>

# Create the dirstribution for each 

folder_prefix="./"
app_prefix="./shiny_apps/"

suffix="/"

for d in ./shiny_apps/*/; do
    
    folder=${d#"$folder_prefix"}
    folder=${folder%"$suffix"}

    app=${d#"$app_prefix"}
    app=${app%"$suffix"}

    echo "$d"
    echo "$folder"
    echo "$app"

    shiny static $folder site_apps --subdir $app 

done

#shiny static shiny_apps/app1 site_apps --subdir app1   

#quarto publish netlify