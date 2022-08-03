#!/bin/zsh


apps_folder_name="shiny_apps"
site_folder="site_apps"


echo "What shall we call the new folder in which the app will be saved?"
read app_name


mkdir $apps_folder_name"/"$app_name

shiny create $apps_folder_name"/"$app_name"/script"