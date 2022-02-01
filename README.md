# Environments

Note: some texture filepaths in xml files need to be mapped to local filesystem. 

For linux systems, try:
>$ cd <PATH_TO_DM_CONTROL_SUITE>
> 
>$ sed -i 's/<ABS_PATH_TO_DM_CONTROL>/\/home\/<USER_NAME>\/<PATH_TO_DM_CONTROL_REPO>/g' *.xml

e.g.,

>$ cd /home/charlie/git_repos/dm_control/dm_control/suite
> 
>$ sed -i 's/<ABS_PATH_TO_DM_CONTROL>/\/home\/charlie\/git_repos\/dm_control/g' *.xml
> 
>$ eog /home/charlie/git_repos/dm_control/dm_control/suite/blue.png

