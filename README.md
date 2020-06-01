# Chimp and See Zooniverse Scripts
Series of python scripts that can be used to interact with the Chimp and See project hosted on Zoonivere

### Installing
Scripts are designed to work with python 3.6+ and pandas 1.0+ in either Windows or Linux.
Aside from this you will need access to the master Google spread sheet, the time_stamps, the Chimp and See exported files 
(subjects, classifications, and tags). These should be saved in a folder that the scripts can access.
You can install the package gspread_pandas (https://gspread-pandas.readthedocs.io/en/latest/getting_started.html) using pip 
from a terminal:

```
pip install gspread-pandas
```

### Usage
There are scripts that contain all the functions (e.g. chimp_spreadsheet_update_functions.py), and a script used to interact with 
specific tabs on the spreadsheet (e.g. chimp_spreadsheet_update.py). 

Within the chimp_spreadsheet_update.py, you will need to modify the data_directory (where the required files are), and the 
output_folder (where the updated spreadsheet will be saved to). Additionally, you may work with a single tab by commenting out 
or multiple tabs by uncommenting out the sites. 

After updating and saving the interact script to work with a specific tab, you can run it from the terminal using:

Linux:
```
python /path/to/file/chimp_spreadsheet_update.py
```

Windows:

```
python C:/path/to/file/chimp_spreadsheet_update
```

These can be modified to work with different animals (e.g. Gorillas) by simply modifying a few of lines in both scripts. 


### Authors
Colleen Stephens (colleenrstephens@gmail.com)
Nicholas Stephens (nbs49@psu.edu)
