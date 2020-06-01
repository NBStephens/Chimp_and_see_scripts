#step 1: open a command line window. For windows computers, you can press the window key, then start typing "command line", then choose "command prompt" or whatever it says on your computer. For mac, press command+space to open spotlight search, then type "terminal", then click on "terminal".
#step 2: starting on line 37, place a # character just before any site you do not want to update
#step 3: copy the line below into the terminal, not including the # at the beginning of the line, then press enter.

# python "D:\Desktop\iDiv_Chimp&See\Website\Updated website\Autopopulate Dec 2019\leopard_spreadsheet_update.py"


"""
Python 3.6+ script to loop through chimp spreadsheet update script.
Chimp & See Project hosted on Zooniverse (https://www.zooniverse.org/projects/sassydumbledore/chimp-and-see).

This script relies on the package gspread_pandas and having a google api set up on the machine that your're working on.
This only needs to be done once.

1. In a console/terminal/command line type in "pip install --upgrade gspread-pandas"
2. Watch the video https://www.youtube.com/watch?v=2yIcNYzfzPw or follow the "Client Credentials" walk through https://gspread-pandas.readthedocs.io/en/latest/getting_started.html

authors: Colleen Stephens and Nick Stephens (nbs49@psu.edu)

For questions relevant to the Chimp and See project please contact Colleen Stephens.
For questions relevant to operation within a python environment please contact Nick Stephens.

"""

import os
import sys
import pathlib
import subprocess

#Directory where the video time_stamp, classification, subjects, and talk tags json files (i.e. input files) are at.
data_directory = r"D:\Desktop\iDiv_Chimp&See\Website\Updated website\Autopopulate Dec 2019\Input files"

#Where you want the updated output to be written to.
output_folder = r"D:\Desktop\iDiv_Chimp&See\Website\Updated website\Autopopulate Dec 2019\Output files"

#Site that you need to loop through.
sites = {
"New Dragonfly": 11110,
# "Xenon Bloom": 11077,
# "Twin Oaks": 11768
#"",
#"",
#"",
#,
}

#Location of the chimp spreadsheet update script.
leopard_spreadsheet_update_script_path = r"D:\Desktop\iDiv_Chimp&See\Website\Updated website\Autopopulate Dec 2019\leopard_spreadsheet_update_functions.py"

#Use a for loop to loop through the items in the sites list.
for key, values in sites.items():

    #Assign the item to the site object.
    site = str(key)
    workflow = int(values)

    #Use subprocess to sent the python command with the relevant commands.
    p = subprocess.Popen(["python",
                          str(leopard_spreadsheet_update_script_path),
                          str(data_directory),
                          str(site),
                          str(workflow),
                          str(output_folder)])

    #Prints the error messages if anything comes up. If there are no errors it will print (None, None)
    print(p.communicate())