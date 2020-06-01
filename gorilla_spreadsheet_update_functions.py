"""
Python 3.6+ script designed to download google spread sheets and update data in them with relevant observations from the
Chimp & See Project hosted on Zooniverse (https://www.zooniverse.org/projects/sassydumbledore/chimp-and-see).

This script relies on the package gspread_pandas and having a google api set up on the machine that you're working on.
The api only needs to be set up once:

1. In a console/terminal/command line type in "pip install --upgrade gspread-pandas"
2. Watch the video https://www.youtube.com/watch?v=2yIcNYzfzPw or follow the "Client Credentials" walk-through https://gspread-pandas.readthedocs.io/en/latest/getting_started.html

Authors: Nick Stephens (nbs49@psu.edu) and Colleen Stephens (colleenrstephens@gmail.com)

For questions relevant to the Chimp&See project, please contact Colleen Stephens.
For questions relevant to operation within a python environment, please contact Nick Stephens.

"""

print("Running gorilla spreadsheet update...\n")
__author__ = ["Nick Stephens", "Colleen Stephens"]

for a in __author__:
    print("Author: {}".format(str(a)))


###############################################################
#                                                             #
#                                                             #
#   Importing libraries to be used in the code's execution    #
#                                                             #
#                                                             #
###############################################################

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import gspread_pandas
from time import time as timer

###############################################################
#                                                             #
#                                                             #
#  Where definitions for the function are read into memory    #
#                                                             #
#                                                             #
###############################################################


def get_google_spread_sheet(spread_sheet_name, site_name):
    '''
    Function that uses gspread_pandas to get a spreadsheet from google sheets and place it into a dataframe. See https://gspread-pandas.readthedocs.io/en/latest/getting_started.html
    for more details.
    :param spread_sheet_name: The name of the google spreadsheet to pull
    :param site_name: The specific tab that you want to work with within the spread sheet.
    :return: Returns a google sheet in a pandas dataframe.
    '''

    #Make sure the the objects are strings. Useful in cases where a site name may be an integer.
    spread_sheet_name = str(spread_sheet_name)
    site_name = str(site_name)

    print("Pulling the Google sheet...")
    #Start a timer to see how long it takes.
    # start = timer()

    #Try to pull the sheet, which requires having a functioning api token.
    try:
        spread = gspread_pandas.Spread(str(spread_sheet_name))
        url = str(spread.url)
        print("Retrieved {} from:\n".format(str(spread_sheet_name)))
        print("                               ", url)

        spread.open_sheet(str(site_name))
        print("Converting to data frame...")
        df = spread.sheet_to_df(header_rows=1)
        print('Data frame size is {rows} rows and {columns} columns'.format(columns=int(df.shape[1]), rows=int(df.shape[0])))
        # end = timer()
    except:
        print("Couldn't retrieve google spread sheet {}".format(site_name))
    # elapsed = abs(start - end)
    # print("Operation took: {:10.4f} seconds".format((float(elapsed))))
    return df


def numpy_concat(*args, join_char="_"):
    """
    Function to deal with NaN values when concatenating portions of a data frame.
    :param args:
    :return: Returns an array that can be placed into a dataframe.
    """
    strs = [str(arg) for arg in args if not pd.isnull(arg)]
    return str(join_char).join(strs) if strs else np.nan

np_concat = np.vectorize(numpy_concat)

###############################################################
#                                                             #
#                                                             #
#      Where code begins to actually execute operations       #
#                                                             #
#                                                             #
###############################################################
#Sets up a timer so we can get a feel for how long the entire process takes
total_start = timer()

#Turn off the annoying pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Define a working directory and then use the os module to switch to it. This should be where all the files are on the
# hard drive (i.e. the subjects csv, timestamp csv, classifcations csv, and json tags.).
# Pathlib makes sure that the \ and / slashes are applied properly between operating systems.

#directory = pathlib.Path(r"D:/Desktop/iDiv_gorilla&See/Website/Updated website/Autopopulate Dec 2019/Input files")
directory = pathlib.Path(sys.argv[1]) #Takes the second argument from the command line.
os.chdir(directory)

# The site name (i.e. sheet name) within the google spread sheet.
#Site_name = "New Dragonfly"
Site_name = sys.argv[2] #Takes the third argument from the command line.

# The workflow ID that you want to use to subset the sheet by.
#workflow_id = '11110'
workflow_id = int(sys.argv[3]) #Takes the fourth argument from the command line.

#Directory where the files will be written to.
output_directory = pathlib.Path(sys.argv[4]) #Takes the fifth argument from the command line.

# This is the name of the sheet that you want to pull from google.
Google_SpreadSheet = "Chimp&See Chimp Video List 2019"

# Define the subjects file.
subjects_file = "chimp-and-see-subjects.csv"

# Define the video timsetamp file using the site name
video_file_name = str(Site_name).lower()
video_file_name = video_file_name.replace(" ", "-")
video_file = str(video_file_name) + "-video-timestamp.csv"

# Define the classification file.
classification_file = 'chimp-and-see-classifications.csv'

# Define the json tags file and use pandas to read them in.
tags_json = "chimp-and-see-talk-tags.json"

###############################################################
#                                                             #
#                                                             #
#      Shouldn't need to modify anything beyond this.         #
#                                                             #
#                                                             #
###############################################################
start = timer()
print("Working with {}, worklflow ID {}...".format(str(Site_name), str(workflow_id)))

# Read in all the data needed to get the new tags and newly identified gorilla subjects.
# print("Reading in subject, video time, classification, and json tags files...")

# use pandas to read in the files defined above.
subjects = pd.read_csv(str(subjects_file), low_memory=False)
print("Read in subject file {}...\n".format(str(subjects_file)))

video_time = pd.read_csv(str(video_file), low_memory=False)
print("Read in video time stamps file {}...\n".format(str(video_file)))

classification = pd.read_csv(str(classification_file), low_memory=False)
print("Read in classification file {}...\n".format(str(classification_file)))

tags = pd.read_json(str(tags_json))
print("Read in json tags {}...\n".format(str(tags_json)))

# end = timer()
# elapsed = abs(start - end)
# print("\nOperation took: {:10.4f} seconds\n".format((float(elapsed))))

# Get the google spread sheet and then replace the strange characters that were entered in on European keyboards.
# df is a common short hand for dataframe in python.
master_googs = get_google_spread_sheet(spread_sheet_name=str(Google_SpreadSheet), site_name=str(Site_name+" Gorillas"))
master_googs = master_googs.replace("´", "'")

# make the index into int
master_googs.index = master_googs.index.astype(int)

###############################################################
#                 Set up classifications                      #
###############################################################
'''
Identify which clips have gorillas in them.
This will be determined by the tag "gorilla" or by one or more "GORILLA" classifications.
'''
# start = timer()

print("\nSetting up the classifications...")

# Create a subset of the dataframe based on the workflow ID given using a boolean (==) match.
class_sub = classification[(classification['workflow_id'] == int(workflow_id))]

# This subsets the classifications further using a string match in the annotation columns.
class_sub = class_sub[(class_sub['annotations'].str.contains("GORILLA"))]

# Takes the class_sub data frame and then asks if the unique values (.value_counts()) contain the subject ids.
class_sub_mit_gorilla = class_sub[class_sub['subject_ids'].isin(class_sub['subject_ids'].value_counts()[class_sub['subject_ids'].value_counts() > 1].index)]

# Make the 'Subject ID' column an integer so it is consistent across the dataframes we are working with.
class_sub_mit_gorilla['Subject ID'] = class_sub_mit_gorilla['subject_ids'].astype(int)

# Collapse by the Subject ID using the groupby function in pandas. We are joining the shared columns by a space ' '.
class_sub_mit_gorilla = class_sub_mit_gorilla.groupby('Subject ID').agg(' '.join)

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))


###############################################################
#        Set up the tags and find the initial omits           #
###############################################################
'''
Find the subjects that need to be omitted from the tags json.
'''

# start = timer()
print("\nSetting up the tags...")

#subset the tags data frame so that there are no NA values in taggable_id
tags = tags.dropna(subset=['taggable_id'])

# Subset the tags data frame by the exact match of 'gorilla' in the name column without spaces.
tag_sub = tags[(tags['name'] == str('gorilla'))]

# Find the individuals that should not appear going forward in the tags data frame using a partial string match.
omit_df = tags[(tags['name'].str.contains('omit'))]

# Set up the index so it is the same datatype (str) as the master_googs dataframe.
omit_df['Subject ID'] = omit_df['taggable_id'].astype(int)

# Set the index to the Subject ID because it is the common feature amongst the data frames.
omit_df = omit_df.set_index('Subject ID')

# Subset the tags data frame so it only contains gorillas.
# Get the unique ids by dropping the duplicates from the subsetted dataframe. Here we are only keeping the first entry.
tag_sub_mit_gorilla = tag_sub.drop_duplicates(subset=['taggable_id'], keep='first')

# Drop those rows that have Na or NaN in the taggable id column.
tag_sub_mit_gorilla.dropna(subset=['taggable_id'], inplace=True)

# Create the 'Subject ID' column from the taggable id column, since they are the same thing. Make certain they're all strings
tag_sub_mit_gorilla['Subject ID'] = tag_sub_mit_gorilla['taggable_id'].astype(int)

# Remove the rows from the dataframe based on the omit dataframe.
# The tilda ~ means reverse, thus flipping the logic of the .isin function.
tag_sub_mit_gorilla = tag_sub_mit_gorilla[~tag_sub_mit_gorilla['Subject ID'].isin(omit_df.index)]

#Collapse the name into the the Subject ID and join common columns by a space ' '
tags_collapsed = tags.groupby('taggable_id').agg(' '.join)

# Take the subjects from the Subject ID and subset the original tags dataframe. Casting first as int, then as str.
tags_collapsed['Subject ID'] = tags_collapsed.index.astype(int)

# Pull out just those instances that match with those previously observed in subset_mit_gorilla
tags_collapsed = tags_collapsed[tags_collapsed['Subject ID'].isin(tag_sub_mit_gorilla['Subject ID'])]

# Grab only the columns we need.
tags_collapsed = tags_collapsed[['Subject ID', 'name']]

#Make sure the sole column has the right name
tags_collapsed.columns = ['Subject ID', 'Tags']

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))


###############################################################
#   Clean up the master_googs, classification, and tags       #
###############################################################
'''
Clean up the google spreadsheet, the subjects, and the tags to make sure they've omitted what they need to and 
that everything is uniform between them. 
'''

# start = timer()
print("\nCleaning up google spreadsheet, subjects, and tags...")

# Remove the rows from the data frame based on the index from the omit dataframe.
# The tilda ~ means reverse, thus flipping the logic of the .isin function.
master_googs = master_googs[~master_googs.index.isin(omit_df.index)]

# The Subject IDs in the tags, and classification data frame have to be in the Subject IDs in the subjects data frame or else they are invalid subjects, probably ones that have been deleted
# Create the 'Subject ID' column from the subject_id column, casting it as a string
# Fill in the nan values with a 0
subjects['workflow_id'] = subjects['workflow_id'].fillna(0)

# Cast it as an integer
subjects['workflow_id'] = subjects['workflow_id'].astype(int)

# Create a subset of the dataframe based on the workflow ID given using a boolean (==) match.
subjects = subjects[(subjects['workflow_id'] == int(workflow_id))]
subjects['Subject ID'] = subjects['subject_id'].astype(int)

# Make sure the tags dataframe is in line with the subjects we want.
tags_collapsed = tags_collapsed[tags_collapsed['Subject ID'].isin(subjects['Subject ID'])]

# Make sure the class dataframe is in line subjects we want.
class_sub_mit_gorilla = class_sub_mit_gorilla[class_sub_mit_gorilla.index.isin(subjects['Subject ID'])]

# Remove those that we hopefully already omitted earlier to be sure.
class_sub_mit_gorilla = class_sub_mit_gorilla[~class_sub_mit_gorilla.index.isin(omit_df.index)]

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))

####################################################################
# Build up a dataframe based on the clean tags and classifications #
####################################################################
'''
Create the new gorilla dataframe, which will be appeneded to the master google spread sheet. 
'''

# start = timer()
print("\nCreating new entries...")

# To determine the new gorilla videos we take the Subject ID column from the tags and subset by the subjects index.
new_gorilla_df = tags_collapsed[~tags_collapsed['Subject ID'].isin(subjects.index)]

# Assign the column name so we are certain they match up.
new_gorilla_df = new_gorilla_df["Subject ID"]
new_gorilla_df.columns = ["Subject ID"]

# Then we reset the index to a numerical one.
new_gorilla_df.reset_index(drop=True, inplace=True)

# Create a second new gorilla dataframe from the classifications.
new_gorilla_df2 = class_sub_mit_gorilla[~class_sub_mit_gorilla.index.isin(subjects.index)]

# Set up the Subject ID column, which is the index here.
new_gorilla_df2["Subject ID"] = new_gorilla_df2.index

# Assign the column names so we are certain they match.
new_gorilla_df2 = new_gorilla_df2["Subject ID"]
new_gorilla_df2.columns = ["Subject ID"]

# Reset the index so it is numerical.
new_gorilla_df2.reset_index(drop=True, inplace=True)

# Merge the accepted Subject IDs to populate the gorilla dataframe, which we append to master_googs later.
new_gorilla_df = pd.merge(new_gorilla_df, new_gorilla_df2, on='Subject ID', how="outer")

# Get the columns from master_googs and then use a loop to add it iteratively.
# This will make sure that all the columns present before are present after.

columns = list(master_googs.columns)
for i in columns:
    new_gorilla_df[str(i)] = "" # Populates the row cells with a blank string.

# Define the base html link that will then be tied to the Subject ID
zoon_url = "https://www.zooniverse.org/projects/sassydumbledore/chimp-and-see/talk/subjects/"

# Create the Link column with string mapping.
new_gorilla_df['Link'] = str(zoon_url) + new_gorilla_df['Subject ID'].map(str)

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))



###############################################################
#          Build new columns from the subjects file           #
###############################################################
'''
Obtain the relevant metadata from the subjects and videos files.
'''

# start = timer()
print("\nGrabbing information from the subjects file's metadata...")

# Just get the relevant new subjects by referencing the new_gorilla_df.
subjects_build = subjects[subjects['Subject ID'].isin(new_gorilla_df['Subject ID'])]

# First we reformat the metadata so it becomes somethings that pandas can parse
subjects_build["metadata"] = subjects_build["metadata"].apply(lambda x: dict(eval(x)))

# Then unpack the headers into a metadata data frame.
metadata_df = pd.concat([subjects_build["Subject ID"], subjects_build["metadata"].apply(pd.Series)], axis=1)

#Look for clip.id in the header of the metadata
if "#clip.id" in metadata_df.columns:
	
	#If it is found it will let us know. 
	print("Found clip.id")
	
	#First, fill the blanks in the clip.id side, so they don't overwrite the clip.name with NaNs later
	metadata_df["#clip.id"].fillna(metadata_df["#clip.name"], inplace=True)
	
	#If the BITRATE string is not found in (~) clip.id, update the clip.name column accordingly 
	metadata_df["#clip.name"] = metadata_df["#clip.id"][~metadata_df['#clip.id'].str.contains("BITRATE", na=True)]     
	
	#Create a temporary dataframe for the reverse split, where we take the right most underscore 
	clip_id_df = metadata_df["#clip.id"].str.rsplit('_', n=1, expand=True)
	
	#Then we overwrite the clip.id with the stuff to the left of the last underscore, which is the 0 column 
	metadata_df["#clip.id"] = clip_id_df[0]
	
	#Then we just update the brooklyn NaN NaN in clip.name with the now matching information, so we can treat it normally.
	metadata_df["#clip.name"].fillna(metadata_df["#clip.id"], inplace=True)
	
	#Clean it up
	metadata_df["#clip.name"] = metadata_df["#clip.name"].str.replace(".mp4", "")

# Split the clip by underscores into a temporary dataframe.
video_name_df = metadata_df["#clip.name"].str.split('_', expand=True)

# Make a copy so any operations done won't impact the clip start time later.
clip_start_df = video_name_df.copy()

# To get the clip start time we take the last two columns and join them with an underscore
# Because there may be different numbers of columns for each row or across sites, we do this row by row.
clip_start_df["Clip start time"] = ['_'.join(row.astype(str)) for row in clip_start_df[clip_start_df.columns[-2:]].values]

# Remove the mp4 from the column with string split at the period. File extensions aren't necessarily 3 characters.
clip_start_df["Clip start time"] = clip_start_df["Clip start time"].str.split('.', expand=True)[0]

# Create a "video_name column", where we are joining all but the last two columns for each row.
# Because there may be different numbers of columns for each row or across sites, we do this row by row.
video_name_df["video_name"] = ['_'.join(row.astype(str)) for row in video_name_df[video_name_df.columns[-2:]].values]

# Remove the file extension from the column using a string split. File extensions aren't always 3 charactes.
video_name_df["video_name"] = video_name_df["video_name"].str.split('_', expand=True)[0]

# To create the folder_name column we join the second to fourth to last columns with underscores for each row.
video_name_df["end"] = ['_'.join(row.astype(str)) for row in video_name_df[video_name_df.columns[2:-3]].values]

# Then we get the start of the name by joining together the first two columns by an underscore.
video_name_df["start"] = np_concat(video_name_df[0], video_name_df[1], join_char="_")

# Finally, create the folder name with map, with a forward slash between the "start" and the "end" column.
video_name_df["folder_name"] = video_name_df["start"].map(str) + "/" + video_name_df["end"]

# Merge them on their respective indices
video_name_df = video_name_df.merge(metadata_df, left_index=True, right_index=True)

#Get the clip start time.
video_name_df["Clip start time"] = clip_start_df["Clip start time"]

# Isolate just get relevant columns.
video_name_df = video_name_df[["Subject ID", "folder_name", "video_name", "Clip start time"]]

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))


###############################################################
#          Isolate relevant data from time_stamp file         #
###############################################################
'''
Get the relevant information along with the video_name and folder_name so it may be matched with the video metadata.
'''

# start = timer()
print("\nGrabbing information from the time stamps files...")

# So now we find the unique instances in subject_ids and taggable_id which should represent the same id.
# Remove the .avi or whatever and then update.
video_name = video_time["video_name"].str.split('.', expand=True)[0]
video_time["video_name"].update(video_name)

# Just get the columns we want.
video_time = video_time[["folder_name", "video_name", "creation.date", "time.offset"]]

# Reassign the creation.date name to file datetime
video_time.columns = ["folder_name", "video_name", "file datetime", "time.offset"]

#get the hours to subtract from time.offset, it's the first character, always 2H 0M 0S or 1H 0M 0S
video_time["time.offset"] = video_time["time.offset"].astype(str).str[0]
video_time["time.offset"] = video_time["time.offset"].astype('timedelta64[h]')


#turn file datetime into a time object
video_time["file datetime"] = pd.to_datetime(video_time["file datetime"], format='%m/%d/%Y %H:%M')


print(video_time[["file datetime", "time.offset"]])

#subtract offset time from file datetime
video_time["file datetime"] = video_time["file datetime"] - video_time["time.offset"]

video_time["file datetime"] = video_time["file datetime"].astype(str)
print(video_time)


# New we need to get the 'Card change date' column from the 'folder_name' column.
# Isolate the last section after the underscore with a reverse split with 1 instance.
video_time['Card change date'] = video_time["folder_name"].str.rsplit('_', n=1, expand=True)[1]

#Match up the video_time and metadata
clip_info = pd.merge(video_name_df, video_time, on=['folder_name', 'video_name'], how="left")

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))

###############################################################
#        Build the new gorilla clips data frame                 #
###############################################################
'''
Update the blank columns and replace the old tag information with the new tags for all rows. 

'''

# start = timer()
print("\nUpdating master Googs with new information...")

#Explicity set Subject ID as an int to ensure it updates correctly 
clip_info["Subject ID"] = clip_info["Subject ID"].astype(int)
new_gorilla_df["Subject ID"] = new_gorilla_df["Subject ID"].astype(int)

#Set the subject ID as the index in both dataframes because we don't trust the numeric index it set up. 
clip_info.set_index("Subject ID", inplace=True)
new_gorilla_df.set_index("Subject ID", inplace=True)

# Then we update the new_gorilla_df based on the information that we collected above.
new_gorilla_df.update(clip_info['Clip start time'])
new_gorilla_df.update(clip_info['Card change date'])
new_gorilla_df.update(clip_info['file datetime'])

print(new_gorilla_df['file datetime'])

# Get the stuff from the master googs
# Make certain the omits have been omitted
new_gorilla_df = new_gorilla_df[~new_gorilla_df.index.isin(omit_df.index)]

# Just get the new gorillas that we need to append to the end
new_gorilla_df = new_gorilla_df[~new_gorilla_df.index.isin(master_googs.index)]


#Combine the two dataframes.
combined_dataframe = master_googs.append(new_gorilla_df)

#This is just so it updates the card change date each time, because there was a blank
combined_dataframe.update(clip_info['Card change date'])
combined_dataframe.update(clip_info['Clip start time'])
combined_dataframe.update(clip_info['file datetime'])

if combined_dataframe.index.name == None:
    combined_dataframe.index.name = "Subject ID"

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))

###############################################################
#                     Update the tags                         #
###############################################################
print("\nUpdating tags...")
# start = timer()

#Set up the Subject ID as the index and update
tags_collapsed["Subject ID"] = tags_collapsed["Subject ID"].astype(int)
tags_collapsed = tags_collapsed.set_index('Subject ID')

#Make certain the index is an int before the update.
combined_dataframe.reset_index(drop=False, inplace=True)
combined_dataframe["Subject ID"] = combined_dataframe["Subject ID"].astype(int)
combined_dataframe = combined_dataframe.set_index('Subject ID')

combined_dataframe.update(tags_collapsed["Tags"])

#sort the entire dataframe by file datetime, then clip start time
combined_dataframe["file datetime"] = pd.to_datetime(combined_dataframe["file datetime"])
combined_dataframe.sort_values(by=['file datetime','Clip start time'], inplace=True)
print(combined_dataframe)

# end = timer()
# elapsed = abs(start - end)
# print("Operation took: {:10.4f} seconds.\n".format((float(elapsed))))

print("\nWriting out {} updated spreadsheet to {}...".format(str(Site_name),
                                                             str(output_directory)))
# start = timer()

#Join the output directory with the name using pathlib, which ensures the slashes are appropriate for the OS.
#replace the space in the site name with underscore
output_name = pathlib.Path(output_directory).joinpath(str(Site_name).replace(" ","_") + "_gorilla_spreadsheet_updated.csv")

#save as .csv
combined_dataframe.to_csv(str(output_name), sep=",", header=None)

total_end = timer()
total_elapsed = abs(total_start - total_end)
print("Running the entire script took: {:10.4f} seconds.\n".format((float(total_elapsed))))

