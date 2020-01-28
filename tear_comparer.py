# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:15:52 2019

@author: Dima's_Monster
"""

import numpy as np
import cv2
import pickle

'''
This is a conceptual module meant to tackle a specific problem, and to be used as a starting point for that problem.

The module is meant to allow easy comparison between torn papers.
The goal is to match the paper tear marks to the notebook tear marks, or to the tear marks of another piece of paper
for the sake of matching halves of a paper together or matching torn notes out of a notebook to the right notebook

The class can be used for:
1) Creating a database of vectors from a database of pictures of tears
2) Comparing all vectors in the database to find closest matches
3) Comparing only a new individual image of a tear to already existing database
4) Saving database to file and loading it from file
5) Saving all found matches to file

All saved files are pythong dictionaries saved through pickle
'''
class TearCompare:
    def __init__(self, database = {}, prep_scale = (512,512)):
        self.db = database
        self.prep_scale = prep_scale
        self.matches = {}
        
    def image_prep(self, image):
        '''
        turns image of paper tear into edges which hug the tear's curvature    
        '''
        
        # resize it to certain size using cubic interpolation, turn into grayscale
        # cubic interpolation to extract larger peaks and valleys and discard smaller height differences in tear peaks and valleys
        # images are assumed to be originally wider than self.prep_scale
        image = cv2.cvtColor(cv2.resize(image, self.prep_scale, interpolation = cv2.INTER_CUBIC),cv2.COLOR_BGR2GRAY)

        # blur and thresh to rid of noise and grain
        image = cv2.blur(image, (2,2))
        _,image = cv2.threshold(image, 150,255, cv2.THRESH_BINARY)

        # get canny edges
        edges = cv2.Canny(image,10,255)
        return edges
    
    def nan_helper(self, y):
        '''Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        '''

        return np.isnan(y), lambda z: z.nonzero()[0]

    def get_edge_identity_vector(self, edges):
        '''
        turn edges which hug the curve of a tear into a normalized identity vector representing the tear
        '''
        # get all the Y positions where there is a 255 pixel value (sometimes there are duplicates)
        # do this by going over every column in the edges image, and taking out the Y coordinate of where the pixels are white
        # because some columns will have multiple white pixels in close proximity (minor noise in data), take the mean of those

        # NOTE: where there is no white pixel, the value will be NaN. This is okay since we will interpolate that
        
        positions=[]
        for col_index in range(self.prep_scale[0]):
                col = edges[:,col_index]
                positions.append(np.around(np.mean(np.where(col==255))))

        # turn the list into an array to be able to interpolate nan values
        positions = np.asarray(positions)        

        # interpolate all the nan values
        nans, non_nans= self.nan_helper(positions)
        positions[nans]= np.interp(non_nans(nans), non_nans(~nans), positions[~nans])

        # now the positions array has no nan values, and any place where there were nans was interpolated

        # get the mean value of the Y coordinate, and then measure every single other coordinate's distance from that mean value

        mean_y_coord = positions.mean()

        # now measure the distances of those and store in the final array
        final_array = []
        for i in range(self.prep_scale[0]):
            dist = positions[i] - mean_y_coord
            final_array.append(dist)
         
        final_array = np.asarray(final_array)

        # normalize final_array according to self.prep_scale height, to values between -1 and 1 at most
        final_array = final_array / self.prep_scale[0]
        
        return final_array

    def compare_with_db(self, image, entry_name, add_vector_to_db = True):
        '''
        compares identity vector of note with database of vectors, via means of linear distance
        also adds the note to the database upon identity vector extraction
        
        entry_name is the key by which you wish to store this specific edge in the database
        
        To be used when you already have a database and simply want to compare a new image to it
        '''
        
        edges = self.image_prep(image)
        id_vector = self.get_edge_identity_vector(edges)
        
        # store all distances and choose the smallest one
        dist_list=[]
        keys=[]
        for key in self.database:
            dist = abs(np.linalg.norm(id_vector - self.database[key]))
            dist_list.append(dist)
            keys.append(key)
        
        dist_array = np.asarray(dist_list)
        
        try:
            min_dist = dist_array.min()
            
            # store the smallest distance as a match
            match_index = dist_list.index(min_dist)
            # store the match key to key references (in both directions)
            self.matches[entry_name] = keys[match_index]
            self.matches[keys[match_index]] = entry_name
        except:
            # this means the database is currently empty and there's nothing to compare to
            pass
        
        if add_vector_to_db == True:
            # add current vector to database
            self.db[entry_name] = id_vector
        
        # return match name (or notice, in case database is empty)
        if len(self.db) != 0:
            return keys[match_index]
        else:
            return 'Database empty, no matches found'
            
    def build_db(self, image, entry_name):
        '''
        simply builds a database in the class instance of vectors, without making any comparisons
        
        advised to use build_db first on all images of tears in your database
        '''
        edges = self.image_prep(image)
        id_vector = self.get_edge_identity_vector(edges)
        self.db[entry_name] = id_vector
        
    def compare_db(self):
        '''
        compares every element in the database to all other elements in the database to find best matches
        '''
        
        for key1 in self.database:
            # only compare if there is no match that was already found
            if key1 not in self.matches:
                # store all distances and choose the smallest one
                dist_list=[]
                keys=[]
                for key2 in self.database:
                    # do not compare with self
                    if key1 != key2:
                        # do not compare to self
                        dist = abs(np.linalg.norm(self.database[key1] - self.database[key2]))
                        dist_list.append(dist)
                        keys.append(key)
                
                # do not store matches with self or if match already exists
            
                dist_array = np.asarray(dist_list)
                
                try:
                    min_dist = dist_array.min()
                    
                    # store the smallest distance as a match
                    match_index = dist_list.index(min_dist)
                    # store the match key to key references (in both directions)
                    self.matches[key1] = keys[match_index]
                    self.matches[keys[match_index]] = key1
                except:
                    # this means the database is currently empty and there's nothing to compare to
                    pass
        
    def load_db(self, dict_file):
        '''
        Load database into class if you already have one
        '''
        try:    
            with open(dict_file, 'rb') as f:
                self.db = pickle.load(f)
                print('database loaded')
        except:
            print('\nError: Could not load database/\nNo "{}" dictionary found\n'.format(dict_file))
            
    def save_db(self, dict_file):
        '''
        Save database to file from the class database
        '''
        # if file doesn't exist, create it
        if not os.path.exists(dict_file):
            open(dict_file, 'a').close()
        # save database to file    
        with open(dict_file, 'wb') as f:
            pickle.dump(self.db, f, pickle.HIGHEST_PROTOCOL)
    
    def save_matches(self, dict_file):
        '''
        Save matches dict to file
        '''
        # if file doesn't exist, create it
        if not os.path.exists(dict_file):
            open(dict_file, 'a').close()
        # save database to file    
        with open(dict_file, 'wb') as f:
            pickle.dump(self.db, f, pickle.HIGHEST_PROTOCOL)