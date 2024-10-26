import pandas as pd
import numpy as np
import scipy.stats.mstats as mstats
import os
import types
import re
from tqdm import tqdm

def parse_oqmd_chemical_formula(chemical_formula):
    '''
        Takes as input a oqmd chemical formula string, O5V3 or Co7Er12 as an example.
        Returns a dictionary, { 'O':5, 'V':3 }, { 'Co':7, 'Er':12 } as an example
    '''    
    # splitting ends with an empty string since the formula always ends with a number
    elements = re.split('\d+', chemical_formula)[:-1]
    
    # splitting starts with an empty string since the formula always starts with a letter
    numbers = re.split('[A-Za-z]+', chemical_formula)[1:]

    # make sure the parse is correct. we must have element, count pairs
    assert len(elements) == len(numbers)

    result = {}
    for i, e in enumerate(elements):
        result[e] = int(numbers[i])
        
    return result

def bias_non_negative(values):
    '''
        Given an array, find the minimum value and add that to 
        all elements of the array, plus a small epsilon of .01
    '''
    lowest = min(values)
    values = values + abs(lowest) + .01
    
    return values

class StudentPropertyMatrix:
    def __init__(self, csv):
        '''
            Initialize the StudentData obejct using a csv file containing
            the property maxtrix.
        '''
        # read the property matrix using pandas' DataFrame
        # in this DataFrame, there are columns, the same that are in the csv
        # and there are rows, one row per element, same as the csv
        self.data_df = pd.read_csv(csv)
        
        # make a copy of columns found in the csv file
        # each column is a property except 'element symbol'
        self.properties = list(self.data_df.columns)
        self.properties.remove('element symbol')
        
        # these values are sometimes negative, but need to be all positive 
        # because we calculate the geometric mean. The
        # function bias_non_negative takes an array and shifts it so that all
        # values are greater than 0.
        # preprocess the electronegativity column
        self.data_df['electronegativity_Muliken'] = bias_non_negative(
                self.data_df['electronegativity_Muliken'].values)
                
        self.data_df['electron_affinity_eV'] = bias_non_negative(
                self.data_df['electron_affinity_eV'].values)  
                
    def get_element(self, element_id):
        '''
            Given an element_id, either a string or an int, return an NameSpace
            containing all the properties needed to calculate Yuping's features
        '''
        if type(element_id) == str:
            # select the row in the DataFrame using the element symbol
            rows = self.data_df[self.data_df['element symbol'] == element_id]
        elif type(element_id) == int:
            # select the row in the DataFrame using the atomic number
            rows = self.data_df[self.data_df['atomic_number'] == element_id]
        else:
            raise ValueError("doesn't recognize element_id of type " + str(type(element_id)))
            
        if len(rows) == 0:
            raise ValueError("element_id not found in matrix " + str(element_id))
        else:
            row = rows.iloc[0]
            
        return row

class Yuping_Dataset:

    def __init__(self, property_matrix):
        '''
            takes a function that will return an element object. element
            object must contain properties in the properties dictionary
        '''
        self.prop_mat = property_matrix
        
        # This is a dictionary that maps function names to functions
        # each function takes an array and returns a floating point value
        self.quantities = [('mean', np.mean), 
                    ('geo_mean', mstats.gmean), 
                    ('stddev', np.std), 
                    ('max', np.max), 
                    ('min', np.min)]

    def get_feature_columns(self):
        feature_column_names = []
        for prop in self.prop_mat.properties:
            for name, quant in self.quantities:
                feature_column_names.append('_'.join(['f', prop, name]))

        return feature_column_names

    def from_df_to_arff_file(self, df, out_arff):
        '''
            takes a DataFrame, calculates features, and writes it out to an arff format file
            example in build_arff.py
            
            returns None
        '''
        # write header
        with open(out_arff, 'w') as oa:
            oa.write('@RELATION mofs\n\n')
            
            feature_column_names = self.get_feature_columns()
            for f_name in feature_column_names:
                    oa.write('@ATTRIBUTE %s NUMERIC\n' % f_name)
            
            if 'label' in df.columns:
                classes = set(df['label'].values)
                oa.write(('@ATTRIBUTE class {'+','.join(['%s']*len(classes))+'}\n\n')%tuple(classes))
            
            oa.write('@DATA\n')
            
            n_feats = len(feature_column_names)
            feat_format_string = ','.join(['%.4f']*n_feats)
            for i, row in tqdm(df.iterrows()):
                features = self.oqmd_forumla_to_yuping_feats(row['chemical_formula'])
                if features is None:
                    print('could not calculate features for', row['chemical_formula'])
                    continue
                oa.write(feat_format_string%tuple(features))
                if 'label' in df.columns:
                    oa.write(',')
                    label = row['label']
                    oa.write(label)
                oa.write('\n')
                
    def from_df_to_arff_file_with_filenames(self, df, out_arff):
        '''
            takes a DataFrame, calculates features, and writes it out to an arff format file
            example in build_arff.py
            
            returns None
        '''
        # write header
        good_files = []
        with open(out_arff, 'w') as oa:
            oa.write('@RELATION mofs\n\n')
            
            feature_column_names = self.get_feature_columns()
            for f_name in feature_column_names:
                    oa.write('@ATTRIBUTE %s NUMERIC\n' % f_name)
            
            classes = ['conducting', 'non-conducting']
            oa.write(('@ATTRIBUTE class {'+','.join(['%s']*len(classes))+'}\n\n')%tuple(classes))
            
            oa.write('@DATA\n')
            
            n_feats = len(feature_column_names)
            feat_format_string = ','.join(['%.4f']*n_feats)
            for i, row in tqdm(df.iterrows()):
                features = self.oqmd_forumla_to_yuping_feats(row['chemical_formula'])
                if features is None:
                    print('could not calculate features for', row['chemical_formula'])
                    continue
                oa.write(feat_format_string%tuple(features))
                oa.write(',')
                oa.write('non-conducting')
                oa.write('\n')
                good_files.append(row['filename'])
                
        return good_files
                
    def from_df_to_arff_file_no_dup(self, df, out_arff):
        '''
            takes a DataFrame, calculates features, and writes it out to an arff format file
            example in build_arff.py
            
            returns None
        '''
        # filter out duplicates that share the same chem_formula and label
        df = df.drop_duplicates(subset=['label', 'chemical_formula'])
        self.from_df_to_arff_file(df, out_arff)
                
    def oqmd_forumla_to_yuping_feats(self, chemical_formula):
        '''
            Takes as input a oqmd chemical formula string, O5V3 or Co7Er12 as an example.
            Returns a numpy.array of 45 features as defined by He2018
        '''    
        # parse the chemical formula
        element_dict = parse_oqmd_chemical_formula(chemical_formula)
        
        # for each element, extract properties]
        df = { property : list() for property in self.prop_mat.properties }
        for k in element_dict.keys():
            num_element = element_dict[k]
            
            try:
                ele = self.prop_mat.get_element(k)
            except:
                return None
                
            for prop in self.prop_mat.properties:
                df[prop] += [ele[prop]] * num_element
                        
        # for each property, find 5 statistical quantities
        feats = np.zeros((len(self.prop_mat.properties)*len(self.quantities)))
        for i, p in enumerate(self.prop_mat.properties):
            for j, (q, f) in enumerate(self.quantities):
                val = f(np.array(df[p], dtype=float))
                if np.isnan(val):
                    return None
                feats[i*len(self.quantities)+j] = val
        
        if np.isnan(np.sum(feats)):
            return None
        
        return feats

class SoLab_Dataset(Yuping_Dataset):
    def __init__(self, property_matrix):
        '''
            takes a function that will return an element object. element
            object must contain properties in the properties dictionary
        '''
        self.prop_mat = property_matrix
        
        # This is a dictionary that maps function names to functions
        # each function takes an array and returns a floating point value
        self.quantities = [('mean', np.mean), 
                    ('geo_mean', mstats.gmean), 
                    ('stddev', np.std), 
                    ('max', np.max), 
                    ('min', np.min)]

        self.excluded_features = ['f_atomic_number_min',
            'f_group_number_max', 'f_group_number_min',
            'f_period_number_min', 'f_electronegativity_Muliken_mean', 
            'f_electronegativity_Muliken_geo_mean', 'f_electronegativity_Muliken_stddev', 
            'f_electronegativity_Muliken_max', 'f_electronegativity_Muliken_min', 
            'f_melting_point_K_max', 'f_melting_point_K_min', 'f_melting_point_K_mean', 
            'f_melting_point_K_stddev', 'f_boiling_point_K_mean', 
            'f_boiling_point_K_stddev', 'f_boiling_point_K_max', 'f_boiling_point_K_min',
            'f_density_g/cm^3_at_298_K_min', 'f_electric_dipole_polarizability_10^-24*cm^3_geo_mean',
            'f_electrical_conductivity_S/m_geo_mean', 'f_electrical_conductivity_S/m_min', 
            'f_thermal_conductivity_W/(mK)_min', 'f_electrical_resistivity_m/S_geo_mean', 
            'f_electrical_resistivity_m/S_stddev', 'f_electrical_resistivity_m/S_max', 
            'f_electrical_resistivity_m/S_min']

    def get_all_column_names(self):
        feature_column_names = []
        for prop in self.prop_mat.properties:
            for name, quant in self.quantities:
                feature_column_names.append('_'.join(['f', prop, name]))

        return feature_column_names

    def get_feature_columns(self):
        feature_column_names = self.get_all_column_names()

        for excluded in self.excluded_features:
            if excluded in feature_column_names:
                feature_column_names.remove(excluded)

        return feature_column_names

    def oqmd_forumla_to_yuping_feats(self, chemical_formula):
        '''
            Takes as input a oqmd chemical formula string, O5V3 or Co7Er12 as an example.
            Returns a numpy.array of features the So Lab created
        '''    
        # parse the chemical formula
        element_dict = parse_oqmd_chemical_formula(chemical_formula)
        
        # for each element, extract properties]
        df = { property : list() for property in self.prop_mat.properties }
        for k in element_dict.keys():
            num_element = element_dict[k]
            
            try:
                ele = self.prop_mat.get_element(k)
            except:
                return None
                
            for prop in self.prop_mat.properties:
                df[prop] += [ele[prop]] * num_element
                        
        # for each property, find 5 statistical quantities
        all_names = self.get_all_column_names()
        feats = np.zeros(len(all_names))
        for i, p in enumerate(self.prop_mat.properties):
            for j, (q, f) in enumerate(self.quantities):
                val = f(np.array(df[p], dtype=float))
                if np.isnan(val):
                    return None
                feats[i*len(self.quantities)+j] = val
                
        if np.isnan(np.sum(feats)):
            return None
        
        feat_dict = {n:f for n, f in zip(all_names, feats)}
        feats = np.array([feat_dict[n] for n in self.get_feature_columns()])
        
        return feats

def from_df_to_df(yd, df):
    '''
        takes a DataFrame, calculates features, and writes it out to an csv format file

        out_csv should be a path to a file.
        
        returns None
    '''
    # step one, create the new DataFrame using appropriate column names
    feature_column_names = yd.get_feature_columns()
    
    # the new data frame must have all features + a label column
    result_df = pd.DataFrame(columns=feature_column_names+['label', 'id', 'chemical_formula'])

    for i, row in tqdm(df.iterrows()):
        chem_formula = row['chemical_formula']
        features = yd.oqmd_forumla_to_yuping_feats(chem_formula)
        if features is None:
            continue
        
        new_row = {column_name:value for column_name, value in zip(feature_column_names, features)}
        new_row['label'] = row['label']
        new_row['id'] = row['id']
        new_row['chemical_formula'] = row['chemical_formula']
        
        result_df = result_df.append(new_row, ignore_index=True)

    return result_df

def from_df_to_df2(yd, df, keep_cols=False):
    '''
        takes a DataFrame, calculates features, and writes it out to an csv format file

        out_csv should be a path to a file.
        
        returns None
    '''
    # step one, create the new DataFrame using appropriate column names
    feature_column_names = yd.get_feature_columns()
    
    # the new data frame must have all features + a label column
    result_dict = {'label':[], 'id':[], 'chemical_formula':[]}
    kept_rows = []
    features_list = []
    
    for ind, (i, row) in tqdm(enumerate(df.iterrows())):
        chem_formula = row['chemical_formula']
        features = yd.oqmd_forumla_to_yuping_feats(chem_formula)
        if features is None:
            continue
        
        features_list.append(features)
        kept_rows.append(ind)
        result_dict['label'].append(row['label'])
        result_dict['id'].append(row['id'])
        result_dict['chemical_formula'].append(row['chemical_formula'])
        
    features_mat = np.vstack(features_list)
    
    result_dict.update({name:fs for name,fs in zip(feature_column_names, features_mat.T)})
    if keep_cols:
        sub_df = df.iloc[kept_rows].copy()
        sub_df.rename(columns={'label':'sanity_label', 'id':'sanity_id', 'chemical_formula':'sanity_chemical_formula'}, inplace=True)
        result_dict.update({c:sub_df[c] for c in sub_df.columns})

    result_df = pd.DataFrame(result_dict)

    assert all([sid==id for sid, id in zip(result_df['id'], result_df['sanity_id'])])

    return result_df

def get_feature_columns(columns):
    cols = [c for c in columns if c.startswith('f_')]
    cols = sorted(cols)
    return cols