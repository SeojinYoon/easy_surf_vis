
# Common Libraries
import os
import sys
import subprocess
import numpy as np
import pandas as pd

# Custom Libraries
if os.getenv("easysurfvis_isRunSource"):
    sys.path.append(os.getenv("easysurfvis_source_home"))
    from cores.general_util import get_multiple_elements_in_list
else:
    from easysurfvis.cores.general_util import get_multiple_elements_in_list

# Functions
def set_afni_abin(abin_path):
    """
    set abin path
    
    :param abin_path: path of afni binnary ex) "/Users/clmn/abin/afni"
    """
    os.environ['PATH'] = os.environ['PATH'] + ":" + abin_path
        
def whereami(x, 
             y, 
             z, 
             coord = "spm", 
             atlas = None, 
             is_show_command = False, 
             is_parsing = True):
    """
    Where is the location?

    https://afni.nimh.nih.gov/pub/dist/doc/program_help/whereami.html

    :param x: (int)
    :param y: (int)
    :param z: (int)
    :param coord: (string) spm, dicom
        -meaning spm: is equal to RAS+, lpi coords
        -meaning dicom: is equal to LPS+ rai coords
        
    :param atlas: (string)
        -Haskins_Pediatric_Nonlinear_1.0
        -CA_ML_18_MNI
        -and so on...
        
    return (pd.DataFrame)
        -row: atlas info
    """
    
    if atlas != None:
        command = f"whereami {x} {y} {z} -{coord} -atlas {atlas}"
    else:
        command = f"whereami {x} {y} {z} -{coord}"
        
    if is_show_command:
        print(command)
        
    # command
    output = subprocess.check_output(command, shell=True)
    output = output.decode('utf-8').split("\n")

    if is_parsing == False:
        return output
    else:
        atlas_lines = search_stringAcrossTarget(output, 
                                                search_keys = ["Atlas"], 
                                                exclude_keys=["nearby"], 
                                                return_type = "index")
        
        search_atlas = []
        search_infos = []
        search_names = []
        for i in range(len(atlas_lines)):
            if len(atlas_lines) == 1:
                start_line_i = atlas_lines[0]
                end_line_i = search_stringAcrossTarget(output, 
                                                       search_keys = ["Please", "caution"], 
                                                       return_type = "index")[0]
            else:
                if i + 1 < len(atlas_lines):
                    # Search region name based on each atlas
                    start_line_i = atlas_lines[i]
                    end_line_i = atlas_lines[i+1] - 1
                else:
                    continue

            atlas_name = output[start_line_i].split(": ")[0].replace("Atlas ", "")
            selected_output = output[start_line_i:end_line_i]

            # search result
            search_results = search_stringAcrossTarget(selected_output, 
                                                       search_keys = ["Focus", "Within"], 
                                                       search_type = "any")

            for result in search_results:
                sp_result = result.split(":")

                info = sp_result[0].strip()
                name = sp_result[1].strip()

                search_atlas.append(atlas_name)
                search_infos.append(info)
                search_names.append(name)
            
            
        result_df = pd.DataFrame({
            "atlas" : search_atlas,
            "info" : search_infos,
            "name" : search_names,
        })

        return result_df

def search_string(target, search_keys, search_type = "any", exclude_keys = []):
    """
    Search string with keys in target
    
    :param target(str): target string
    :param keys(list - str): search key
    :param search_type(str): search type - 'any', 'all', any is 'or' condition, all is 'and' condition
    :param exclude_keys(list - str): exclude key
    
    return boolean
    """
    if search_type == "any":
        search_result = any([key in target for key in search_keys])
    elif search_type == "all":
        search_result = all([key in target for key in search_keys])
    elif search_type == "correct":
        assert(len(search_keys) == 1), "please input only one search key"
        search_result = search_keys[0] == target
    
    if exclude_keys != None and exclude_keys != []:
        exclude_result = not any([key in target for key in exclude_keys])
    
        return search_result and exclude_result
    else:
        return search_result

def search_stringAcrossTarget(targets, 
                              search_keys,
                              search_type = "any", 
                              exclude_keys = [], 
                              validation_type = None,
                              return_type = "string"):
    """
    Search string across target strings
    
    :param target(list): target string
    :param keys(str): search key
    :param search_type(str): search type - 'any', 'all', any is or condition, all is and condition
    :param exclude_keys(list - str): exclude key
    :param validation_type(File_validation): kinds of validation checking from search result
    :param return_type(string): 'string' or 'index'
    
    return list of searched string
    """
    search_results = [search_string(target = target, 
                                    search_keys = search_keys, 
                                    search_type = search_type,
                                    exclude_keys = exclude_keys) for target in targets]
    search_flags = np.array(search_results)
    indexes = np.where(search_flags == True)[0]
    result = get_multiple_elements_in_list(targets, indexes)
    
    # search validation
    if validation_type == None:
        pass
    else:
        if validation_type.value & File_validation.exist.value != 0:
            # Check the search result existed
            assert len(result) != 0, "Please check to exist file"
        if validation_type.value & File_validation.only.value != 0:
            if len(result) > 1:
                print(result)
                raise Exception("Multiple similar files")
    
    # return
    def return_func():
        if return_type == "index":
            return indexes
        elif return_type == "flag":
            return search_flags
        else:
            return result
        
    if validation_type == None:
        return return_func()
    else:        
        if validation_type.value & File_validation.only.value != 0:
            if return_type == "index":
                return indexes[0]
            elif return_type == "flag":
                return search_flags
            else:
                return result[0]
        else:
            return return_func()


    
if __name__ == "__main__":
    set_afni_abin("/Users/clmn/abin/afni")
    whereami(x = 10, y = 10, z = 10)
