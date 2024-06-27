"""Python script to execute MOI operations.
"""

# Standard imports
import json
import os
from pathlib import Path
import sys

# Third-party imports
import numpy as np

# Local imports
from moi.Input import Input
from moi.Integrate import Integrate
from moi.Output import Output


def get_basin_data(basin_json,index_to_run):
    """Extract reach identifiers and return dictionary.
    
    Dictionary is organized with a key of reach identifier and a value of
    SoS file as a Path object.
    """
    #index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    #index = 0

    if index_to_run == -235:
        index=int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    else:
        index=index_to_run
        print('Running offline, with index = ',index)


    with open(basin_json) as json_file:
        data = json.load(json_file)


    # ~~Error Handling~~
    # there is an issue where running on one basin causes an index error here
    # here we will check to see if the index we are looking for exists
    # this SHOULD allways exist if there is more than one reach
    # There should be a more elegant way to check the number of sets, 
    # but the data structure changes when only one set is written out.

    try:
        test_index = data[index]
        return {
            #"basin_id" : int(data[index]["basin_id"]),
            "basin_id" : data[index]["basin_id"], #hope it's ok not to have basin ids always integers?
            "reach_ids" : data[index]["reach_id"],
            "sos" : data[index]["sos"],
            "sword": data[index]["sword"]
        }
    except:
                return {
            #"basin_id" : int(data["basin_id"]),
            "basin_id" : data["basin_id"], #hope it's ok not to have basin ids always integers?
            "reach_ids" : data["reach_id"],
            "sos" : data["sos"],
            "sword": data["sword"]
        }

def get_all_sword_reach_in_basin(input,Verbose):

    # find all those that match the basin id 
    BasinLevel=len(str(input.basin_dict['basin_id']))

    # basin_reach_list_all includes all reaches in SWORD that match the current basin id
    basin_reach_list_all=[]
    for reachid in input.sword_dict['reach_id']:
        reachidstr=str(reachid)
        if reachidstr[0:BasinLevel] == str(input.basin_dict['basin_id']):
            basin_reach_list_all.append(reachid)

    if Verbose:
        print('There are a total of',len(basin_reach_list_all),'reaches in SWORD for this basin')

    # create reach_ids_all list
    nadd=0
    input.basin_dict['reach_ids_all']=[]
    for reachid in basin_reach_list_all:
        if str(reachid) not in input.basin_dict['reach_ids']:
            #if Verbose:
            #   print('reachid',reachid,'is not in basin json file, but is in SWORD.')
            nadd+=1
        input.basin_dict['reach_ids_all'].append(str(reachid))

    #if Verbose:
    #   print('Total of ',nadd, 'reaches in SWORD that were not in basin json')

    return input 

def apply_sword_patches(input,Verbose):
    # this is included here to test custom patches. 
    # not run as part of normal confluence runs
    #patch_json = Path("/home/mdurand_umass_edu/dev-confluence/mnt/").joinpath('sword_patches_v216.json')
    patch_json = Path("/Users/mtd/Analysis/SWOT/Discharge/Confluence/ohio_offline_runs/mnt/").joinpath('sword_patches_v216.json')
    with open(patch_json) as json_file:
        patch_data = json.load(json_file)

    reaches_to_patch=list(patch_data['reach_data'].keys())

    if Verbose:
        print('Read in patches for:',len(reaches_to_patch))
        print('... for reaches: ',list(reaches_to_patch))

    for reachid in reaches_to_patch:
       
        try:
            k=np.argwhere(input.sword_dict['reach_id'][:]==np.int64(reachid))
            k=k[0,0]
        except: 
            if Verbose:
                print(reachid , 'is not in this domain. not patching')
            continue

        if Verbose:
           print('Patching reach:',reachid)

        for data_element in patch_data['reach_data'][reachid]:

            if data_element != 'metadata':

                if data_element == 'n_rch_up' or data_element == 'n_rch_down':
                    data_type='scalar'
                elif data_element == 'rch_id_up' or data_element == 'rch_id_dn':
                     data_type='vector'
                else:
                     print('unknown data type found in patch! crash imminent...')

                #if Verbose:
                   #print('  Patching data element:',data_element)
                   #print('    In the patch:',patch_data['reach_data'][reachid][data_element])
                   #if data_type == 'vector':
                   #    print('    In SWORD:',input.sword_dict[data_element][:,k])
                   #elif data_type == 'scalar':
                   #    print('    In SWORD:',input.sword_dict[data_element][k])

                # apply patch
                if data_type=='vector':
                    input.sword_dict[data_element][:,k]=patch_data['reach_data'][reachid][data_element]
                elif data_type=='scalar':
                    input.sword_dict[data_element][k]=patch_data['reach_data'][reachid][data_element]

                
                #if Verbose:
                #   if data_type=='vector':
                #       print('    In SWORD after fix:',input.sword_dict[data_element][:,k])
                #   elif data_type=='scalar':
                #       print('    In SWORD after fix:',input.sword_dict[data_element][k])

    return input

def set_moi_params():

    moi_params={
        'FLPE_Uncertainty': 0.67, #default: 0.67
        'Gage_Uncertainty': 0.10, #default: 0.10
        'Fill_Uncertainty': 1.0,  #default: 1.0
        'norm': 0.5,              #default: 0.5
        'rho': 0.7,               #default: 0.7
        'niter': 4,               #default: 4
        'method':'linear',        #default: 'linear'
        'quit_before_flpe':False, #default: False
        'apply_patches': False #default: False
    }

    return moi_params


def main():

    print('basin file:',sys.argv[1])
    print('verbose flag:',sys.argv[2])
    print('branch:',sys.argv[3])
    try:
        print('index:',sys.argv[4])
    except:
        print('running on AWS index')

    # verbose 
    try: 
        VerboseFlag=sys.argv[2]
        if VerboseFlag == '-v': 
            Verbose=True
        else:
            Verbose=False
    except IndexError:
        Verbose=False

    #branch
    try:
        Branch=sys.argv[3]
    except IndexError:
        Branch='unconstrained'

    #context
    try:
        index_to_run=int(sys.argv[4]) #integer
    except IndexError:
        index_to_run=-235

    #data directories
    if index_to_run == -235 or type(os.environ.get("AWS_BATCH_JOB_ID")) != type(None):
        INPUT_DIR = Path("/mnt/data/input")
        FLPE_DIR = Path("/mnt/data/flpe")
        OUTPUT_DIR = Path("/mnt/data/output")
    else:
        #basedir=Path("/home/mdurand_umass_edu/dev-confluence/mnt/")
        basedir=Path("/Users/mtd/Analysis/SWOT/Discharge/Confluence/ohio_offline_runs/mnt")
        INPUT_DIR = basedir.joinpath("input") 
        FLPE_DIR = basedir.joinpath("flpe")
        OUTPUT_DIR = basedir.joinpath("moi")

    #basin data
    try:
        basin_json = INPUT_DIR.joinpath(sys.argv[1]) #turn this on for standard operations: AWS or running default basin file
        #basin_json = Path("/home/mdurand_umass_edu/dev-confluence/mnt/").joinpath(sys.argv[1]) #turn this on to use a local basin file
        print('Using',basin_json)           
    except IndexError:
        basin_json = INPUT_DIR.joinpath("basin.json") 

    basin_data = get_basin_data(basin_json,index_to_run)

    print('Running ',Branch,' branch.')

    print('setting moi params')
    params_dict=set_moi_params()

    input = Input(FLPE_DIR, INPUT_DIR / "sos/", INPUT_DIR / "swot", INPUT_DIR / "sword", basin_data,Branch,Verbose)
    print('Exctracting sword...')
    input.extract_sword()

    if params_dict['apply_patches']:
        print('applying patches...')
        input=apply_sword_patches(input,Verbose)

    print('getting all sword reaches in basin')
    input=get_all_sword_reach_in_basin(input,Verbose)
    print('extracting swot')
    input.extract_swot()
    print('extracting sos')
    input.extract_sos()
    print('extracting alg')
    input.extract_alg()
    
    print('integrating')
    integrate = Integrate(input.alg_dict, input.basin_dict, input.sos_dict, input.sword_dict,input.obs_dict,params_dict,Branch,Verbose)
    integrate.integrate()

    output = Output(input.basin_dict, OUTPUT_DIR, integrate.integ_dict, integrate.alg_dict, integrate.obs_dict, input.sword_dir)
    output.write_output()
    output.write_sword_output(Branch)

if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Execution time: {end - start}")
