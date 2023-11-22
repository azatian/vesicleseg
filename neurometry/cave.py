# %%
from starlette.config import Config
from caveclient import CAVEclient
from cloudvolume import CloudVolume
import numpy as np


# %%
#sets up CAVEClient
def set_up():
    datastack_name = "flywire_fafb_production"
    config = Config(".env")
    client_direct = CAVEclient(datastack_name, auth_token=config("CAVECLIENT_ACCESS_TOKEN"))
    return client_direct

#gets the volume of the dataset from cloud volume
def get_cloud_volume(link):
    vol = CloudVolume(link, fill_missing=True, mip=0, use_https=True)
    return vol

#get cutout in flywire space
def get_cutout(tupl, vol, xsize, ysize, zsize):
    x = tupl[0]
    y = tupl[1]
    z = tupl[2]
    img = vol.download_point(
        (x, y, z), # point in neuroglancer
        mip=0, 
        size=(xsize, ysize, zsize),
        coord_resolution=(4,4,40) # neuroglancer display resolution
    )
    raw = np.array(img[:,:,:,0], dtype=np.uint8)
    #counterclockwise rotation by 90 degrees
    rotated = np.rot90(raw).copy()
    #flip vertical
    flipped = np.flipud(rotated).copy()
    return flipped

#Gets synapse meta info from CAVEClient
def get_synapses_meta(client):
    meta_data_syn = client.annotation.get_table_metadata("synapses_nt_v1")
    return meta_data_syn

#Gets cell type meta info from CAVEClient
def get_celltypes_meta(client):
    meta_data_celltype = client.annotation.get_table_metadata("cambridge_celltypes")
    return meta_data_celltype

#Add to neurometry later 
def get_cellinfo_meta(client):
    meta_data_cellinfo = client.annotation.get_table_metadata("neuron_information_v2")
    return meta_data_cellinfo

def cellidentity_query(client, ids):
    return client.materialize.query_table('neuron_information_v2', 
                                        filter_in_dict={"pt_root_id": ids})

#Queries synapse predictions with user input for pre synaptic partners 
def synapse_query(client, pre_synaptic_partners):
    return client.materialize.query_table('synapses_nt_v1', 
                                        filter_in_dict={"pre_pt_root_id": pre_synaptic_partners})
#Queries synapse based on ID
def synapse_query_by_id(client, synapse_id):
    return client.materialize.query_table('synapses_nt_v1', 
                                        filter_in_dict={"id": synapse_id})

#Queries neuropil table based on synapse ID
def neuropil_query_by_id(client, synapse_id):
    return client.materialize.query_table('fly_synapses_neuropil', 
                                        filter_in_dict={"id": synapse_id})

#Queries nuclei segmentation based on nucleus ID
def nucleus_query_by_id(client, nucleus_id):
    return client.materialize.query_table('nuclei_v1', 
                                        filter_in_dict={"id": nucleus_id})

#Standard filters for synapse predictions i.e making sure cleft scores are at or above 50, post_pt_root_id is not 0 
#and pre synaptic partner is not the same id as the post synaptic partner
def synapse_standard_filters(syn):
    syn = syn[syn["cleft_score"] >= 50]
    syn = syn[syn["post_pt_root_id"] != 0]
    syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
    return syn

def synapse_strict_filters(syn):
    syn = syn[syn["cleft_score"] >= 75]
    syn = syn[syn["post_pt_root_id"] != 0]
    syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
    return syn


# %%
