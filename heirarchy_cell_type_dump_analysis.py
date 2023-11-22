# %%
import pandas as pd
from fafbseg import flywire

merged_set = pd.read_csv("data/merged_dcv_neuron_soma_sourcing_4_4_23.csv")
annotations = pd.read_csv("data/annotation_export_for_codex_310323.csv")


# %%
the_root_ids = list(merged_set["root_id"].unique())
# %%
updated_root_ids = flywire.update_ids(the_root_ids)
# %%
updated_merged_set = merged_set.merge(updated_root_ids, how="left", 
                                      left_on="root_id", right_on="old_id")
# %%
updated_merged_set_annotations=updated_merged_set.merge(annotations, how="left",
                                                        left_on="new_id", right_on="root_id")
# %%
updated_merged_set_annotations.to_csv("outputs/updated_merged_set_annotations_4_4_23.csv", index=False)
# %%
