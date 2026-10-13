

Scripts:

Wavelet Packet Transforms:

This script converts each protease into a signal using the cpvh vector used and defined at the beginning of the script. We then go on to decompose  the signal up to 3 levels, then run a correlation with the decompositions of Hip1. The final output is a cross correlation score between the proteases and Hip1.

Protease Substrate Acquisition:

This script very simply accepts a set of UniProt IDs of the serine proteases we had, and first converts the UniProt IDs to Merops IDs, and then tabulates the substrates of each protease along with the substrates and their Merops IDs.

ESM2 embedding:

This script runs the PLM, which accepts the protease and substrate data and encodes it onto the drive. This data is then used by the LightGBM model to train.

LightGBM training:

This is a CPU-based decision tree model that is trained based on the encoded data provided by ESM2. This model, when trained, can then predict the cleavability of the given substrate provided a protease. The negative samples used here are the different windows within which the protease does not cleave the substrate. The positives, then, are obviously the binding windows of the proteases.


Data:
