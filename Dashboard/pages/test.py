import sys
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.entropy import entropy
import pandas as pd

seq = load_fasta_file('D:\\Studia\\Python_projects\\Genome-viz\\Data\\Neurospora crassa OR74A\\GCA_000182925.2_NC12_protein.faa')

fs = FeatureSet("")
fs.add(Feature(entropy, window=10).then(min))

ent = [{'protein_id': s.identifier.split(" ", 1)[0], 'complexity': s.data[0]} for s in fs(seq)]
df = pd.DataFrame(ent)

print(df.head(20).to_string())
