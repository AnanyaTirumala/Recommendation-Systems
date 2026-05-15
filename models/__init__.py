from .neumf    import NeuMF
from .diffrec  import DiffRec, LDiffRec
from .giffcf   import GiffCF
from .cfdiff   import CFDiff
from .gdmcf    import GDMCF
from .lightgcn import LightGCN

MODEL_REGISTRY = {
    "neumf":    NeuMF,
    "diffrec":  DiffRec,
    "ldiffrec": LDiffRec,
    "giffcf":   GiffCF,
    "cfdiff":   CFDiff,
    "gdmcf":    GDMCF,
    "lightgcn": LightGCN,
}
