from .GraphEncoder import *
from .TextEncoder import *
from . import CodeCLIP as _CodeCLIP_module
from . import CodeBert as _CodeBert_module
from . import Unixcoder as _Unixcoder_module
from . import ablations

# Re-export the modules directly
CodeCLIP = _CodeCLIP_module
CodeBert = _CodeBert_module
Unixcoder = _Unixcoder_module

# Create namespace objects for ablations to mimic the module structure
class CodeCLIP_ablation_graph_only:
    from .ablations.CodeCLIP_ablation_graph_only import Pretrain, Downstream

class CodeCLIP_ablation_no_penalty:
    from .ablations.CodeCLIP_ablation_no_penalty import Pretrain, Downstream

class CodeCLIP_ablation_no_router:
    from .ablations.CodeCLIP_ablation_no_router import Downstream

__all__ = [
    'CodeCLIP',
    'CodeCLIP_ablation_graph_only',
    'CodeCLIP_ablation_no_penalty',
    'CodeCLIP_ablation_no_router',
    'CodeBert',
    'Unixcoder',
]