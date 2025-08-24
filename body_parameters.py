

import torch

    
class OptimizationSMPL(torch.nn.Module):
    """
    Class used to optimize SMPL parameters.
    """
    def __init__(self, cfg: dict):
        super(OptimizationSMPL, self).__init__()

        # Get device from config
        device = cfg.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # self.pose = torch.nn.Parameter(torch.zeros(1, 72).to(device))
        # self.beta = torch.nn.Parameter((torch.zeros(1, 10).to(device)))
        # self.trans = torch.nn.Parameter(torch.zeros(1, 3).to(device))
        # self.scale = torch.nn.Parameter(torch.ones(1).to(device)*1)

        pose = torch.zeros(1, 72, device=device)
        beta = torch.zeros(1, 10, device=device)
        trans = torch.zeros(1, 3, device=device)
        scale = torch.ones(1, device=device)*1

        if "init_params" in cfg:
            init_params = cfg["init_params"]
            if "pose" in init_params:
                pose = cfg["init_params"]["pose"].to(device)
            if "shape" in init_params:
                beta = cfg["init_params"]["shape"].to(device)

            if "trans" in init_params:
                trans = cfg["init_params"]["trans"].to(device)

            if "scale" in init_params:
                scale = cfg["init_params"]["scale"].to(device)
        

        if "refine_params" in cfg:
            params_to_refine = cfg["refine_params"]
            if "pose" in params_to_refine:
                self.pose = torch.nn.Parameter(pose)
            else:
                self.pose = pose
            if "shape" in params_to_refine:
                self.beta = torch.nn.Parameter(beta)
            else:
                self.beta = beta
            if "trans" in params_to_refine:
                self.trans = torch.nn.Parameter(trans)
            else:
                self.trans = trans
            if "scale" in params_to_refine:
                self.scale = torch.nn.Parameter(scale)
            else:
                self.scale = scale
        else:
            self.pose = torch.nn.Parameter(pose)
            self.beta = torch.nn.Parameter(beta)
            self.trans = torch.nn.Parameter(trans)
            self.scale = torch.nn.Parameter(scale)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale
    


class BodyParameters():

    def __new__(cls, cfg):

        possible_model_types = ["smpl"] #["smpl", "smplx"]
        model_type = cfg["body_model"].lower()

        if model_type == "smpl":
            return OptimizationSMPL(cfg)
        # elif model_type == "smplx":
        #     return OptimizationSMPLX()
        else:
            msg = f"Model type {model_type} not defined. \
                    Possible model types are: {possible_model_types}"
            raise NotImplementedError(msg)
        

